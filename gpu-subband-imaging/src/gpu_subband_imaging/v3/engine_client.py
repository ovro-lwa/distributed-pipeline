"""Worker side of workers/gsi_v3_engine.py (gsi v3 steps 4 and 6).

``EngineClient`` starts the engine in the imager venv, sends JSON requests tagged with an id
and matches the replies ("GSI_V3 <json>") to them in a reader thread, so several requests can
be in flight (the engine images one slot while the next is sent).  The robustness rules are
those of stages/peel.py ``ZestDaemon.zest_slot``:

- every request has a deadline; no reply in time -> the engine is killed (SIGKILL of our own
  child, waited for at most ``kill_wait`` s) and restarted, the request raises EngineTimeout;
- the engine exits (crash, CUDA fault) -> every request in flight raises EngineDied, the engine
  is restarted for the next request;
- a reply marked "fatal" (CUDA context errors) -> restarted as well, the request raises;
- replies of a killed engine (an older generation) are never taken for a new request.

After a restart ``on_restart(client)`` runs first (the worker re-opens the hour store there:
the engine reloads the records of its checkpoint file).  ``restarts`` counts them.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

log = logging.getLogger("gsi.v3.engine")
PREFIX = "GSI_V3 "


class EngineError(RuntimeError):
    """The engine answered with status error (the engine is still running)."""

    def __init__(self, msg: str, reply: Optional[dict] = None):
        super().__init__(msg)
        self.reply = reply or {}


class EngineDied(RuntimeError):
    """The engine exited (or was killed after a fatal error) while the request was in flight."""


class EngineFailed(EngineDied):
    """No more restarts (exhausted, or a restart / on_restart failed): every later request fails."""


class EngineTimeout(RuntimeError):
    """No reply within the deadline: the engine was killed (and is restarted on next use)."""


class _Pending:
    __slots__ = ("ev", "reply", "gen", "op")

    def __init__(self, gen: int, op: str):
        self.ev = threading.Event()
        self.reply: Optional[dict] = None
        self.gen, self.op = gen, op


class EngineClient:
    def __init__(self, python: str, engine_py: str, env: Mapping[str, str], start_timeout: float = 600.0,
                 kill_wait: float = 30.0, on_restart: Optional[Callable[["EngineClient"], None]] = None,
                 max_restarts: int = 3, stderr=None):
        self._cmd = [python, engine_py]
        self._env = dict(env)
        self._start_timeout, self._kill_wait = start_timeout, kill_wait
        self.on_restart = on_restart
        self.max_restarts = max_restarts
        self._stderr = stderr
        self._gate = threading.RLock()                # (re)start + on_restart: one thread at a time
        self._lock = threading.Lock()                 # the process handle and its stdin
        self._plock = threading.Lock()                # the pending table (the reader thread takes only this)
        self._p: Optional[subprocess.Popen] = None
        self._gen = 0
        self._pending: Dict[int, _Pending] = {}
        self._ids = itertools.count(1)
        self.restarts = 0
        self.abandoned: List[int] = []
        self.info: dict = {}
        self.closed = False
        self.failed = ""                              # set when restarts are exhausted / a restart failed
        with self._gate:
            self._start()

    # -- process --------------------------------------------------------------------------------
    @property
    def pid(self) -> Optional[int]:
        p = self._p
        return p.pid if p is not None else None

    @property
    def alive(self) -> bool:
        p = self._p
        return p is not None and p.poll() is None

    def _start(self) -> None:
        """Start the engine and wait for its ready line (caller holds _gate)."""
        with self._lock:
            self._gen += 1
            gen = self._gen
            p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                 env=self._env, bufsize=0)
            self._p = p
        ready = _Pending(gen, "start")
        with self._plock:
            self._pending[0] = ready
        threading.Thread(target=self._reader, args=(p, gen), name=f"v3-engine-{gen}", daemon=True).start()
        ok = ready.ev.wait(self._start_timeout)
        with self._plock:
            self._pending.pop(0, None)
        if not ok:
            self._kill(f"not ready within {self._start_timeout:.0f} s", p)
            raise EngineTimeout(f"gsi v3 engine not ready within {self._start_timeout:.0f} s (killed)")
        r = ready.reply or {}
        if r.get("status") != "ready":
            self._kill("did not start", p)
            raise EngineDied(f"gsi v3 engine did not start: {r.get('error', r)}")
        self.info = r
        log.info("gsi v3 engine pid %d ready (backend %s, imager %s)", p.pid, r.get("backend"), r.get("imager"))

    def _reader(self, p: subprocess.Popen, gen: int) -> None:
        buf = b""
        fd = p.stdout.fileno()                         # type: ignore[union-attr]
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode(errors="replace")
                if not s.startswith(PREFIX):
                    continue
                try:
                    r = json.loads(s[len(PREFIX):])
                except ValueError:
                    log.warning("V3_ENGINE bad reply line: %s", s[:200])
                    continue
                rid = r.get("id")
                with self._plock:
                    pend = self._pending.get(rid)
                    if pend is None or pend.gen != gen or pend.ev.is_set():
                        log.warning("V3_ENGINE_STALE_REPLY id=%s skipped: %s", rid, s[:200])
                        continue
                    pend.reply = r
                    pend.ev.set()
        rc = p.wait()
        self._fail_pending(gen, f"engine exited (rc={rc})")
        if rc not in (0, None):
            log.error("V3_ENGINE_EXIT pid=%d rc=%s", p.pid, rc)

    def _fail_pending(self, gen: int, why: str) -> None:
        with self._plock:
            for rid, pend in list(self._pending.items()):
                if pend.gen == gen and not pend.ev.is_set():
                    pend.reply = {"id": rid, "status": "died", "error": why}
                    pend.ev.set()

    def _kill(self, why: str, p: Optional[subprocess.Popen] = None) -> None:
        """SIGKILL the engine (our own child, by its Popen) and fail what it had in flight."""
        with self._lock:
            p = p or self._p
            if p is None:
                return
            if self._p is p:
                self._p = None
            gen = self._gen
        log.error("V3_ENGINE_KILL pid=%d: %s", p.pid, why)
        try:
            p.kill()
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=self._kill_wait)
        except subprocess.TimeoutExpired:
            self.abandoned.append(p.pid)
            log.critical("V3_ENGINE_ABANDONED pid=%d did not exit %.0f s after SIGKILL", p.pid, self._kill_wait)
        self._fail_pending(gen, f"engine killed: {why}")

    def restart(self, why: str) -> None:
        """Kill now; the next request starts a new engine (after on_restart)."""
        self._kill(why)

    def _ensure(self) -> None:
        """A running engine for the next request; a restart runs on_restart first."""
        with self._gate:
            if self.failed:
                raise EngineFailed(f"gsi v3 engine failed for good: {self.failed}")
            if self.alive:
                return
            if self.closed:
                raise EngineDied("gsi v3 engine client is closed")
            if self.restarts >= self.max_restarts:
                self.failed = f"died {self.restarts + 1} times"
                log.critical("V3_ENGINE_FAILED: died %d times; giving up", self.restarts + 1)
                raise EngineFailed(f"gsi v3 engine died {self.restarts + 1} times; giving up")
            self.restarts += 1
            log.error("V3_ENGINE_RESTART #%d", self.restarts)
            try:
                self._start()
                if self.on_restart is not None:
                    self.on_restart(self)           # re-entrant: its requests pass the gate
            except BaseException as e:              # a restarted engine that cannot serve is no engine
                self.failed = f"restart #{self.restarts} failed: {e}"
                log.critical("V3_ENGINE_FAILED: restart #%d failed: %s", self.restarts, e)
                self._kill(f"restart failed: {e}")
                raise EngineFailed(f"gsi v3 engine restart failed: {e}") from e

    # -- requests -----------------------------------------------------------------------------------
    def submit(self, op: str, **kw) -> "_Request":
        """Send one request; returns a handle whose ``result(timeout)`` waits for the reply."""
        self._ensure()
        with self._lock:
            p = self._p
            if p is None:
                raise EngineDied("gsi v3 engine is not running")
            rid = next(self._ids)
            pend = _Pending(self._gen, op)
            with self._plock:
                self._pending[rid] = pend
            try:
                p.stdin.write((json.dumps(dict(kw, id=rid, op=op)) + "\n").encode())    # type: ignore[union-attr]
                p.stdin.flush()                                                       # type: ignore[union-attr]
                err = None
            except (BrokenPipeError, OSError, AttributeError, ValueError) as e:
                err = e
        if err is not None:
            with self._plock:
                self._pending.pop(rid, None)
            self._kill(f"stdin write failed: {err}", p)
            raise EngineDied(f"gsi v3 engine does not take requests: {err}") from err
        return _Request(self, rid, pend)

    def request(self, op: str, timeout: Optional[float] = 600.0, **kw) -> dict:
        return self.submit(op, **kw).result(timeout)

    def _finish(self, rid: int, pend: _Pending, timeout: Optional[float]) -> dict:
        ok = pend.ev.wait(timeout)
        with self._plock:
            self._pending.pop(rid, None)
        if not ok:
            if pend.gen == self._gen:
                self._kill(f"no reply to {pend.op} id={rid} within {timeout:.0f} s")
            raise EngineTimeout(f"gsi v3 engine: no reply to {pend.op} within {timeout:.0f} s "
                                f"(engine killed; restarted for the next request)")
        r = pend.reply or {}
        if r.get("status") == "died":
            raise EngineDied(f"gsi v3 engine: {pend.op} id={rid}: {r.get('error')}")
        if r.get("fatal"):
            if pend.gen == self._gen:
                self._kill(f"fatal error in {pend.op}: {str(r.get('error'))[:300]}")
            raise EngineDied(f"gsi v3 engine: fatal error in {pend.op}: {r.get('error')}")
        if r.get("status") != "ok":
            raise EngineError(f"gsi v3 engine {pend.op}: {r.get('error', r.get('status'))}", r)
        return r

    def close(self, timeout: float = 120.0) -> None:
        if self.closed:
            return
        if not self.alive:
            self.closed = True
            return
        try:
            self.submit("close").result(timeout)
        except Exception as e:
            log.warning("gsi v3 engine close: %s", e)
        with self._lock:
            p = self._p
        if p is not None:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill("did not exit after close", p)
        with self._lock:
            if self._p is p:
                self._p = None
        self.closed = True


class _Request:
    def __init__(self, client: EngineClient, rid: int, pend: _Pending):
        self.client, self.id, self._pend = client, rid, pend

    def done(self) -> bool:
        return self._pend.ev.is_set()

    def result(self, timeout: Optional[float] = 600.0) -> dict:
        return self.client._finish(self.id, self._pend, timeout)


def engine_py() -> str:
    return str(Path(os.environ.get("GSI_WORKERS", Path(__file__).resolve().parents[3] / "workers"))
               / "gsi_v3_engine.py")
