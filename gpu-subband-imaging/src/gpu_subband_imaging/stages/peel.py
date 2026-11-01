"""Control the persistent Julia TTCalX peeling process."""
from __future__ import annotations

import itertools
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("gsi.peel")


class ZestTimeout(RuntimeError):
    """The daemon gave no reply in time; it was killed and restarted."""


class ZestDaemon:
    def __init__(self, julia: str, project: str, daemon_jl: str, sources: str,
                 env: Dict[str, str], zest_env: Dict[str, str],
                 start_timeout: Optional[float] = None, kill_wait: float = 30.0):
        """Start the daemon; `zest_env` holds its ZEST_* settings (PeelParams.daemon_env).

        `start_timeout` (s, None = wait forever as v2) bounds the wait for DAEMON_READY, also
        on restarts; `kill_wait` bounds the wait for a killed daemon to exit.
        """
        cmd = [julia, f"--project={project}", daemon_jl, sources]
        run_env = dict(env)
        run_env.update(zest_env)
        if zest_env.get("ZEST_DEVICE") == "cpu":
            run_env.setdefault("JULIA_NUM_THREADS", "auto")
        self._cmd, self._run_env = cmd, run_env
        self._start_timeout, self._kill_wait = start_timeout, kill_wait
        self._lines: Optional["queue.Queue[Optional[str]]"] = None   # set by zest_slot
        self._ids = itertools.count(1)
        self.restarts = 0
        self.abandoned: List[int] = []     # pids that did not exit after SIGKILL (D state)
        self._start()

    def _start(self) -> None:
        self._p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=self._run_env,
                                   bufsize=1)
        self._lines = None
        if self._start_timeout is None:
            self._await("DAEMON_READY")
        else:
            self._await_deadline("DAEMON_READY", self._start_timeout)

    @property
    def pid(self) -> int:
        return self._p.pid

    def _await(self, token: str) -> None:
        tail: list = []
        for line in self._p.stdout:                     # type: ignore[union-attr]
            if line.startswith(token):
                return
            tail = (tail + [line.rstrip()])[-15:]
        raise RuntimeError("zest daemon exited before ready (rc=%s):\n%s"
                           % (self._p.wait(), "\n".join(tail)))

    def _await_deadline(self, token: str, timeout: float) -> None:
        """_await in a thread that stops reading at `token`, so stdout is free afterwards."""
        result: "queue.Queue[Optional[BaseException]]" = queue.Queue()

        def run():
            try:
                self._await(token)
                result.put(None)
            except BaseException as e:                  # handed to the caller
                result.put(e)

        threading.Thread(target=run, name="zest-start", daemon=True).start()
        try:
            err = result.get(timeout=timeout)
        except queue.Empty:
            self._kill(f"not ready within {timeout:.0f} s")
            raise RuntimeError(f"zest daemon not ready within {timeout:.0f} s (killed)")
        if err is not None:
            raise err

    def zest(self, ms: Path) -> float:
        """Peel one MS and return the daemon's elapsed time."""
        if self._lines is not None:                     # this daemon also served slots
            return self._reply(str(ms), None, None, ms=str(ms))
        self._p.stdin.write(str(ms) + "\n")             # type: ignore[union-attr]
        self._p.stdin.flush()                           # type: ignore[union-attr]
        for line in self._p.stdout:                      # type: ignore[union-attr]
            if line.startswith(("PEEL_", "ZEST_TIMING", "UVW_GEOMETRIC")):
                log.info(line.strip())
                continue
            if line.startswith("ZESTED "):
                return float(line.split()[2])
            if line.startswith("FAILED "):
                raise RuntimeError(f"zest failed: {line.strip()}")
        raise RuntimeError("zest daemon closed unexpectedly")

    # -- gsi v3: peel a /dev/shm slot (v3/slots.py) in place -------------------------------
    # Replies are read by a thread into a queue, so a reply can have a deadline (body.tex P1).
    # Slot commands carry an id ("SHM <slot> <id>", "RESET_WARM <id>"); the daemon echoes it
    # (ZESTED <slot> <s> id=<id> / FAILED <slot> id=<id> ... / RESET_WARM_OK id=<id>), and a
    # reply with another id (left over from an earlier command) is skipped.

    @staticmethod
    def _reader(p: subprocess.Popen, q: "queue.Queue[Optional[str]]") -> None:
        for line in p.stdout:                           # type: ignore[union-attr]
            q.put(line)
        q.put(None)

    def _ensure_reader(self) -> "queue.Queue[Optional[str]]":
        if self._lines is None:
            self._lines = queue.Queue()
            threading.Thread(target=self._reader, args=(self._p, self._lines),
                             name="zest-stdout", daemon=True).start()
        return self._lines

    def _send(self, command: str) -> None:
        """Write one command; a daemon that died since the last one is restarted (once)."""
        for attempt in (1, 2):
            rc = self._p.poll()
            if rc is not None:
                self._kill_restart(f"daemon had exited (rc={rc}) before {command.split()[0]}")
            try:
                self._p.stdin.write(command + "\n")     # type: ignore[union-attr]
                self._p.stdin.flush()                   # type: ignore[union-attr]
                return
            except (BrokenPipeError, OSError) as e:
                if attempt == 2:
                    raise RuntimeError(f"zest daemon does not take commands: {e}") from e
                self._kill_restart(f"stdin write failed: {e}")

    def _reply(self, command: str, rid: Optional[int], timeout: Optional[float],
               ms: Optional[str] = None, ok_token: str = "ZESTED") -> float:
        self._send(command)
        q = self._ensure_reader()
        tag = None if rid is None else f"id={rid}"
        name = ms or command
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else deadline - time.monotonic()
            try:
                if left is not None and left <= 0:
                    raise queue.Empty
                line = q.get(timeout=left)
            except queue.Empty:
                self._kill_restart(f"no reply to {name} within {timeout:.0f} s")
                raise ZestTimeout(f"zest timeout after {timeout:.0f} s: {name} "
                                  f"(daemon killed and restarted)")
            if line is None:
                rc = self._p.wait()
                self._kill_restart(f"daemon exited (rc={rc}) during {name}")
                raise RuntimeError(f"zest daemon closed unexpectedly (rc={rc}) on {name}")
            if line.startswith(("PEEL_", "ZEST_TIMING", "UVW_GEOMETRIC")):
                log.info(line.strip())
                continue
            if line.startswith((ok_token + " ", "FAILED ")):
                parts = line.split()
                mine = (tag in parts) if tag is not None else (len(parts) > 1 and parts[1] == ms)
                if not mine:
                    log.warning("ZEST_STALE_REPLY skipped while waiting for %s: %s",
                                name, line.strip()[:300])
                    continue
                if line.startswith("FAILED "):
                    raise RuntimeError(f"zest failed: {line.strip()}")
                return float(parts[2]) if ok_token == "ZESTED" else 0.0

    def _kill(self, why: str) -> None:
        """SIGKILL this daemon (our own child, by its Popen) and wait at most kill_wait s."""
        log.error("ZEST_DAEMON_KILL pid=%d: %s", self._p.pid, why)
        try:
            self._p.kill()
        except ProcessLookupError:
            pass
        try:
            self._p.wait(timeout=self._kill_wait)
        except subprocess.TimeoutExpired:
            # e.g. stuck in a CUDA driver call (D state): it cannot be reaped now
            self.abandoned.append(self._p.pid)
            log.critical("ZEST_DAEMON_ABANDONED pid=%d did not exit %.0f s after SIGKILL "
                         "(D state?); starting a new daemon anyway", self._p.pid, self._kill_wait)

    def _kill_restart(self, why: str) -> None:
        self._kill(why)
        self.restarts += 1
        log.error("ZEST_DAEMON_RESTART #%d", self.restarts)
        self._start()

    def zest_slot(self, slot: Path, timeout: Optional[float] = 600.0) -> float:
        """Peel one shm slot in place ("SHM <slot> <id>"); returns the daemon's elapsed time.

        Without a reply within `timeout` seconds the daemon is killed and restarted and
        ZestTimeout is raised.  The slot is then still PREPPED (killed before the write-back;
        the slot can be peeled again) or FREE (killed during it; the slot contents are not
        valid).  None waits forever.
        """
        rid = next(self._ids)
        return self._reply(f"SHM {slot} {rid}", rid, timeout, ms=str(slot))

    def reset_warm(self, timeout: Optional[float] = 60.0) -> None:
        """Forget the warm-start gains (as the daemon does itself after FAILED)."""
        rid = next(self._ids)
        self._reply(f"RESET_WARM {rid}", rid, timeout, ok_token="RESET_WARM_OK")

    def close(self) -> None:
        try:
            if self._p.stdin:
                self._p.stdin.write("QUIT\n")
                self._p.stdin.flush()
            self._p.wait(timeout=30)
        except Exception:
            self._p.kill()
            try:
                self._p.wait(timeout=self._kill_wait)
            except subprocess.TimeoutExpired:
                log.critical("ZEST_DAEMON_ABANDONED pid=%d at close", self._p.pid)
