"""Control the persistent Julia TTCalX peeling process."""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("gsi.peel")


class ZestTimeout(RuntimeError):
    """The daemon gave no ZESTED/FAILED reply in time; it was killed and restarted."""


class ZestDaemon:
    def __init__(self, julia: str, project: str, daemon_jl: str, sources: str,
                 env: Dict[str, str], zest_env: Dict[str, str]):
        """Start the daemon; `zest_env` holds its ZEST_* settings (PeelParams.daemon_env)."""
        cmd = [julia, f"--project={project}", daemon_jl, sources]
        run_env = dict(env)
        run_env.update(zest_env)
        if zest_env.get("ZEST_DEVICE") == "cpu":
            run_env.setdefault("JULIA_NUM_THREADS", "auto")
        self._cmd, self._run_env = cmd, run_env
        self._lines: Optional["queue.Queue[Optional[str]]"] = None   # set by zest_slot
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        self._p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=self._run_env,
                                   bufsize=1)
        self._lines = None
        self._await("DAEMON_READY")

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

    def zest(self, ms: Path) -> float:
        """Peel one MS and return the daemon's elapsed time."""
        if self._lines is not None:                     # this daemon also served slots
            return self._reply(str(ms), str(ms), None)
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
    # The replies are read by a thread into a queue, so a reply can have a deadline
    # (body.tex P1: a stalled CUDA call used to hang the batch until its timeout).

    @staticmethod
    def _reader(p: subprocess.Popen, q: "queue.Queue[Optional[str]]") -> None:
        for line in p.stdout:                           # type: ignore[union-attr]
            q.put(line)
        q.put(None)

    def _reply(self, command: str, name: str, timeout: Optional[float]) -> float:
        if self._lines is None:
            self._lines = queue.Queue()
            threading.Thread(target=self._reader, args=(self._p, self._lines),
                             name="zest-stdout", daemon=True).start()
        q = self._lines
        self._p.stdin.write(command + "\n")             # type: ignore[union-attr]
        self._p.stdin.flush()                           # type: ignore[union-attr]
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else deadline - time.monotonic()
            try:
                if left is not None and left <= 0:
                    raise queue.Empty
                line = q.get(timeout=left)
            except queue.Empty:
                self._kill_restart(f"no reply for {name} within {timeout:.0f} s")
                raise ZestTimeout(f"zest timeout after {timeout:.0f} s: {name} "
                                  f"(daemon killed and restarted)")
            if line is None:
                rc = self._p.wait()
                self._kill_restart(f"daemon exited (rc={rc}) while peeling {name}")
                raise RuntimeError(f"zest daemon closed unexpectedly (rc={rc}) on {name}")
            if line.startswith(("PEEL_", "ZEST_TIMING", "UVW_GEOMETRIC")):
                log.info(line.strip())
                continue
            if line.startswith("ZESTED "):
                return float(line.split()[2])
            if line.startswith("FAILED "):
                raise RuntimeError(f"zest failed: {line.strip()}")

    def _kill_restart(self, why: str) -> None:
        """Kill this daemon (our own child, by its Popen) and start a fresh one."""
        log.error("ZEST_DAEMON_RESTART pid=%d: %s", self._p.pid, why)
        try:
            self._p.kill()
        except ProcessLookupError:
            pass
        self._p.wait()
        self.restarts += 1
        self._start()

    def zest_slot(self, slot: Path, timeout: Optional[float] = 600.0) -> float:
        """Peel one shm slot in place ("SHM <slot>"); returns the daemon's elapsed time.

        Without a ZESTED/FAILED reply within `timeout` seconds the daemon is killed and
        restarted and ZestTimeout is raised.  The slot is then still PREPPED (killed before
        the write-back; the slot can be peeled again) or FREE (killed during it; the slot
        contents are not valid).  None waits forever.
        """
        return self._reply(f"SHM {slot}", str(slot), timeout)

    def close(self) -> None:
        try:
            if self._p.stdin:
                self._p.stdin.write("QUIT\n")
                self._p.stdin.flush()
            self._p.wait(timeout=30)
        except Exception:
            self._p.kill()
