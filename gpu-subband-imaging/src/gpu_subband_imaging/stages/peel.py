"""Control the persistent Julia TTCalX peeling process."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict

log = logging.getLogger("gsi.peel")


class ZestDaemon:
    def __init__(self, julia: str, project: str, daemon_jl: str, sources: str,
                 env: Dict[str, str], zest_env: Dict[str, str]):
        """Start the daemon; `zest_env` holds its ZEST_* settings (PeelParams.daemon_env)."""
        cmd = [julia, f"--project={project}", daemon_jl, sources]
        run_env = dict(env)
        run_env.update(zest_env)
        if zest_env.get("ZEST_DEVICE") == "cpu":
            run_env.setdefault("JULIA_NUM_THREADS", "auto")
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=run_env,
                                   bufsize=1)
        self._await("DAEMON_READY")

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

    def close(self) -> None:
        try:
            if self._p.stdin:
                self._p.stdin.write("QUIT\n")
                self._p.stdin.flush()
            self._p.wait(timeout=30)
        except Exception:
            self._p.kill()
