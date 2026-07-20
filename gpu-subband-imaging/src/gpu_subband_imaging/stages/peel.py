"""Control the persistent Julia TTCalX peeling process."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict


class ZestDaemon:
    def __init__(self, julia: str, project: str, daemon_jl: str, sources: str,
                 env: Dict[str, str], peel_opts: Dict[str, str]):
        cmd = [julia, f"--project={project}", daemon_jl, sources]
        run_env = dict(env)
        run_env.update({f"ZEST_{k.upper()}": str(v) for k, v in peel_opts.items()})
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=run_env,
                                   bufsize=1)
        self._await("DAEMON_READY")

    def _await(self, token: str) -> None:
        for line in self._p.stdout:                     # type: ignore[union-attr]
            if line.startswith(token):
                return
        raise RuntimeError("zest daemon exited before ready")

    def zest(self, ms: Path) -> float:
        """Peel one MS and return the daemon's elapsed time."""
        self._p.stdin.write(str(ms) + "\n")             # type: ignore[union-attr]
        self._p.stdin.flush()                           # type: ignore[union-attr]
        for line in self._p.stdout:                      # type: ignore[union-attr]
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
