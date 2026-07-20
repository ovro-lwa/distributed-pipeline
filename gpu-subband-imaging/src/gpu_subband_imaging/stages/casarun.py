"""Run CASA code in isolated subprocesses because CASA is not thread-safe."""
from __future__ import annotations

import subprocess
import sys


class CasaError(RuntimeError):
    pass


def run_py(code: str, timeout: int = 900) -> str:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=timeout)
    if p.returncode != 0:
        raise CasaError((p.stderr or p.stdout)[-800:])
    return p.stdout
