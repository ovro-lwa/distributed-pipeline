"""Run CASA code in isolated subprocesses because CASA is not thread-safe."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class CasaError(RuntimeError):
    pass


_CASA_CONFIG = """\
logfile = "/dev/null"
telemetry_enabled = False
crashreporter_enabled = False
"""


def run_py(code: str, timeout: int = 900) -> str:
    with tempfile.TemporaryDirectory(prefix="gsi-casa-", dir="/tmp") as tmp:
        home = Path(tmp)
        casa_dir = home / ".casa"
        casa_dir.mkdir()
        (casa_dir / "config.py").write_text(_CASA_CONFIG)
        env = dict(os.environ, HOME=str(home))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout, cwd=str(home), env=env)
    if p.returncode != 0:
        raise CasaError((p.stderr or p.stdout)[-800:])
    return p.stdout
