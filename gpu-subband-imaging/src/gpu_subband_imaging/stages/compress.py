"""Compress snapshot FITS with CFITSIO `fpack`."""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List


@lru_cache(maxsize=1)
def _fpack_cmd() -> List[str]:
    env = os.environ.get("FPACK_BIN")
    if env:
        return [env]
    if shutil.which("fpack"):
        return ["fpack"]
    return ["conda", "run", "-n", "development", "fpack"]


def fpack(fits: Path) -> Path:
    """Return the compressed path and remove the source FITS on success."""
    subprocess.run(_fpack_cmd() + ["-v", str(fits)], check=True, capture_output=True)
    fz = fits.with_suffix(fits.suffix + ".fz")
    if not fz.exists():
        raise FileNotFoundError(f"fpack produced no {fz}")
    fits.unlink()
    return fz
