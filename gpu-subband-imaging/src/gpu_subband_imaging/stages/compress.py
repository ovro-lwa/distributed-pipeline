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


def fpack_args(deterministic: bool = False) -> List[str]:
    """fpack's options.  Default: quantization level 4 with the dither seed from the system clock (fpack's own
    default; the pixel values of two runs on the same image differ in the dither).  deterministic
    (pipeline.yaml fpack_deterministic): `-qt 4`, the same level with the seed from the first tile's
    checksum, so the same image always gives the same compressed pixels."""
    return (["-qt", "4"] if deterministic else []) + ["-v"]


def fpack(fits: Path, deterministic: bool = False) -> Path:
    """Return the compressed path and remove the source FITS on success."""
    subprocess.run(_fpack_cmd() + fpack_args(deterministic) + [str(fits)], check=True, capture_output=True)
    fz = fits.with_suffix(fits.suffix + ".fz")
    if not fz.exists():
        raise FileNotFoundError(f"fpack produced no {fz}")
    fits.unlink()
    return fz
