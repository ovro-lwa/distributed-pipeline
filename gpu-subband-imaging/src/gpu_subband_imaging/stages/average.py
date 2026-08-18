"""Channel-average corrected data before peeling."""
from __future__ import annotations

from pathlib import Path


def average(ms: Path, out_ms: Path, chanbin: int) -> Path:
    from .casarun import run_py
    run_py(f"from casatasks import mstransform\n"
           f"mstransform(vis='{ms}', outputvis='{out_ms}', datacolumn='corrected', "
           f"chanaverage=True, chanbin={chanbin})")
    return out_ms
