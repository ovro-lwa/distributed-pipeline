"""Create dirty Stokes I/V snapshots with WSClean."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from ..config import ImagingGeom, ImagingParams


def image(ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
          threads: int, wsclean_bin: str = "wsclean") -> List[Path]:
    cmd = [wsclean_bin, "-j", str(threads), "-pol", "IV", "-niter", "0",
           "-mem", "20", "-no-dirty", "-no-update-model-required",
           "-size", str(geom.pixels), str(geom.pixels),
           "-scale", f"{geom.scale}",
           "-weight", p.weight, str(p.robust),
           "-taper-inner-tukey", str(p.taper_inner_tukey),
           "-name", str(out_prefix), str(ms)]
    subprocess.run(cmd, check=True, capture_output=True)
    return [Path(f"{out_prefix}-{pol}-image.fits") for pol in ("I", "V")]
