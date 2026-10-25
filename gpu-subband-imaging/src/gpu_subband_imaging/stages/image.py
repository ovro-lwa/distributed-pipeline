"""Create Stokes I/V snapshots with WSClean.

mode=dirty: dirty I and V from one ``-pol IV -niter 0`` run.
mode=clean: CLEAN Stokes I with the main pipeline's SNAPSHOT_CLEAN_I_PARAMS and a
separate dirty Stokes V run (the main pipeline's pilot SNAPSHOT_PARAMS, ``-pol V``).
Both write ``<prefix>-I-image.fits`` and ``<prefix>-V-image.fits``.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ImagingGeom, ImagingParams

log = logging.getLogger("gsi.image")


def _geom_args(geom: ImagingGeom) -> List[str]:
    return ["-size", str(geom.pixels), str(geom.pixels), "-scale", f"{geom.scale}"]


def _mem_args(p: ImagingParams) -> List[str]:
    return ["-abs-mem", f"{p.abs_mem:g}"] if p.abs_mem else []


def _weight_args(p: ImagingParams) -> List[str]:
    args = ["-weight", p.weight, str(p.robust)]
    if p.taper_inner_tukey:
        args = ["-taper-inner-tukey", str(p.taper_inner_tukey)] + args
    return args


def dirty_iv_cmd(ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
                 threads: int, wsclean_bin: str = "wsclean") -> List[str]:
    return ([wsclean_bin, "-j", str(threads), "-pol", "IV", "-niter", "0",
             "-mem", str(p.dirty_mem)] + _mem_args(p) + ["-no-dirty", "-no-update-model-required"]
            + _geom_args(geom) + _weight_args(p) + ["-name", str(out_prefix), str(ms)])


def clean_i_cmd(ms: Path, name: str, geom: ImagingGeom, p: ImagingParams,
                threads: int, wsclean_bin: str = "wsclean") -> List[str]:
    """orca SNAPSHOT_CLEAN_I_PARAMS with the band's -size/-scale."""
    cmd = [wsclean_bin, "-j", str(threads), "-pol", "I",
           "-niter", str(p.clean_niter), "-mgain", str(p.clean_mgain),
           "-auto-mask", str(p.clean_auto_mask),
           "-auto-threshold", str(p.clean_auto_threshold)]
    if p.clean_local_rms:
        cmd.append("-local-rms")
    if p.clean_horizon_mask:
        cmd += ["-horizon-mask", p.clean_horizon_mask]
    cmd += ["-mem", str(p.clean_mem)] + _mem_args(p) + ["-no-dirty"]
    return (cmd + _geom_args(geom) + _weight_args(p)
            + ["-no-update-model-required", "-name", name, str(ms)])


def dirty_v_cmd(ms: Path, name: str, geom: ImagingGeom, p: ImagingParams,
                threads: int, wsclean_bin: str = "wsclean") -> List[str]:
    """orca pilot SNAPSHOT_PARAMS (-pol V -niter 0) with the band's -size/-scale."""
    return ([wsclean_bin, "-j", str(threads), "-pol", "V", "-niter", "0",
             "-mem", str(p.dirty_mem)] + _mem_args(p) + _geom_args(geom) + _weight_args(p)
            + ["-no-dirty", "-no-update-model-required", "-name", name, str(ms)])


def _single_pol_image(name: str, pol: str) -> Path:
    """WSClean names a single-polarisation image <name>-image.fits."""
    for cand in (Path(f"{name}-image.fits"), Path(f"{name}-{pol}-image.fits")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"wsclean produced no image for {name}")


def _run(cmd: List[str], retries: int = 0) -> None:
    for attempt in range(retries + 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        tail = (r.stdout + r.stderr)[-2000:]
        if attempt < retries:
            log.warning("wsclean failed (rc=%s), retry %d/%d: %s", r.returncode, attempt + 1,
                        retries, tail[-300:])
            time.sleep(5)
    raise RuntimeError(f"wsclean failed (rc={r.returncode}): {' '.join(cmd)}\n{tail}")


def image(ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
          threads: int, wsclean_bin: str = "wsclean",
          timings: Optional[Dict[str, float]] = None) -> List[Path]:
    """Image one MS; return [I image, V image].  ``timings`` receives seconds per step."""
    timings = {} if timings is None else timings
    out = [Path(f"{out_prefix}-{pol}-image.fits") for pol in ("I", "V")]
    if p.mode == "dirty":
        t0 = time.perf_counter()
        _run(dirty_iv_cmd(ms, out_prefix, geom, p, threads, wsclean_bin), p.retries)
        timings["image_dirty_iv"] = time.perf_counter() - t0
        return out

    t0 = time.perf_counter()
    name_i = f"{out_prefix}-Iclean"
    _run(clean_i_cmd(ms, name_i, geom, p, threads, wsclean_bin), p.retries)
    _single_pol_image(name_i, "I").rename(out[0])
    for suffix in ("psf", "model", "residual", "dirty"):
        Path(f"{name_i}-{suffix}.fits").unlink(missing_ok=True)
    timings["image_clean_i"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    name_v = f"{out_prefix}-Vdirty"
    _run(dirty_v_cmd(ms, name_v, geom, p, threads, wsclean_bin), p.retries)
    _single_pol_image(name_v, "V").rename(out[1])
    timings["image_dirty_v"] = time.perf_counter() - t0
    return out
