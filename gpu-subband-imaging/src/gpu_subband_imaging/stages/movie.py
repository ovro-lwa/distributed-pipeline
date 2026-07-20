"""Render masked Stokes I frames and stitch hourly H.264 movies."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np


def _load2d(fits_path: str) -> np.ndarray:
    from astropy.io import fits
    return np.squeeze(fits.getdata(fits_path)).astype(np.float64)


def _downsample(a: np.ndarray, max_px: int) -> np.ndarray:
    f = max(1, a.shape[0] // max_px)
    n = a.shape[0] - a.shape[0] % f
    return a[:n, :n].reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def _horizon_mask(a: np.ndarray, radius_fraction: float) -> np.ndarray:
    """Mask pixels outside the centered horizon disk."""
    ny, nx = a.shape
    y, x = np.ogrid[:ny, :nx]
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    r = float(radius_fraction) * min(nx, ny)
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return np.where(inside, a, np.nan)


def robust_rms(a: np.ndarray) -> float:
    n = a.shape[0]
    c = a[n // 4:3 * n // 4, n // 4:3 * n // 4]
    return float(1.4826 * np.nanmedian(np.abs(c - np.nanmedian(c))))


def _finite_rms(a: np.ndarray) -> float:
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 0.0
    return float(1.4826 * np.nanmedian(np.abs(vals - np.nanmedian(vals))))


def render_frame(fits_i: str, out_png: Path, vmax: float, max_px: int,
                 horizon_mask: bool = False,
                 horizon_radius_fraction: float = 0.49) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = _downsample(_load2d(fits_i), max_px)
    cmap = plt.get_cmap("inferno")
    if horizon_mask:
        img = _horizon_mask(img, horizon_radius_fraction)
        cmap = cmap.copy()
        cmap.set_bad("black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_png), img, origin="lower", cmap=cmap, vmin=0, vmax=vmax)


def band_vmax(sample_fits: List[str], mode: str, fixed_vmax: float,
              max_px: int, k: float = 10.0,
              horizon_mask: bool = False,
              horizon_radius_fraction: float = 0.49) -> float:
    if mode == "fixed":
        return fixed_vmax
    samples = []
    for f in sample_fits:
        img = _downsample(_load2d(f), max_px)
        if horizon_mask:
            img = _horizon_mask(img, horizon_radius_fraction)
            samples.append(_finite_rms(img))
        else:
            samples.append(robust_rms(img))
    rms = np.nanmedian(samples)
    return max(1.0, k * float(rms))


def _ffmpeg_bin() -> str:
    """Find FFmpeg from the environment, PATH, or imageio-ffmpeg."""
    import os
    import shutil as sh
    env = os.environ.get("FFMPEG_BIN")
    if env:
        return env
    if sh.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def stitch(frame_dir: Path, out_mp4: Path, fps: int, crf: int,
           ffmpeg_bin: Optional[str] = None) -> Optional[Path]:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    listfile = frame_dir / "_frames.txt"
    listfile.write_text("".join(f"file '{f.name}'\n" for f in frames))
    tmp_mp4 = out_mp4.with_name(f".{out_mp4.name}.tmp.mp4")
    if tmp_mp4.exists():
        tmp_mp4.unlink()
    subprocess.run([ffmpeg_bin or _ffmpeg_bin(), "-y", "-r", str(fps),
                    "-f", "concat", "-safe", "0", "-i", str(listfile),
                    "-c:v", "libx264",
                    "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                    "-crf", str(crf), str(tmp_mp4)],
                   check=True, capture_output=True, cwd=str(frame_dir))
    tmp_mp4.replace(out_mp4)
    return out_mp4
