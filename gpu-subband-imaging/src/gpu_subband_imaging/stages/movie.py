"""Render masked Stokes I frames and stitch hourly H.264 movies."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

_STAMP_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")


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


def _utc_label(fits_path: str) -> Optional[str]:
    match = _STAMP_RE.search(Path(fits_path).name)
    if not match:
        return None
    stamp = datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    return f"UTC time: {stamp:%Y-%m-%d %H:%M:%S}"


def render_frame(fits_i: str, out_png: Path, vmax: float, max_px: int,
                 horizon_mask: bool = False,
                 horizon_radius_fraction: float = 0.49) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    img = _downsample(_load2d(fits_i), max_px)
    cmap = matplotlib.colormaps["inferno"]
    if horizon_mask:
        img = _horizon_mask(img, horizon_radius_fraction)
        cmap = cmap.copy()
        cmap.set_bad("black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    height, width = img.shape
    panel_width = max(64, round(width * 0.14))
    if (width + panel_width) % 2:
        panel_width += 1
    total_width = width + panel_width
    dpi = 100
    fig = Figure(figsize=(total_width / dpi, height / dpi), dpi=dpi,
                 facecolor="black", frameon=True)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, width / total_width, 1))
    image = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="nearest")
    ax.set_axis_off()

    label = _utc_label(fits_i)
    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                ha="left", va="top", color="white",
                fontsize=max(8, min(12, width / 45)),
                bbox={"facecolor": "black", "alpha": 0.6,
                      "edgecolor": "none", "pad": 2})

    cax = fig.add_axes(((width + panel_width * 0.20) / total_width,
                        0.16,
                        panel_width * 0.22 / total_width,
                        0.68))
    cax.set_facecolor("black")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.set_title("Jy/beam", color="white", fontsize=8, pad=6)
    colorbar.ax.tick_params(colors="white", labelsize=8, length=3)
    colorbar.outline.set_edgecolor("white")

    fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0)


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
