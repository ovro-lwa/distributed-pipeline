# orca/tasks/peel_experiment.py
"""Experimental peeling pipeline tasks.

Provides Celery tasks for testing and developing peeling workflows
including:
- TTCal peeling with configurable source lists
- RFI source peeling
- Multiple peeling iterations with different maxiter settings
- Diagnostic image generation

This is an experimental module for peeling algorithm development.
"""
from __future__ import absolute_import, unicode_literals

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import glob
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from casatasks import applycal
from orca.wrapper.wsclean import wsclean

from orca.celery import app   # Celery app

# --------- PATH CONFIG ---------

WORKDIR = Path("/lustre/nkosogor/peel_test")
BASE_73 = Path("/lustre/pipeline/night-time/averaged/73MHz")

SOURCES_JSON = WORKDIR / "sources.json"
RFI_JSON     = WORKDIR / "rfi_43.2_ver20251101.json"

JULIA060_ENV_NAME    = "julia060"
TTCAL_DEV_ENV_PREFIX = "/opt/devel/pipeline/envs/ttcal_dev"

CAL_TABLE = {
    "2024-12-18": WORKDIR / "calibration_2024-12-18_01h.B.flagged",
    "2025-05-06": WORKDIR / "calibration_2025-05-06_18h.B.flagged",
}

DEFAULT_MAXITER = 5  # used if caller doesn't specify


# ---- geometry / plotting constants ----

OVRO = EarthLocation(
    lon=-118.282 * u.deg,
    lat=37.243 * u.deg,
    height=1200 * u.m
)

CYG_A = SkyCoord("19h59m28s", "40d44m02s", frame="icrs")
CAS_A = SkyCoord("23h23m24s", "58d48m54s", frame="icrs")

R_MASK = 1833
VMIN, VMAX = -2, 12


# ================== LOW-LEVEL HELPERS ==================

def run_ttcal_with_conda(env_type: str, ms_name: str, json_path: Path, maxiter: int):
    """
    env_type: 'sources' or 'rfi'
    """
    json_path = json_path.resolve()
    ms_name = str(ms_name)

    base_cmd = [
        "conda", "run",
    ]

    if env_type == "sources":
        base_cmd += ["-n", JULIA060_ENV_NAME]
    elif env_type == "rfi":
        base_cmd += ["-p", TTCAL_DEV_ENV_PREFIX]
    else:
        raise ValueError(f"Unknown env_type {env_type}")

    cmd = base_cmd + [
        "ttcal.jl", "zest", ms_name, str(json_path),
        "--beam", "constant",
        "--minuvw", "10",
        "--maxiter", str(maxiter),
        "--tolerance", "1e-4",
    ]

    subprocess.run(cmd, check=True)


def run_wsclean_stage(ms_name: str, suffix: str):
    ms_dir = ms_name
    workdir = Path.cwd()

    extra_wsclean = [
        "-pol", "IV", "-size", "4096", "4096",
        "-scale", "0.03125",
        "-niter", "0",
        "-weight", "briggs", "0", "-horizon-mask", "10deg",
        "-taper-inner-tukey", "30",
    ]

    prefix_base = os.path.splitext(os.path.basename(ms_dir))[0]
    filename_prefix = f"{prefix_base}_0_iterations_{suffix}"

    wsclean(
        ms_list=[ms_dir],
        out_dir=str(workdir),
        filename_prefix=filename_prefix,
        extra_arg_list=extra_wsclean,
        num_threads=4,
        mem_gb=50,
    )


def run_applycal_pre_peel(ms_name: str, caltable_path: Path):
    applycal(
        vis=ms_name,
        gaintable=[str(caltable_path)],
        spw="",
        spwmap=[[13]],
        interp="linear,nearestflag",
        calwt=False,
        parang=False,
    )


# --------- time / sky helpers ----------

def parse_utc_from_tag(tag: str) -> Time:
    """
    tag example: '2025-05-06_05_20250506_050008_73MHz_averaged_maxiter05'
    or without the _maxiterXX suffix.
    """
    core = tag
    if "_maxiter" in core:
        core = core.split("_maxiter")[0]

    parts = core.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected tag: {tag}")
    date_code = parts[2]   # '20250506'
    time_code = parts[3]   # '050008'
    dt = datetime.strptime(date_code + time_code, "%Y%m%d%H%M%S")
    return Time(dt, scale="utc", location=OVRO)


def compute_lst_and_alts(t: Time):
    lst = t.sidereal_time("apparent")
    altaz_frame = AltAz(obstime=t, location=OVRO)
    cyg_alt = CYG_A.transform_to(altaz_frame).alt.deg
    cas_alt = CAS_A.transform_to(altaz_frame).alt.deg
    return lst, cyg_alt, cas_alt


def lst_to_hms_str(lst_angle):
    hours = lst_angle.hour
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    h %= 24
    return f"{h:02d}:{m:02d}"


# -------- image / RMS helpers ---------

def _mask_image(data, wcs, r_mask=R_MASK):
    ny, nx = data.shape
    yy, xx = np.indices((ny, nx))
    world = wcs.pixel_to_world(xx, yy)
    invalid_wcs = (~np.isfinite(world.ra.deg) |
                   ~np.isfinite(world.dec.deg))
    cy, cx = ny // 2, nx // 2
    outside_circle = (xx - cx)**2 + (yy - cy)**2 > r_mask**2
    mask = invalid_wcs | outside_circle
    return mask, (cx, cy)


def load_masked_image(fits_path, r_mask=R_MASK):
    with fits.open(fits_path) as hdul:
        data   = hdul[0].data[0, 0, :, :].astype(float)
        header = hdul[0].header
    wcs = WCS(header).celestial
    mask, center = _mask_image(data, wcs, r_mask=r_mask)
    data_masked = data.copy()
    data_masked[mask] = np.nan
    return data_masked, wcs, center


def robust_rms(vals, nsig=5.0, max_iter=5):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan

    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad == 0:
        return np.std(vals)
    sigma = 1.4826 * mad

    for _ in range(max_iter):
        diff = vals - median
        keep = np.abs(diff) <= nsig * sigma
        new_vals = vals[keep]
        if new_vals.size == 0:
            break
        new_median = np.median(new_vals)
        new_mad = np.median(np.abs(new_vals - new_median))
        if new_mad == 0:
            vals = new_vals
            median = new_median
            break
        new_sigma = 1.4826 * new_mad
        if np.isclose(new_sigma, sigma, rtol=1e-3):
            vals = new_vals
            median = new_median
            sigma = new_sigma
            break
        vals = new_vals
        median = new_median
        sigma = new_sigma

    return np.std(vals)


def compute_rms_from_fits(fits_path):
    data_masked, _, _ = load_masked_image(fits_path)
    good = np.isfinite(data_masked)
    return robust_rms(data_masked[good])


def make_three_panel_png(run_dir: Path, tag: str,
                         rms_pre: float,
                         rms_after1: float,
                         rms_after2: float):
    pre_fits    = glob.glob(str(run_dir / "*_0_iterations_pre_peel-I-image.fits"))
    after1_fits = glob.glob(str(run_dir / "*_0_iterations_after_peel-I-image.fits"))
    after2_fits = glob.glob(str(run_dir / "*_0_iterations_after_2nd_peel-I-image.fits"))

    if not (pre_fits and after1_fits and after2_fits):
        return

    pre_fits    = pre_fits[0]
    after1_fits = after1_fits[0]
    after2_fits = after2_fits[0]

    data_pre,    wcs_pre,    center = load_masked_image(pre_fits)
    data_after1, wcs_after1, _      = load_masked_image(after1_fits)
    data_after2, wcs_after2, _      = load_masked_image(after2_fits)
    cx, cy = center

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(15, 5))
    wcss   = [wcs_pre, wcs_after1, wcs_after2]
    datas  = [data_pre, data_after1, data_after2]
    titles = [
        f"Pre-peel (RMS={rms_pre:.2f})",
        f"After 1st peel (RMS={rms_after1:.2f})",
        f"After 2nd peel (RMS={rms_after2:.2f})",
    ]

    im = None
    for i, (img, wcs, panel_title) in enumerate(zip(datas, wcss, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection=wcs)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.coords[0].set_ticks_visible(False)
        ax.coords[1].set_ticks_visible(False)
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

        overlay = ax.get_coords_overlay("fk5")
        overlay.grid(color="white", ls="dotted", lw=0.8, alpha=0.8)
        overlay[0].set_ticks(spacing=20*u.deg)
        overlay[1].set_ticks(spacing=20*u.deg)

        im = ax.imshow(img, origin="lower", cmap=cmap,
                       vmin=VMIN, vmax=VMAX)

        border = Circle((cx, cy), R_MASK,
                        transform=ax.get_transform("pixel"),
                        fill=False, edgecolor="black",
                        linewidth=1.0, alpha=1.0)
        ax.add_patch(border)
        ax.set_title(panel_title, fontsize=12)

    plt.subplots_adjust(wspace=0.02, right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(r"Intensity (Jy/beam)", fontsize=12)

    t = parse_utc_from_tag(tag)
    lst, cyg_alt, cas_alt = compute_lst_and_alts(t)
    lst_str = lst_to_hms_str(lst)
    utc_str = t.utc.datetime.strftime("%Y-%m-%d %H:%M:%S")

    fig.suptitle(
        f"{tag}   {utc_str} UTC  (LST {lst_str})  "
        f"Cyg A {cyg_alt:.1f}°, Cas A {cas_alt:.1f}°",
        fontsize=13
    )

    out_png = run_dir / f"{tag}_3panel_peeling.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_diff_image(run_dir: Path, tag: str):
    pre_fits    = glob.glob(str(run_dir / "*_0_iterations_pre_peel-I-image.fits"))
    after2_fits = glob.glob(str(run_dir / "*_0_iterations_after_2nd_peel-I-image.fits"))
    if not (pre_fits and after2_fits):
        return None, None

    pre_fits    = pre_fits[0]
    after2_fits = after2_fits[0]

    with fits.open(pre_fits) as h_pre, fits.open(after2_fits) as h_after:
        data_pre    = h_pre[0].data.astype(np.float32)
        data_after2 = h_after[0].data.astype(np.float32)
        diff_data   = data_pre - data_after2
        h_diff = fits.HDUList([fits.PrimaryHDU(data=diff_data, header=h_pre[0].header)])

    diff_fits = run_dir / f"{tag}_0_iterations_diff_pre_minus_after2-I-image.fits"
    h_diff.writeto(diff_fits, overwrite=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(diff_data[0, 0, :, :], origin="lower", cmap="coolwarm")
    plt.colorbar(label="Pre - After2 (Jy/beam)")
    plt.title(f"Diff image: {tag}")
    diff_png = run_dir / f"{tag}_diff_pre_minus_after2.png"
    plt.savefig(diff_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    return diff_fits, diff_png


# ================== THE CELERY TASK ==================

@app.task(name="orca.tasks.peel_experiment.peel_experiment_task")
def peel_experiment_task(ms_rel_path: str, maxiter: int = DEFAULT_MAXITER):
    """
    Big experiment task:
      - copy MS to WORKDIR/tag_maxiterXX
      - applycal
      - image pre-peel -> RMS_pre
      - ttcal (sources, maxiter=...) + image -> RMS_after1
      - ttcal (RFI, maxiter=...)     + image -> RMS_after2
      - make 3-panel PNG
      - make pre-minus-after2 difference image

    ms_rel_path is relative to BASE_73, e.g.
      '2025-05-06/05/20250506_050008_73MHz_averaged.ms'
    """

    eff_maxiter = int(maxiter) if maxiter is not None else DEFAULT_MAXITER

    base_ms_path = BASE_73 / ms_rel_path
    if not base_ms_path.is_dir():
        raise FileNotFoundError(f"Base MS not found: {base_ms_path}")

    parts = Path(ms_rel_path).parts  # [date, hour, msname]
    date_str = parts[0]
    hour_str = parts[1]
    msname   = parts[2]

    caltab = CAL_TABLE.get(date_str)
    if caltab is None or not caltab.exists():
        raise FileNotFoundError(f"No calibration table for {date_str}: {caltab}")

    tag_core = f"{date_str}_{hour_str}_{msname[:-3]}"
    tag      = f"{tag_core}_maxiter{eff_maxiter:02d}"
    run_dir  = WORKDIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    ms_basename = msname
    dest_ms = run_dir / ms_basename
    if not dest_ms.exists():
        shutil.copytree(base_ms_path, dest_ms)

    t0 = time.time()
    os.chdir(run_dir)

    # 1) applycal + pre-peel image
    run_applycal_pre_peel(ms_basename, caltab)
    run_wsclean_stage(ms_basename, "pre_peel")
    pre_fits = glob.glob("*_0_iterations_pre_peel-I-image.fits")[0]
    rms_pre = compute_rms_from_fits(pre_fits)

    # 2) first peel (sources) + image
    run_ttcal_with_conda("sources", ms_basename, SOURCES_JSON, eff_maxiter)
    run_wsclean_stage(ms_basename, "after_peel")
    after1_fits = glob.glob("*_0_iterations_after_peel-I-image.fits")[0]
    rms_after1 = compute_rms_from_fits(after1_fits)

    # 3) second peel (RFI) + image
    run_ttcal_with_conda("rfi", ms_basename, RFI_JSON, eff_maxiter)
    run_wsclean_stage(ms_basename, "after_2nd_peel")
    after2_fits = glob.glob("*_0_iterations_after_2nd_peel-I-image.fits")[0]
    rms_after2 = compute_rms_from_fits(after2_fits)

    # 4) 3-panel PNG
    make_three_panel_png(run_dir, tag, rms_pre, rms_after1, rms_after2)

    # 5) difference image
    diff_fits, diff_png = make_diff_image(run_dir, tag)

    elapsed = time.time() - t0
    os.chdir(WORKDIR)

    return {
        "tag": tag,
        "ms_rel_path": ms_rel_path,
        "run_dir": str(run_dir),
        "maxiter": eff_maxiter,
        "rms_pre": float(rms_pre),
        "rms_after1": float(rms_after1),
        "rms_after2": float(rms_after2),
        "diff_fits": str(diff_fits) if diff_fits else None,
        "diff_png": str(diff_png) if diff_png else None,
        "elapsed_sec": elapsed,
    }

