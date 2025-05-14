# orca/tasks/imaging_tasks.py
import os, shutil, uuid, glob
from typing import List, Tuple, Optional
from casatasks import applycal
from orca.celery import app
from orca.wrapper.wsclean import wsclean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import logging
from orca.transform.flagging import flag_ants


logger = logging.getLogger(__name__)

NVME_ROOT = "/fast/pipeline"

def _nvme_workspace(ms_path: str) -> (str, str):
    base = os.path.basename(ms_path.rstrip("/"))
    work = os.path.join(NVME_ROOT, f"{base}-{uuid.uuid4().hex[:8]}")
    os.makedirs(work, exist_ok=True)
    return work, os.path.join(work, base)

def _plot_dirty(dirty_fits: str, png_out: str):
    with fits.open(dirty_fits) as hdul:
        data = hdul[0].data[0, 0, :, :]
        wcs  = WCS(hdul[0].header, naxis=[1, 2])

    plt.figure(figsize=(20, 15))
    ax = plt.subplot(projection=wcs)
    im = ax.imshow(
        data,
        cmap='jet',
        vmin=-15, vmax=95,          
        origin="lower"
    )
    ax.set_xlabel('RA'); ax.set_ylabel('Dec')
    ax.get_coords_overlay('fk5').grid(color='white', ls='dotted')
    cbar = plt.colorbar(im); cbar.set_label('Intensity')
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    plt.close()

def _shared_nvme_workspace(batch_root: str, ms_path: str) -> Tuple[str, str]:
    """
    Re-use a single directory (batch_root) as the workspace.
    Returns (workdir, ms_copy_path).
    """
    workdir = batch_root          # caller creates it once
    os.makedirs(workdir, exist_ok=True)
    ms_copy = os.path.join(workdir, os.path.basename(ms_path.rstrip("/")))
    return workdir, ms_copy

@app.task
def imaging_pipeline_task(
    ms_path: str,
    delay_table: str,
    bandpass_table: str,
    final_dir: str,
    extra_wsclean: List[str]=None
) -> str:
    """
    Runs copy→applycal→WSClean→PNG→remove extra files→export→purge NVMe
    and returns the path to the saved PNG.
    """
    # ---------- 1. copy to NVMe ----------
    workdir, nvme_ms = _nvme_workspace(ms_path)
    shutil.copytree(ms_path, nvme_ms)

    # ---------- 2. apply calibration ----------
    applycal(
        vis       = nvme_ms,
        gaintable = [delay_table, bandpass_table],
        calwt     = [False],
        flagbackup=True
    )

    # ---------- 3. imaging with WSClean ----------
    if extra_wsclean is None:
        extra_wsclean = [
            '-pol', 'I',
            '-size', '4096', '4096',
            '-scale', '0.03125',
            '-niter', '0',
            '-weight', 'briggs', '0',
            '-horizon-mask', '10deg',
            '-taper-inner-tukey', '30'
        ]
    prefix = os.path.join(workdir, os.path.splitext(os.path.basename(nvme_ms))[0])
    wsclean(
        ms_list=[nvme_ms], out_dir=workdir,
        filename_prefix=os.path.basename(prefix),
        extra_arg_list=extra_wsclean,
        num_threads=4, mem_gb=50
    )
    dirty_fits = f"{prefix}-dirty.fits"

    # ---------- 4. make PNG (same colourscale) ----------
    png_out = f"{prefix}-dirty.png"
    _plot_dirty(dirty_fits, png_out)

    # ---------- 5. trim workspace ----------
    for fp in glob.glob(os.path.join(workdir, "*")):
        if fp not in (dirty_fits, png_out):
            shutil.rmtree(fp, ignore_errors=True) if os.path.isdir(fp) else os.remove(fp)

    # ---------- 6. export & clean ----------
    os.makedirs(final_dir, exist_ok=True)
    dst_fits = shutil.move(dirty_fits, os.path.join(final_dir, os.path.basename(dirty_fits)))
    dst_png  = shutil.move(png_out,   os.path.join(final_dir, os.path.basename(png_out)))
    shutil.rmtree(workdir, ignore_errors=True)

    return dst_png     


@app.task(bind=True,
          name='orca.tasks.imaging_tasks.imaging_shared_pipeline_task',
          autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def imaging_shared_pipeline_task(
    self,
    ms_path: str,
    delay_table: str,
    bandpass_table: str,
    final_dir: str,
    workdir_root: str,
    keep_full_products: bool = False,
    extra_wsclean: Optional[List[str]] = None,
    bad_corrs: Optional[List[int]] = None,
) -> str:
    """
    Shared-workspace version:
      • All MSs in a batch share *workdir_root*.
      • Only the first MS (or any with keep_full_products=True) keeps the full
        WSClean output; others retain just -dirty FITS + PNG.
      • Automatic retries wipe the MS sub-dir first to avoid half-baked data.
    """
    # ---------- 1. workspace  --------------------------------------------
    workdir, nvme_ms = _shared_nvme_workspace(workdir_root, ms_path)

    if os.path.exists(nvme_ms):       # ★ wipe half-done copy on retry
        shutil.rmtree(nvme_ms, ignore_errors=True)
    shutil.copytree(ms_path, nvme_ms)

    # ---------- 2. apply calibration -------------------------------------
    applycal(
        vis=nvme_ms,
        gaintable=[delay_table, bandpass_table],
        calwt=[False],
        flagbackup=True,
    )
    # ---------- 2a. flag bad antennas if provided ------------------------
    if bad_corrs:
        logger.info(f"[{self.request.id}] Flagging bad corr numbers: {bad_corrs}")
        flag_ants(nvme_ms, bad_corrs)

    # ---------- 3. imaging -----------------------------------------------
    if extra_wsclean is None:
        extra_wsclean = [
            '-pol', 'I', '-size', '4096', '4096',
            '-scale', '0.03125', 
            '-niter', '1000' if keep_full_products else '0',
            #'-niter', '0',
            '-weight', 'briggs', '0', '-horizon-mask', '10deg',
            '-taper-inner-tukey', '30',
        ]
    logger.info(f"[{self.request.id}] Running WSClean with args: {extra_wsclean}")

    prefix     = os.path.join(workdir,
                              os.path.splitext(os.path.basename(nvme_ms))[0])
    dirty_fits = f"{prefix}-dirty.fits"
    png_out    = f"{prefix}-dirty.png"

    wsclean(
        ms_list=[nvme_ms], out_dir=workdir,
        filename_prefix=os.path.basename(prefix),
        extra_arg_list=extra_wsclean,
        num_threads=4, mem_gb=50,
    )

    # ---------- 4. PNG ----------------------------------------------------
    _plot_dirty(dirty_fits, png_out)

    # ---------- 5. trim ---------------------------------------------------
    if not keep_full_products:
        for fp in glob.glob(f"{prefix}*"):
            if fp not in (dirty_fits, png_out):
                os.remove(fp) if os.path.isfile(fp) else shutil.rmtree(fp)

    shutil.rmtree(nvme_ms, ignore_errors=True)   # drop calibrated copy

    # ---------- 6. export -------------------------------------------------
    os.makedirs(final_dir, exist_ok=True)
    shutil.move(dirty_fits, os.path.join(final_dir, os.path.basename(dirty_fits)))
    shutil.move(png_out,   os.path.join(final_dir, os.path.basename(png_out)))
    
    if keep_full_products:
        extra_products = [
            f"{prefix}-image.fits",
            f"{prefix}-model.fits",
            f"{prefix}-psf.fits",
            f"{prefix}-residual.fits",
            f"{prefix}-horizon-mask.fits",
        ]
        for prod in extra_products:
            if os.path.exists(prod):
                shutil.move(prod, os.path.join(final_dir, os.path.basename(prod)))

        # also move the .flagversions directory if it exists
        flagversions_dir = os.path.join(workdir, os.path.basename(nvme_ms) + '.flagversions')
        if os.path.exists(flagversions_dir):
            shutil.move(flagversions_dir, os.path.join(final_dir, os.path.basename(flagversions_dir)))



    return png_out
