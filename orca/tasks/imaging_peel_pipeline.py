"""
Celery task that runs

    copy → applycal → flag_ants → AOFlagger → TTCal peel → WSClean

and then moves the whole NVMe workspace back to Lustre.

Nothing in the old codebase is touched; `orca/tasks/imaging_tasks.py`
will import this module so that Celery registers the task automatically.
"""
from __future__ import annotations
import os, shutil, uuid, glob, logging
from typing import List, Optional, Dict
from pathlib import Path

from casatasks import applycal
from orca.celery import app                       # same Celery instance
from orca.transform.flagging import (
    flag_ants,
    flag_with_aoflagger,
)
from orca.utils.paths import get_aoflagger_strategy
from orca.wrapper.ttcal import peel_with_ttcal
from orca.wrapper.wsclean import wsclean
from orca.tasks.imaging_tasks import (            # reuse helpers
    _nvme_workspace, _plot_dirty,
)
import subprocess
def peel_with_ttcal_maxiter5(ms: str, sources: str):
    """
    A local version of peel_with_ttcal but forcing --maxiter=5.
    Leaves orca/wrapper/ttcal.py untouched.
    """
    env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
               AIPSPATH='/opt/astro/casa-data dummy dummy')

    julia = '/opt/devel/pipeline/envs/julia060/bin/julia'
    ttcal = '/opt/devel/pipeline/envs/julia060/bin/ttcal.jl'

    cmd = [julia, ttcal, 'peel', ms, sources,
           '--beam', 'sine', '--maxiter', '5',
           '--tolerance', '1e-4', '--minuvw', '10']

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        LOG.error(f'TTCal failed with stderr: {proc.stderr}')
        LOG.info(f'stdout: {proc.stdout}')
        raise RuntimeError("TTCal failed.")
    LOG.info(f"TTCal peel stdout: {proc.stdout}")


LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
@app.task(
    bind=True,
    name="orca.tasks.imaging_peel_pipeline.peel_imaging_pipeline_task",
    queue="imaging",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def peel_imaging_pipeline_task(
    self,
    ms_path: str,
    delay_table: str,
    bandpass_table: str,
    final_dir: str,
    bad_corrs: Optional[List[int]] = None,
    aoflag_strategy: str | None     = "LWA_opt_GH1.lua",
    peel_sources_json: str | None   = "/home/pipeline/sources.json",
    extra_wsclean: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Full-featured imaging pipeline **added** on top of the old tasks.

    Returns  {'dirty_png': <file>, 'workspace': <directory>}
    """
    # 1. ---------------------------------------------------------------- copy
    workdir, nvme_ms = _nvme_workspace(ms_path)
    shutil.copytree(ms_path, nvme_ms)
    LOG.info("[%s] copied → %s", self.request.id, nvme_ms)

    # 2. ----------------------------------------------------------- applycal
    applycal(
        vis       = nvme_ms,
        gaintable = [delay_table, bandpass_table],
        calwt     = [False],
        flagbackup=True,
    )

    # 3. -------------------------------------------------------- flag_ants
    if bad_corrs:
        LOG.info("[%s] flag_ants %s", self.request.id, bad_corrs)
        flag_ants(nvme_ms, bad_corrs)

    # 4. ----------------------------------------------------- AOFlagger RFI
    if aoflag_strategy:
        lua_file = (
            get_aoflagger_strategy(aoflag_strategy)
            if "/" not in aoflag_strategy else aoflag_strategy
        )
        flag_with_aoflagger(nvme_ms, strategy=lua_file)
        LOG.info("[%s] AOFlagger %s done", self.request.id, lua_file)

    # 5. ---------------------------------------------------------- TTCal peel
    #if peel_sources_json:
    #    peel_with_ttcal(nvme_ms, peel_sources_json)
    #    LOG.info("[%s] TTCal peel done", self.request.id)
    if peel_sources_json:
        peel_with_ttcal_maxiter5(nvme_ms, peel_sources_json)
        LOG.info("[%s] TTCal peel done with maxiter=5", self.request.id)

    # 6. ------------------------------------------------------------ WSClean
    if extra_wsclean is None:
        extra_wsclean = [
            "-pol", "I", "-size", "4096", "4096",
            "-scale", "0.03125", "-niter", "5",
            "-weight", "briggs", "0", "-horizon-mask", "10deg",
            "-taper-inner-tukey", "30",
        ]
    #prefix     = os.path.join(workdir, os.path.splitext(os.path.basename(nvme_ms))[0])
    #dirty_fits = f"{prefix}-dirty.fits"
    #png_out    = f"{prefix}-dirty.png"
    prefix = os.path.join(workdir, os.path.splitext(os.path.basename(nvme_ms))[0])

    wsclean(
        ms_list=[nvme_ms], out_dir=workdir,
        filename_prefix=os.path.basename(prefix),
        extra_arg_list=extra_wsclean,
        num_threads=4, mem_gb=50,
    )
    #_plot_dirty(dirty_fits, png_out)
    dirty_fits_files = glob.glob(os.path.join(workdir, "*-dirty.fits"))
    png_files = []
    for dfits in dirty_fits_files:
        png_file = dfits.replace(".fits", ".png")
        _plot_dirty(dfits, png_file)
        png_files.append(png_file)
    # build your desired dir name (without UUID)
    final_subdir = os.path.join(final_dir, os.path.splitext(os.path.basename(nvme_ms))[0])

    # create it
    os.makedirs(final_subdir, exist_ok=True)

    # move all individual dirty fits & png into it
    for dfits in dirty_fits_files:
        shutil.move(dfits, os.path.join(final_subdir, os.path.basename(dfits)))
    for png in png_files:
        shutil.move(png, os.path.join(final_subdir, os.path.basename(png)))

    # now move *contents* of workdir (not the directory itself) into final_subdir
    for item in os.listdir(workdir):
        src = os.path.join(workdir, item)
        dst = os.path.join(final_subdir, item)
        shutil.move(src, dst)

    # remove the empty workdir
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "dirty_pngs": [os.path.join(final_subdir, os.path.basename(png)) for png in png_files],
        "workspace":  final_subdir,
    }

    '''os.makedirs(final_dir, exist_ok=True)
    for dfits in dirty_fits_files:
        shutil.move(dfits, os.path.join(final_dir, os.path.basename(dfits)))
    for png in png_files:
        shutil.move(png, os.path.join(final_dir, os.path.basename(png)))
    shutil.move(workdir, os.path.join(final_dir, os.path.basename(workdir)))

    return {
        "dirty_pngs": [os.path.join(final_dir, os.path.basename(png)) for png in png_files],
        "workspace":  os.path.join(final_dir, os.path.basename(workdir)),
    }
'''
