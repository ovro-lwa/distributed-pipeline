"""Celery tasks for subband processing on NVMe-local calim servers.

Architecture
------------
The processing is split into two phases so that Celery handles the
per-MS parallelism in Phase 1, while Phase 2 runs sequentially on the
same node after all individual MS files are ready.

Phase 1 — per-MS  (embarrassingly parallel, one Celery task per MS):
    copy to NVMe → flag bad antennas → apply calibration → peel (sky + RFI)

Phase 2 — per-subband  (sequential, runs after all Phase 1 tasks finish):
    concatenate → fix field ID → chgcentre → AOFlagger → pilot snapshots →
    snapshot QA → hot baseline removal → science imaging → archive to Lustre

Both phases are routed to the **same node-specific queue** (e.g. ``calim08``)
so that all I/O stays on the node's local NVMe.

Usage (from a submission script or notebook)::

    from celery import chord
    from orca.tasks.subband_tasks import prepare_one_ms_task, process_subband_task
    from orca.resources.subband_config import get_queue_for_subband

    queue = get_queue_for_subband('73MHz')

    # Phase 1: parallel per-MS
    header_tasks = [
        prepare_one_ms_task.s(
            src_ms=ms, nvme_work_dir=work_dir,
            bp_table=bp, xy_table=xy,
            peel_sky=True, peel_rfi=True,
        ) for ms in ms_files
    ]

    # Phase 2: runs after all Phase 1 tasks complete; receives list of MS paths
    pipeline = chord(header_tasks)(
        process_subband_task.s(
            work_dir=work_dir, subband='73MHz',
            lst_label='14h', obs_date='2025-06-15',
            run_label='Run_20250615',
        )
    )
"""
import os
import glob
import shutil
import socket
import logging
import traceback

# WSClean (linked against OpenBLAS) refuses to start if OpenBLAS multi-
# threading is enabled.  Setting this early ensures every subprocess inherits it.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from typing import List, Optional

import numpy as np
from celery import chord
from casacore.tables import table

from orca.celery import app
from orca.wrapper.ttcal import zest_with_ttcal
from orca.wrapper.change_phase_centre import change_phase_center

from orca.transform.subband_processing import (
    redirect_casa_log,
    copy_ms_to_nvme,
    apply_calibration,
    flag_bad_antennas,
    concatenate_ms,
    fix_field_id,
    analyze_snapshot_quality,
    flag_bad_integrations,
    add_timestamps_to_images,
    plot_snapshot_diagnostics,
    archive_results,
    run_subprocess,
    apply_pb_correction_to_images,
)
from orca.resources.subband_config import (
    PEELING_PARAMS,
    AOFLAGGER_STRATEGY,
    SNAPSHOT_PARAMS,
    IMAGING_STEPS,
    NVME_BASE_DIR,
    LUSTRE_ARCHIVE_DIR,
    get_queue_for_subband,
    get_image_resources,
)

logger = logging.getLogger(__name__)


# ============================================================================
#  PHASE 1 — Per-MS task  (runs in parallel via Celery)
# ============================================================================

@app.task(
    bind=True,
    name='orca.tasks.subband_tasks.prepare_one_ms_task',
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={'max_retries': 2},
    acks_late=True,
)
def prepare_one_ms_task(
    self,
    src_ms: str,
    nvme_work_dir: str,
    bp_table: str,
    xy_table: str,
    peel_sky: bool = False,
    peel_rfi: bool = False,
) -> str:
    """Copy one MS to NVMe, flag, calibrate, and optionally peel.

    This is the Phase 1 workhorse.  Each MS file gets its own Celery task
    so they run in parallel across the worker's concurrency slots.

    Args:
        src_ms: Source MS path on Lustre.
        nvme_work_dir: NVMe working directory (same for all MS in a subband).
        bp_table: Bandpass calibration table path.
        xy_table: XY-phase calibration table path.
        peel_sky: Run TTCal zest with sky model.
        peel_rfi: Run TTCal zest with RFI model.

    Returns:
        Path to the processed MS on NVMe.
    """
    node = socket.gethostname()
    logger.info(
        f"[{self.request.id}] Phase 1 START on {node}: {os.path.basename(src_ms)}"
    )

    os.makedirs(nvme_work_dir, exist_ok=True)
    redirect_casa_log(nvme_work_dir)

    # 1. Copy to NVMe
    nvme_ms = copy_ms_to_nvme(src_ms, nvme_work_dir)

    # 2. Flag bad antennas (via mnc_python)
    flag_bad_antennas(nvme_ms)

    # 3. Apply calibration (bandpass + XY-phase)
    cal_ok = apply_calibration(nvme_ms, bp_table, xy_table)
    if not cal_ok:
        logger.error(f"Calibration failed for {os.path.basename(nvme_ms)}; removing")
        shutil.rmtree(nvme_ms, ignore_errors=True)
        raise RuntimeError(f"Calibration failed for {src_ms}")

    # 4. Peeling
    if peel_sky:
        logger.info(f"Peeling sky model on {os.path.basename(nvme_ms)}")
        zest_with_ttcal(
            ms=nvme_ms,
            sources=PEELING_PARAMS['sky_model'],
            beam=PEELING_PARAMS['beam'],
            minuvw=PEELING_PARAMS['minuvw'],
            maxiter=PEELING_PARAMS['maxiter'],
            tolerance=PEELING_PARAMS['tolerance'],
        )

    if peel_rfi:
        logger.info(f"Peeling RFI model on {os.path.basename(nvme_ms)}")
        # RFI peeling may use a different conda env (ttcal_dev) than sky (julia060).
        # The orca wrapper zest_with_ttcal uses julia060 hardcoded.
        # If the envs are different, use shell-based invocation like the
        # Slurm pipeline does.
        rfi_env = PEELING_PARAMS.get('rfi_env', 'julia060')
        sky_env = PEELING_PARAMS.get('sky_env', 'julia060')
        if rfi_env != sky_env:
            # Shell-based invocation matching process_subband.py
            peel_env = os.environ.copy()
            peel_env["OMP_NUM_THREADS"] = "8"
            cmd = (
                f"source ~/.bashrc && conda activate {rfi_env} && "
                f"ttcal.jl zest {nvme_ms} {PEELING_PARAMS['rfi_model']} "
                f"{PEELING_PARAMS['args']}"
            )
            import subprocess
            subprocess.run(
                cmd, shell=True, check=True,
                executable='/bin/bash', env=peel_env,
            )
        else:
            zest_with_ttcal(
                ms=nvme_ms,
                sources=PEELING_PARAMS['rfi_model'],
                beam=PEELING_PARAMS['beam'],
                minuvw=PEELING_PARAMS['minuvw'],
                maxiter=PEELING_PARAMS['maxiter'],
                tolerance=PEELING_PARAMS['tolerance'],
            )

    logger.info(
        f"[{self.request.id}] Phase 1 DONE: {os.path.basename(nvme_ms)}"
    )
    return nvme_ms


# ============================================================================
#  PHASE 2 — Per-subband task  (runs after all Phase 1 tasks complete)
# ============================================================================

@app.task(
    bind=True,
    name='orca.tasks.subband_tasks.process_subband_task',
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={'max_retries': 1},
    acks_late=True,
    time_limit=14400,       # 4 hours hard limit
    soft_time_limit=13800,  # 3h50m soft limit
)
def process_subband_task(
    self,
    ms_paths: List[str],
    work_dir: str,
    subband: str,
    lst_label: str,
    obs_date: str,
    run_label: str,
    hot_baselines: bool = False,
    skip_cleanup: bool = False,
    cleanup_nvme: bool = False,
) -> str:
    """Phase 2: concatenate, image, and archive one subband.

    This task is used as the callback in a ``chord``: it receives the list
    of NVMe MS paths returned by the Phase 1 tasks.

    Args:
        ms_paths: NVMe paths returned by prepare_one_ms_task (via chord).
        work_dir: NVMe working directory (same as Phase 1).
        subband: Frequency label, e.g. '73MHz'.
        lst_label: LST label, e.g. '14h'.
        obs_date: Observation date string 'YYYY-MM-DD'.
        run_label: Pipeline run label.
        hot_baselines: Whether to run hot-baseline diagnostics.
        skip_cleanup: If True, keep the concat MS on NVMe.
        cleanup_nvme: If True, remove the entire NVMe work_dir after archiving.

    Returns:
        Path to the Lustre archive directory with final products.
    """
    node = socket.gethostname()
    logger.info(
        f"[{self.request.id}] Phase 2 START on {node}: "
        f"{subband} ({len(ms_paths)} files)"
    )

    os.chdir(work_dir)
    redirect_casa_log(work_dir)

    # Filter out any Nones (from failed Phase 1 tasks that were retried and
    # still returned nothing — shouldn't happen with autoretry, but be safe).
    valid_ms = [p for p in ms_paths if p and os.path.isdir(p)]

    # Check if a concat MS already exists from a previous (retried) attempt.
    # On retry, individual MS files may have been cleaned up, but the concat
    # MS survives — we can resume from it.
    existing_concat = os.path.join(work_dir, f"{subband}_concat.ms")
    have_concat = os.path.isdir(existing_concat)

    if not valid_ms and not have_concat:
        raise RuntimeError(f"No valid MS files for {subband}")

    # ------------------------------------------------------------------
    #  Create output directory structure
    # ------------------------------------------------------------------
    for d in ['I/deep', 'V/deep', 'I/10min', 'V/10min', 'snapshots', 'QA']:
        os.makedirs(os.path.join(work_dir, d), exist_ok=True)

    # ------------------------------------------------------------------
    #  1. Concatenation  (skip if concat MS already exists from prior attempt)
    # ------------------------------------------------------------------
    if have_concat:
        concat_ms = existing_concat
        logger.info(f"Resuming from existing concat MS: {concat_ms}")
    else:
        logger.info("Concatenating MS files...")
        concat_ms = concatenate_ms(valid_ms, work_dir, subband)
        if not concat_ms:
            raise RuntimeError("Concatenation failed")

        # Clean up individual MS files (they are on NVMe, space is precious)
        if not skip_cleanup:
            for ms in valid_ms:
                if os.path.exists(ms):
                    shutil.rmtree(ms)

    # ------------------------------------------------------------------
    #  2. Fix FIELD_ID
    # ------------------------------------------------------------------
    fix_field_id(concat_ms)

    # ------------------------------------------------------------------
    #  3. Change phase centre
    # ------------------------------------------------------------------
    hour_int = int(lst_label.replace('h', ''))
    phase_center = f"{hour_int:02d}h30m00s 37d12m57.057s"
    logger.info(f"Changing phase centre → {phase_center}")
    change_phase_center(concat_ms, phase_center)

    # ------------------------------------------------------------------
    #  4. AOFlagger
    # ------------------------------------------------------------------
    aoflagger_bin = os.environ.get('AOFLAGGER_BIN', '/opt/bin/aoflagger')
    logger.info(f"Running AOFlagger with strategy {AOFLAGGER_STRATEGY}")
    run_subprocess(
        [aoflagger_bin, '-strategy', AOFLAGGER_STRATEGY, concat_ms],
        "AOFlagger (Post-Concat)",
    )

    # ------------------------------------------------------------------
    #  5. Pilot snapshots + QA
    # ------------------------------------------------------------------
    try:
        t = table(concat_ms, ack=False)
        times = t.getcol("TIME")
        n_ints = len(np.unique(times))
        t.close()
    except Exception:
        n_ints = 357

    pilot_name = f"{subband}-{SNAPSHOT_PARAMS['suffix']}"
    pilot_path = os.path.join(work_dir, "snapshots", pilot_name)

    wsclean_bin = os.environ.get('WSCLEAN_BIN', '/opt/bin/wsclean')
    _, _, wsclean_j = get_image_resources(subband)
    cmd_pilot = (
        [wsclean_bin]
        + ['-j', str(wsclean_j)]
        + SNAPSHOT_PARAMS['args']
        + ['-name', pilot_path, '-intervals-out', str(n_ints), concat_ms]
    )
    run_subprocess(cmd_pilot, "Pilot snapshot imaging")

    add_timestamps_to_images(
        os.path.join(work_dir, "snapshots"), pilot_name, concat_ms, n_ints,
    )

    pilot_v = sorted(glob.glob(
        os.path.join(work_dir, "snapshots", f"{pilot_name}*-V-image.fits")
    ))
    bad_idx, stats = analyze_snapshot_quality(pilot_v)
    plot_snapshot_diagnostics(stats, bad_idx, work_dir, subband)

    if bad_idx:
        flag_bad_integrations(concat_ms, bad_idx, n_ints)

    # ------------------------------------------------------------------
    #  6. Hot baseline removal (optional)
    # ------------------------------------------------------------------
    if hot_baselines:
        try:
            _run_hot_baseline_diagnostics(concat_ms, work_dir)
        except Exception as e:
            logger.error(f"Hot baseline diagnostics failed: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    #  7. Science imaging + PB correction
    # ------------------------------------------------------------------
    logger.info(f"Starting Science Imaging for {subband}...")
    logger.info(f"wsclean thread limit: -j {wsclean_j}")

    for step in IMAGING_STEPS:
        target_dir = os.path.join(work_dir, step['pol'], step['category'])
        base = f"{subband}-{step['suffix']}"
        full_path = os.path.join(target_dir, base)

        cmd = [wsclean_bin] + ['-j', str(wsclean_j)] + step['args'] + ['-name', full_path]

        if step.get('per_integration'):
            n_out = n_ints
            cmd += ['-intervals-out', str(n_ints)]
        elif '-intervals-out' in step['args']:
            idx = step['args'].index('-intervals-out')
            n_out = int(step['args'][idx + 1])
        else:
            n_out = 1

        cmd.append(concat_ms)
        run_subprocess(cmd, f"Imaging {step['suffix']}")
        add_timestamps_to_images(target_dir, base, concat_ms, n_out)

        # Apply Primary Beam Correction to all images from this step
        pb_count = apply_pb_correction_to_images(target_dir, base)
        if pb_count > 0:
            logger.info(f"PB corrected {pb_count} images for {step['suffix']}")

    # ------------------------------------------------------------------
    #  8. Archive to Lustre
    # ------------------------------------------------------------------
    archive_base = os.path.join(
        LUSTRE_ARCHIVE_DIR,
        lst_label, obs_date, run_label, subband,
    )
    archive_results(
        work_dir, archive_base,
        cleanup_concat=not skip_cleanup,
        cleanup_workdir=cleanup_nvme,
    )

    logger.info(f"[{self.request.id}] Phase 2 COMPLETE: {subband} → {archive_base}")
    return archive_base


# ============================================================================
#  Convenience: submit an entire subband as a chord
# ============================================================================

def submit_subband_pipeline(
    ms_files: List[str],
    subband: str,
    bp_table: str,
    xy_table: str,
    lst_label: str,
    obs_date: str,
    run_label: str,
    peel_sky: bool = False,
    peel_rfi: bool = False,
    hot_baselines: bool = False,
    skip_cleanup: bool = False,
    cleanup_nvme: bool = False,
    nvme_work_dir: Optional[str] = None,
    queue_override: Optional[str] = None,
) -> 'celery.result.AsyncResult':
    """Submit the full two-phase subband pipeline as a Celery chord.

    This is the main entry point for the submission script.

    Args:
        ms_files: List of source MS paths on Lustre.
        subband: Frequency label, e.g. '73MHz'.
        bp_table: Bandpass calibration table.
        xy_table: XY-phase calibration table.
        lst_label: LST label, e.g. '14h'.
        obs_date: Observation date 'YYYY-MM-DD'.
        run_label: Human-readable run identifier.
        peel_sky: Peel astronomical sky sources.
        peel_rfi: Peel RFI sources.
        hot_baselines: Run hot-baseline diagnostics.
        skip_cleanup: Keep intermediate files on NVMe.
        cleanup_nvme: Remove entire NVMe work_dir after archiving to Lustre.
        nvme_work_dir: Override NVMe work directory.
        queue_override: Force routing to this queue instead of the
            default node.  E.g. 'calim08' to run 18MHz on calim08.

    Returns:
        Celery AsyncResult for the chord (Phase 2 result).
    """
    queue = queue_override or get_queue_for_subband(subband)

    if nvme_work_dir is None:
        nvme_work_dir = os.path.join(
            NVME_BASE_DIR, lst_label, obs_date, run_label, subband,
        )

    # Phase 1: one task per MS, all routed to the same node queue
    phase1_tasks = [
        prepare_one_ms_task.s(
            src_ms=ms,
            nvme_work_dir=nvme_work_dir,
            bp_table=bp_table,
            xy_table=xy_table,
            peel_sky=peel_sky,
            peel_rfi=peel_rfi,
        ).set(queue=queue)
        for ms in ms_files
    ]

    # Phase 2: runs after all Phase 1 tasks complete
    phase2_callback = process_subband_task.s(
        work_dir=nvme_work_dir,
        subband=subband,
        lst_label=lst_label,
        obs_date=obs_date,
        run_label=run_label,
        hot_baselines=hot_baselines,
        skip_cleanup=skip_cleanup,
        cleanup_nvme=cleanup_nvme,
    ).set(queue=queue)

    # chord(Phase1)(Phase2) — Phase2 receives list of Phase1 return values
    pipeline = chord(phase1_tasks)(phase2_callback)
    logger.info(
        f"Submitted {subband} → queue={queue}: "
        f"{len(ms_files)} MS files, work_dir={nvme_work_dir}"
    )
    return pipeline


# ============================================================================
#  Internal helpers
# ============================================================================

def _run_hot_baseline_diagnostics(concat_ms: str, work_dir: str) -> None:
    """Run hot-baseline heatmap and UV diagnostics.

    Runs the hot-baseline analysis using orca.transform.hot_baselines.
    """
    from orca.resources.subband_config import HOT_BASELINE_PARAMS

    try:
        from orca.transform import hot_baselines
    except ImportError:
        logger.warning("orca.transform.hot_baselines not importable; skipping diagnostics")
        return

    class HotArgs:
        ms = concat_ms
        col = "CORRECTED_DATA"
        uv_cut = 0.0
        uv_cut_lambda = HOT_BASELINE_PARAMS['uv_cut_lambda']
        sigma = HOT_BASELINE_PARAMS['heatmap_sigma']
        uv_sigma = HOT_BASELINE_PARAMS['uv_sigma']
        threshold = HOT_BASELINE_PARAMS['bad_antenna_threshold']
        apply_antenna_flags = HOT_BASELINE_PARAMS['apply_flags']
        apply_baseline_flags = HOT_BASELINE_PARAMS['apply_flags']
        run_uv = HOT_BASELINE_PARAMS['run_uv_analysis']
        run_heatmap = HOT_BASELINE_PARAMS['run_heatmap_analysis']

    qa_dir = os.path.join(work_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(qa_dir)
    try:
        hot_baselines.run_diagnostics(HotArgs, logger)
    finally:
        os.chdir(cwd)
