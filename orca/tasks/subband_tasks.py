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
import time
import traceback

# WSClean (linked against OpenBLAS) refuses to start if OpenBLAS multi-
# threading is enabled.  Setting this early ensures every subprocess inherits it.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from typing import List, Optional

import numpy as np
from celery import chord
from casacore.tables import table
from astropy.io import fits

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
    find_deep_image,
    extract_sources_to_df,
)
from orca.resources.subband_config import (
    PEELING_PARAMS,
    AOFLAGGER_STRATEGY,
    SNAPSHOT_PARAMS,
    SNAPSHOT_CLEAN_PARAMS,
    IMAGING_STEPS,
    NVME_BASE_DIR,
    LUSTRE_ARCHIVE_DIR,
    VLSSR_CATALOG,
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
    _p1_t0 = time.time()
    logger.info(
        f"[{self.request.id}] Phase 1 START on {node}: {os.path.basename(src_ms)}"
    )

    os.makedirs(nvme_work_dir, exist_ok=True)
    redirect_casa_log(nvme_work_dir)

    # 1. Copy to NVMe
    _t = time.time()
    nvme_ms = copy_ms_to_nvme(src_ms, nvme_work_dir)
    logger.info(f"[TIMER] copy_to_nvme: {time.time() - _t:.1f}s")

    # 2. Flag bad antennas (via mnc_python)
    _t = time.time()
    flag_bad_antennas(nvme_ms)
    logger.info(f"[TIMER] flag_bad_antennas: {time.time() - _t:.1f}s")

    # 3. Apply calibration (bandpass + XY-phase)
    _t = time.time()
    cal_ok = apply_calibration(nvme_ms, bp_table, xy_table)
    logger.info(f"[TIMER] apply_calibration: {time.time() - _t:.1f}s")
    if not cal_ok:
        logger.error(f"Calibration failed for {os.path.basename(nvme_ms)}; removing")
        shutil.rmtree(nvme_ms, ignore_errors=True)
        raise RuntimeError(f"Calibration failed for {src_ms}")

    # 4. Peeling
    if peel_sky:
        _t = time.time()
        logger.info(f"Peeling sky model on {os.path.basename(nvme_ms)}")
        zest_with_ttcal(
            ms=nvme_ms,
            sources=PEELING_PARAMS['sky_model'],
            beam=PEELING_PARAMS['beam'],
            minuvw=PEELING_PARAMS['minuvw'],
            maxiter=PEELING_PARAMS['maxiter'],
            tolerance=PEELING_PARAMS['tolerance'],
        )
        logger.info(f"[TIMER] peel_sky: {time.time() - _t:.1f}s")

    if peel_rfi:
        _t = time.time()
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
        logger.info(f"[TIMER] peel_rfi: {time.time() - _t:.1f}s")

    logger.info(
        f"[{self.request.id}] Phase 1 DONE: {os.path.basename(nvme_ms)}"
    )
    logger.info(f"[TIMER] phase1_total: {time.time() - _p1_t0:.1f}s ({os.path.basename(nvme_ms)})")
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
    targets: Optional[List[str]] = None,
    catalog: Optional[str] = None,
    snapshot_clean: bool = False,
) -> str:
    """Phase 2: concatenate, image, run science, and archive one subband.

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
        targets: List of target-list file paths for photometry.
        catalog: Path to BDSF catalog for transient search masking.
        snapshot_clean: If True, use CLEAN imaging for pilot snapshots.

    Returns:
        Path to the Lustre archive directory with final products.
    """
    node = socket.gethostname()
    _p2_t0 = time.time()
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
    for d in ['I/deep', 'V/deep', 'I/10min', 'V/10min', 'snapshots', 'QA',
              'samples', 'detections', 'Dewarp_Diagnostics']:
        os.makedirs(os.path.join(work_dir, d), exist_ok=True)

    # ------------------------------------------------------------------
    #  1. Concatenation  (skip if concat MS already exists from prior attempt)
    # ------------------------------------------------------------------
    _t = time.time()
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
    logger.info(f"[TIMER] concatenation: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  2. Fix FIELD_ID
    # ------------------------------------------------------------------
    _t = time.time()
    fix_field_id(concat_ms)
    logger.info(f"[TIMER] fix_field_id: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  3. Change phase centre
    # ------------------------------------------------------------------
    _t = time.time()
    hour_int = int(lst_label.replace('h', ''))
    phase_center = f"{hour_int:02d}h30m00s 37d12m57.057s"
    logger.info(f"Changing phase centre → {phase_center}")
    change_phase_center(concat_ms, phase_center)
    logger.info(f"[TIMER] chgcentre: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  4. AOFlagger
    # ------------------------------------------------------------------
    _t = time.time()
    aoflagger_bin = os.environ.get('AOFLAGGER_BIN', '/opt/bin/aoflagger')
    logger.info(f"Running AOFlagger with strategy {AOFLAGGER_STRATEGY}")
    run_subprocess(
        [aoflagger_bin, '-strategy', AOFLAGGER_STRATEGY, concat_ms],
        "AOFlagger (Post-Concat)",
    )
    logger.info(f"[TIMER] aoflagger: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  5. Pilot snapshots + QA
    # ------------------------------------------------------------------
    _t = time.time()
    try:
        t = table(concat_ms, ack=False)
        times = t.getcol("TIME")
        n_ints = len(np.unique(times))
        t.close()
    except Exception:
        n_ints = 357

    pilot_name = f"{subband}-{SNAPSHOT_PARAMS['suffix']}"
    pilot_path = os.path.join(work_dir, "snapshots", pilot_name)

    snapshot_cfg = SNAPSHOT_CLEAN_PARAMS if snapshot_clean else SNAPSHOT_PARAMS
    wsclean_bin = os.environ.get('WSCLEAN_BIN', '/opt/bin/wsclean')
    _, _, wsclean_j = get_image_resources(subband)
    cmd_pilot = (
        [wsclean_bin]
        + ['-j', str(wsclean_j)]
        + snapshot_cfg['args']
        + ['-name', pilot_path, '-intervals-out', str(n_ints), concat_ms]
    )
    run_subprocess(cmd_pilot, "Pilot snapshot imaging")

    add_timestamps_to_images(
        os.path.join(work_dir, "snapshots"), pilot_name, concat_ms, n_ints,
    )

    pilot_v = sorted(glob.glob(
        os.path.join(work_dir, "snapshots", f"{pilot_name}*-V-image*.fits")
    ))
    bad_idx, stats = analyze_snapshot_quality(pilot_v)
    plot_snapshot_diagnostics(stats, bad_idx, work_dir, subband)

    if bad_idx:
        flag_bad_integrations(concat_ms, bad_idx, n_ints)
    logger.info(f"[TIMER] pilot_snapshots_qa: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  6. Hot baseline removal (optional)
    # ------------------------------------------------------------------
    if hot_baselines:
        _t = time.time()
        try:
            _run_hot_baseline_diagnostics(concat_ms, work_dir)
        except Exception as e:
            logger.error(f"Hot baseline diagnostics failed: {e}")
            traceback.print_exc()
        logger.info(f"[TIMER] hot_baselines: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  7. Science imaging + PB correction
    # ------------------------------------------------------------------
    _t_imaging_all = time.time()
    logger.info(f"Starting Science Imaging for {subband}...")
    logger.info(f"wsclean thread limit: -j {wsclean_j}")

    for step in IMAGING_STEPS:
        _t_step = time.time()
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
        logger.info(f"[TIMER] imaging_{step['suffix']}: {time.time() - _t_step:.1f}s")
    logger.info(f"[TIMER] imaging_all: {time.time() - _t_imaging_all:.1f}s")

    # ------------------------------------------------------------------
    #  7b. SCIENCE PHASES (all on NVMe)
    # ------------------------------------------------------------------
    try:
        freq_mhz = float(subband.replace('MHz', ''))
    except Exception:
        freq_mhz = 50.0

    # --- A. Ionospheric Dewarping (VLSSr cross-match) ---
    _t = time.time()
    logger.info("--- Science A: Ionospheric Dewarping (VLSSr) ---")
    try:
        from orca.transform.ionospheric_dewarping import (
            load_ref_catalog, generate_warp_screens, apply_warp,
        )
        from astropy.wcs import WCS as _WCS

        vlssr = load_ref_catalog(VLSSR_CATALOG, "VLSSr")
        # Find all PB-corrected AND raw images to dewarp
        files_to_warp = glob.glob(
            os.path.join(work_dir, "*", "*", "*pbcorr*.fits"))
        files_to_warp = [f for f in files_to_warp
                         if "_dewarped" not in f]
        raw_images = glob.glob(
            os.path.join(work_dir, "*", "*", "*image*.fits"))
        raw_images = [f for f in raw_images
                      if "pbcorr" not in f and "_dewarped" not in f]
        files_to_warp.extend(raw_images)

        calc_img = find_deep_image(work_dir, freq_mhz, 'I')

        if calc_img and vlssr:
            df = extract_sources_to_df(calc_img)
            if not df.empty:
                with fits.open(calc_img) as h:
                    wcs_calc = _WCS(h[0].header).celestial
                    calc_shape = h[0].data.squeeze().shape
                    bmaj_deg = h[0].header.get('BMAJ', 5.0 / 60.0)

                diag_dir = os.path.join(work_dir, "Dewarp_Diagnostics")
                os.makedirs(diag_dir, exist_ok=True)
                warp_base = os.path.join(diag_dir, f"{subband}_warp")

                prev_cwd = os.getcwd()
                os.chdir(diag_dir)
                try:
                    sx, sy, _, _ = generate_warp_screens(
                        df, vlssr, wcs_calc, calc_shape,
                        freq_mhz, 74.0,
                        bmaj_deg, 5.0, base_name=warp_base,
                    )
                finally:
                    os.chdir(prev_cwd)

                if sx is not None:
                    n_warped = 0
                    for f in files_to_warp:
                        out = f.replace('.fits', '_dewarped.fits')
                        if os.path.exists(out):
                            continue
                        try:
                            with fits.open(f) as hf:
                                fdata = hf[0].data.squeeze()
                                if fdata.shape == sx.shape:
                                    warped = apply_warp(fdata, sx, sy)
                                    if warped is not None:
                                        fits.writeto(
                                            out, warped, hf[0].header,
                                            overwrite=True)
                                        n_warped += 1
                        except Exception:
                            pass
                    logger.info(f"Dewarped {n_warped}/{len(files_to_warp)} images")
                else:
                    logger.warning("Warp screen generation failed — skipping dewarping.")
            else:
                logger.warning("No sources extracted for dewarping.")
        else:
            logger.warning("No deep I image or VLSSr catalog — skipping dewarping.")
    except ImportError as e:
        logger.warning(f"Dewarping modules not available — skipping: {e}")
    except Exception as e:
        logger.error(f"Dewarping failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_dewarping: {time.time() - _t:.1f}s")

    # --- B. Target Photometry ---
    _t = time.time()
    logger.info("--- Science B: Target Photometry ---")
    if targets:
        try:
            from orca.transform.cutout import (
                load_targets as _load_targets,
                process_target as _process_target,
            )
            local_samples = os.path.join(work_dir, "samples")
            local_detects = os.path.join(work_dir, "detections")
            os.makedirs(local_samples, exist_ok=True)
            os.makedirs(local_detects, exist_ok=True)

            for t_file in targets:
                if not os.path.exists(t_file):
                    logger.warning(f"Target file not found: {t_file}")
                    continue
                logger.info(f"Processing target file: {t_file}")
                try:
                    s_name = os.path.splitext(os.path.basename(t_file))[0]
                    target_list = _load_targets(t_file)
                    logger.info(
                        f"  Loaded {len(target_list)} targets from "
                        f"{os.path.basename(t_file)}")
                    for nm, crd, det_stokes, confusing_sources in target_list:
                        try:
                            _process_target(
                                work_dir, nm, crd, s_name,
                                local_samples, local_detects,
                                fallback_dir=work_dir,
                                detection_stokes=det_stokes,
                                confusing_sources=confusing_sources,
                            )
                        except Exception as e:
                            logger.error(f"  Target '{nm}' failed: {e}")
                except Exception as e:
                    logger.error(f"Failed to process target file {t_file}: {e}")
                    traceback.print_exc()
        except ImportError as e:
            logger.warning(f"cutout module not available — skipping target photometry: {e}")
    else:
        logger.info("No target files specified — skipping target photometry.")
    logger.info(f"[TIMER] science_target_photometry: {time.time() - _t:.1f}s")

    # --- B2. Solar System Body Photometry ---
    _t = time.time()
    logger.info("--- Science B2: Solar System Photometry ---")
    try:
        from orca.transform.solar_system_cutout import process_solar_system
        local_samples = os.path.join(work_dir, "samples")
        local_detects = os.path.join(work_dir, "detections")
        os.makedirs(local_samples, exist_ok=True)
        os.makedirs(local_detects, exist_ok=True)
        process_solar_system(
            work_dir, local_samples, local_detects,
            fallback_dir=work_dir, logger=logger,
        )
    except ImportError as e:
        logger.warning(f"solar_system_cutout not available — skipping: {e}")
    except Exception as e:
        logger.error(f"Solar system photometry failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_solar_system: {time.time() - _t:.1f}s")

    # --- C. Transient Search ---
    _t = time.time()
    logger.info("--- Science C: Transient Search ---")
    if catalog:
        try:
            from orca.transform.transient_search import run_test as _run_test

            local_transient_detections = os.path.join(
                work_dir, "detections")
            os.makedirs(local_transient_detections, exist_ok=True)

            def _find_transient_images(pol, category, suffix_filter=None):
                """Find tapered, optionally dewarped, non-pbcorr images."""
                pat = os.path.join(
                    work_dir, pol, category, "*Taper*_dewarped.fits")
                imgs = [f for f in glob.glob(pat)
                        if "pbcorr" not in f
                        and "_dewarped_dewarped" not in f]
                if not imgs:
                    pat = os.path.join(
                        work_dir, pol, category, "*Taper*image*.fits")
                    imgs = [f for f in glob.glob(pat)
                            if "pbcorr" not in f and "dewarped" not in f]
                if suffix_filter:
                    filtered = [f for f in imgs
                                if suffix_filter in os.path.basename(f)
                                and "NoTaper" not in os.path.basename(f)]
                    if filtered:
                        imgs = filtered
                return imgs

            # Deep I reference (Robust-0) for masking + subtraction
            ref_i_imgs = _find_transient_images(
                "I", "deep", suffix_filter="Robust-0-")
            ref_i_path = ref_i_imgs[0] if ref_i_imgs else None
            if ref_i_path:
                logger.info(
                    f"Deep I reference: {os.path.basename(ref_i_path)}")

            # Stokes V: blind search (no subtraction)
            logger.info("Running Stokes V Blind Search...")
            v_deep = _find_transient_images("V", "deep")
            v_10min = _find_transient_images("V", "10min")
            v_detections = []
            for v_img in v_deep + v_10min:
                try:
                    result = _run_test(
                        None, v_img, ref_i_path, catalog,
                        output_dir=local_transient_detections)
                    if result:
                        v_detections.extend(
                            result if isinstance(result, list) else [result])
                except Exception as e:
                    logger.error(
                        f"V transient search failed on "
                        f"{os.path.basename(v_img)}: {e}")

            # Stokes I: subtract deep from 10min snapshots
            logger.info("Running Stokes I Subtraction Search...")
            i_snaps = _find_transient_images("I", "10min")
            i_detections = []
            if ref_i_path:
                for i_img in i_snaps:
                    try:
                        result = _run_test(
                            ref_i_path, i_img, ref_i_path, catalog,
                            output_dir=local_transient_detections)
                        if result:
                            i_detections.extend(
                                result if isinstance(result, list)
                                else [result])
                    except Exception as e:
                        logger.error(
                            f"I transient search failed on "
                            f"{os.path.basename(i_img)}: {e}")

            total_det = len(v_detections) + len(i_detections)
            logger.info(
                f"Transient candidates: {len(v_detections)} Stokes V, "
                f"{len(i_detections)} Stokes I")
            if total_det > 10:
                logger.warning(
                    f"QUALITY FLAG: {total_det} candidates — "
                    f"data quality may be poor.")
        except ImportError as e:
            logger.warning(
                f"transient_search not available — skipping: {e}")
        except Exception as e:
            logger.error(f"Transient search failed: {e}")
            traceback.print_exc()
    else:
        logger.info("No catalog specified — skipping transient search.")
    logger.info(f"[TIMER] science_transient_search: {time.time() - _t:.1f}s")

    # --- D. Flux Scale Check ---
    _t = time.time()
    logger.info("--- Science D: Flux Scale Check ---")
    try:
        from orca.transform.flux_check_cutout import run_flux_check
        run_flux_check(work_dir, logger=logger)
    except ImportError as e:
        logger.warning(f"flux_check_cutout not available — skipping: {e}")
    except Exception as e:
        logger.error(f"Flux check failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_flux_check: {time.time() - _t:.1f}s")

    # ------------------------------------------------------------------
    #  8. Archive to Lustre
    # ------------------------------------------------------------------
    _t = time.time()
    archive_base = os.path.join(
        LUSTRE_ARCHIVE_DIR,
        lst_label, obs_date, run_label, subband,
    )
    archive_results(
        work_dir, archive_base,
        subband=subband,
        cleanup_concat=not skip_cleanup,
        cleanup_workdir=cleanup_nvme,
    )
    logger.info(f"[TIMER] archive_to_lustre: {time.time() - _t:.1f}s")

    logger.info(f"[TIMER] phase2_total: {time.time() - _p2_t0:.1f}s")
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
    targets: Optional[List[str]] = None,
    catalog: Optional[str] = None,
    snapshot_clean: bool = False,
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
        targets: List of target-list file paths for photometry.
        catalog: Path to BDSF catalog for transient search masking.
        snapshot_clean: If True, use CLEAN imaging for pilot snapshots.

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
        targets=targets,
        catalog=catalog,
        snapshot_clean=snapshot_clean,
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
        uv_window_size = HOT_BASELINE_PARAMS.get('uv_window_size', 100)
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
