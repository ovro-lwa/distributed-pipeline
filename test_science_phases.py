#!/usr/bin/env python3
"""Quick test: re-run just the science phases on already-archived images.

Usage (from a calim node with orca installed):
    python test_science_phases.py

This bypasses Phase 1 + imaging entirely — it works on the images
already sitting in the Lustre archive from a previous run.
"""
import os
import sys
import glob
import time
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger("test_science")

# ── Point this at your existing archived run directory ──────────────
ARCHIVE_DIR = "/lustre/pipeline/images/10h/2024-12-18/Run_20260215_011016/55MHz"
SUBBAND = "55MHz"
FREQ_MHZ = 55.0
VLSSR_CATALOG = "/lustre/gh/calibration/pipeline/reference/surveys/FullVLSSCatalog.text"

# Target lists for photometry (Phase B)
_RES = os.path.join(os.path.dirname(__file__), "orca", "resources")
TARGETS = [
    os.path.join(_RES, "10pc_sample.csv"),
    os.path.join(_RES, "OVRO_LWA_Hot_Warm_Jupiters_2026.csv"),
]
# Source catalog for transient search masking (Phase C)
CATALOG = os.path.join(_RES, "OVRO_LWA_Local_Volume_Targets.csv")
# ────────────────────────────────────────────────────────────────────

work_dir = ARCHIVE_DIR

def main():
    logger.info(f"=== Science Phase Test on {work_dir} ===")

    # Quick sanity check — list what images exist
    all_fits = glob.glob(os.path.join(work_dir, "**", "*.fits"), recursive=True)
    logger.info(f"Found {len(all_fits)} FITS files total")
    for f in sorted(all_fits)[:20]:
        logger.info(f"  {os.path.relpath(f, work_dir)}")
    if len(all_fits) > 20:
        logger.info(f"  ... and {len(all_fits) - 20} more")

    # ── 0. PB Correction ────────────────────────────────────────────
    # ExoPipe only PB-corrects the 7 science imaging outputs
    # (I/deep, I/10min, V/deep, V/10min), NOT snapshots.
    _t = time.time()
    logger.info("--- Phase 0: PB Correction ---")
    try:
        from orca.transform.pb_correction import apply_pb_correction
        count = 0
        science_dirs = [os.path.join(work_dir, p, c)
                        for p in ("I", "V") for c in ("deep", "10min")]
        raw_images = []
        for sd in science_dirs:
            if os.path.isdir(sd):
                raw_images.extend(
                    f for f in glob.glob(os.path.join(sd, "*image*.fits"))
                    if "pbcorr" not in f and "dewarped" not in f)
        logger.info(f"Found {len(raw_images)} science images for PB correction "
                     f"(excluding snapshots)")
        for img in raw_images:
            try:
                result = apply_pb_correction(img)
                if result:
                    count += 1
            except Exception as e:
                logger.error(f"  PB failed: {os.path.basename(img)}: {e}")
        logger.info(f"PB corrected {count}/{len(raw_images)} images")
    except Exception as e:
        logger.error(f"PB correction import/run failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] pb_correction: {time.time() - _t:.1f}s")

    # ── A. Ionospheric Dewarping ────────────────────────────────────
    _t = time.time()
    logger.info("--- Science A: Ionospheric Dewarping (VLSSr) ---")
    try:
        from orca.transform.ionospheric_dewarping import (
            load_ref_catalog, generate_warp_screens, apply_warp,
        )
        from orca.transform.subband_processing import (
            find_deep_image, extract_sources_to_df,
        )
        from astropy.io import fits
        from astropy.wcs import WCS

        vlssr = load_ref_catalog(VLSSR_CATALOG, "VLSSr")
        calc_img = find_deep_image(work_dir, FREQ_MHZ, 'I')
        logger.info(f"Deep I image: {calc_img}")

        if calc_img and vlssr:
            df = extract_sources_to_df(calc_img)
            logger.info(f"Extracted {len(df)} sources for dewarping")
            if not df.empty:
                with fits.open(calc_img) as h:
                    wcs_calc = WCS(h[0].header).celestial
                    calc_shape = h[0].data.squeeze().shape
                    bmaj_deg = h[0].header.get('BMAJ', 5.0 / 60.0)

                diag_dir = os.path.join(work_dir, "Dewarp_Diagnostics")
                os.makedirs(diag_dir, exist_ok=True)
                warp_base = os.path.join(diag_dir, f"{SUBBAND}_warp")

                prev_cwd = os.getcwd()
                os.chdir(diag_dir)
                try:
                    sx, sy, _, _ = generate_warp_screens(
                        df, vlssr, wcs_calc, calc_shape,
                        FREQ_MHZ, 74.0, bmaj_deg, 5.0,
                        base_name=warp_base,
                    )
                finally:
                    os.chdir(prev_cwd)

                if sx is not None:
                    files_to_warp = glob.glob(
                        os.path.join(work_dir, "*", "*", "*pbcorr*.fits"))
                    files_to_warp = [f for f in files_to_warp if "_dewarped" not in f]
                    raw = glob.glob(
                        os.path.join(work_dir, "*", "*", "*image*.fits"))
                    raw = [f for f in raw if "pbcorr" not in f and "_dewarped" not in f]
                    files_to_warp.extend(raw)

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
                                        fits.writeto(out, warped, hf[0].header, overwrite=True)
                                        n_warped += 1
                        except Exception:
                            pass
                    logger.info(f"Dewarped {n_warped}/{len(files_to_warp)} images")
                else:
                    logger.warning("Warp screen generation failed.")
            else:
                logger.warning("No sources extracted.")
        else:
            logger.warning(f"calc_img={calc_img}, vlssr={'loaded' if vlssr else 'None'}")
    except Exception as e:
        logger.error(f"Dewarping failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_dewarping: {time.time() - _t:.1f}s")

    # ── B. Target Photometry ────────────────────────────────────────
    _t = time.time()
    logger.info("--- Science B: Target Photometry ---")
    if TARGETS:
        try:
            from orca.transform.cutout import (
                load_targets as _load_targets,
                process_target as _process_target,
            )
            local_samples = os.path.join(work_dir, "samples")
            local_detects = os.path.join(work_dir, "detections")
            os.makedirs(local_samples, exist_ok=True)
            os.makedirs(local_detects, exist_ok=True)

            for t_file in TARGETS:
                if not os.path.exists(t_file):
                    logger.warning(f"Target file not found: {t_file}")
                    continue
                logger.info(f"Processing target file: {t_file}")
                s_name = os.path.splitext(os.path.basename(t_file))[0]
                target_list = _load_targets(t_file)
                logger.info(f"  Loaded {len(target_list)} targets from {os.path.basename(t_file)}")
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
            logger.error(f"Target photometry failed: {e}")
            traceback.print_exc()
    else:
        logger.info("No target files — skipping.")
    logger.info(f"[TIMER] science_target_photometry: {time.time() - _t:.1f}s")

    # ── B2. Solar System Body Photometry ────────────────────────────
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
    except Exception as e:
        logger.error(f"Solar system failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_solar_system: {time.time() - _t:.1f}s")

    # ── C. Transient Search ─────────────────────────────────────────
    _t = time.time()
    logger.info("--- Science C: Transient Search ---")
    if CATALOG:
        try:
            from orca.transform.transient_search import run_test as _run_test

            local_detects = os.path.join(work_dir, "detections")
            os.makedirs(local_detects, exist_ok=True)

            def _find_transient_images(pol, category):
                pat = os.path.join(work_dir, pol, category, "*Taper*_dewarped.fits")
                imgs = [f for f in glob.glob(pat)
                        if "pbcorr" not in f and "_dewarped_dewarped" not in f]
                if not imgs:
                    pat = os.path.join(work_dir, pol, category, "*Taper*image*.fits")
                    imgs = [f for f in glob.glob(pat)
                            if "pbcorr" not in f and "dewarped" not in f]
                return [f for f in imgs if "NoTaper" not in os.path.basename(f)]

            ref_i_imgs = [f for f in _find_transient_images("I", "deep")
                          if "Robust-0-" in os.path.basename(f)]
            ref_i_path = ref_i_imgs[0] if ref_i_imgs else None
            if ref_i_path:
                logger.info(f"Deep I reference: {os.path.basename(ref_i_path)}")

            # Stokes V blind search
            logger.info("Running Stokes V Blind Search...")
            v_imgs = _find_transient_images("V", "deep") + _find_transient_images("V", "10min")
            v_det = 0
            for v_img in v_imgs:
                try:
                    result = _run_test(None, v_img, ref_i_path, CATALOG,
                                       output_dir=local_detects)
                    if result:
                        v_det += len(result) if isinstance(result, list) else 1
                except Exception as e:
                    logger.error(f"V search failed: {os.path.basename(v_img)}: {e}")

            # Stokes I subtraction search
            logger.info("Running Stokes I Subtraction Search...")
            i_snaps = _find_transient_images("I", "10min")
            i_det = 0
            if ref_i_path:
                for i_img in i_snaps:
                    try:
                        result = _run_test(ref_i_path, i_img, ref_i_path, CATALOG,
                                           output_dir=local_detects)
                        if result:
                            i_det += len(result) if isinstance(result, list) else 1
                    except Exception as e:
                        logger.error(f"I search failed: {os.path.basename(i_img)}: {e}")

            logger.info(f"Transient candidates: {v_det} Stokes V, {i_det} Stokes I")
        except Exception as e:
            logger.error(f"Transient search failed: {e}")
            traceback.print_exc()
    else:
        logger.info("No catalog — skipping transient search.")
    logger.info(f"[TIMER] science_transient_search: {time.time() - _t:.1f}s")

    # ── D. Flux Scale Check ─────────────────────────────────────────
    _t = time.time()
    logger.info("--- Science D: Flux Scale Check ---")
    try:
        from orca.transform.flux_check_cutout import run_flux_check
        run_flux_check(work_dir, logger=logger)
    except Exception as e:
        logger.error(f"Flux check failed: {e}")
        traceback.print_exc()
    logger.info(f"[TIMER] science_flux_check: {time.time() - _t:.1f}s")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
