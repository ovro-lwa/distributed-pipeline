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
# Optional — set to None to skip
TARGETS = None  # e.g. ["/path/to/10pc_sample.csv"]
CATALOG = None  # e.g. "/path/to/OVRO_LWA_Local_Volume_Targets.csv"
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
    _t = time.time()
    logger.info("--- Phase 0: PB Correction ---")
    try:
        from orca.transform.pb_correction import apply_pb_correction
        count = 0
        raw_images = [f for f in all_fits
                      if "image" in f and "pbcorr" not in f and "dewarped" not in f]
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

    # ── B2. Solar System Photometry ─────────────────────────────────
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
