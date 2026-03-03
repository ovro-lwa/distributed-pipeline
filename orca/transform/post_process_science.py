"""Phase 3 — Post-subband wideband science aggregation.

Runs AFTER all per-subband workers complete.  Operates on Lustre.

Steps
-----
1. Wideband stacking: inverse-variance weighted co-adds across subbands
   (Red 18–41 MHz, Green 41–64 MHz, Blue 64–85 MHz).
2. 3-colour PNG composites.
3. Wideband transient search on stacked images.
4. Wideband solar system photometry.
5. Detection gathering (target, transient, solar system CSVs from all bands).
6. Email summary report.

Ported from ``ExoPipe/post_process_science.py`` into the orca package.
"""

import os
import re
import glob
import json
import shutil
import logging
import smtplib
import traceback

import numpy as np
import pandas as pd
from astropy.io import fits
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
EMAIL_RECIPIENT = "gh@astro.caltech.edu"
OUTPUT_CAT_DIR = "/lustre/gh/main/catalogs/"
SECRETS_FILE = os.path.expanduser("~/pipeline_cred.json")

WIDEBAND_BANDS = {
    'Red':   (18, 41),
    'Green': (41, 64),
    'Blue':  (64, 85),
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_inner_rms(fits_path: str) -> float:
    """RMS of the inner 25 % (50 % per side) of a FITS image."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.squeeze()
            h, w = data.shape
            ch, cw = h // 2, w // 2
            rh, rw = h // 4, w // 4
            region = data[ch - rh:ch + rh, cw - rw:cw + rw]
            return float(np.nanstd(region))
    except Exception as e:
        logger.warning(f"RMS failed for {os.path.basename(fits_path)}: {e}")
        return np.nan


def _stack_images(img_list, weights, out_path, history_str):
    """Inverse-variance weighted co-add of FITS images."""
    if not img_list:
        return False
    try:
        data_sum = None
        weight_sum = 0.0
        ref_header = None

        for img, w in zip(img_list, weights):
            with fits.open(img) as hdul:
                data = hdul[0].data.squeeze()
                if data_sum is None:
                    data_sum = np.zeros_like(data, dtype=np.float64)
                    ref_header = hdul[0].header.copy()
                if data.shape != data_sum.shape:
                    continue
                data_sum += data * w
                weight_sum += w

        if weight_sum > 0:
            final_data = data_sum / weight_sum
            ref_header['BTYPE'] = 'Intensity'
            ref_header['HISTORY'] = history_str
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fits.writeto(out_path, final_data.astype(np.float32),
                         ref_header, overwrite=True)
            return True
    except Exception as e:
        logger.error(f"Stack failed for {out_path}: {e}")
    return False


def _make_3color_png(red_path, green_path, blue_path, out_png, title=""):
    """Generate a 3-colour PNG from Red/Green/Blue FITS images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        channels = []
        for fpath in [red_path, green_path, blue_path]:
            if fpath and os.path.exists(fpath):
                with fits.open(fpath) as hdul:
                    channels.append(hdul[0].data.squeeze().astype(np.float64))
            else:
                channels.append(None)

        valid = [c for c in channels if c is not None]
        if len(valid) < 2:
            logger.warning(f"3-colour PNG needs ≥2 bands, got {len(valid)}")
            return

        ref_shape = valid[0].shape
        for i in range(3):
            if channels[i] is None:
                channels[i] = np.zeros(ref_shape, dtype=np.float64)
            elif channels[i].shape != ref_shape:
                channels[i] = np.zeros(ref_shape, dtype=np.float64)

        normed = []
        for ch in channels:
            vmin, vmax = np.nanpercentile(ch, [0.5, 99.8])
            if vmax <= vmin:
                vmax = vmin + 1
            clipped = np.clip(ch, vmin, vmax)
            normed.append((clipped - vmin) / (vmax - vmin))

        rgb = np.stack(normed, axis=-1)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.imshow(rgb, origin='lower', interpolation='nearest')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=14, color='white',
                         bbox=dict(facecolor='black', alpha=0.7))
        plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        logger.info(f"  Saved 3-colour PNG: {os.path.basename(out_png)}")
    except Exception as e:
        logger.error(f"3-colour PNG failed: {e}")


# ---------------------------------------------------------------------------
#  Step 1 — Wideband stacking
# ---------------------------------------------------------------------------

def run_wideband_stacking(run_dir, catalog_path=None):
    """Build inverse-variance-weighted wideband co-adds and run transient search."""
    logger.info("Starting Wideband Co-addition …")

    wb_dir = os.path.join(run_dir, "Wideband")
    os.makedirs(wb_dir, exist_ok=True)

    # --- 1. Noise analysis from Stokes V deep images ---
    v_images = glob.glob(
        os.path.join(run_dir, "*MHz", "V", "deep", "*_dewarped*.fits"))
    if not v_images:
        v_images = glob.glob(
            os.path.join(run_dir, "*MHz", "V", "deep", "*image*.fits"))
        v_images = [f for f in v_images
                    if "pbcorr" not in f and "dewarped" not in f]
    if not v_images:
        logger.warning("No V images found — cannot stack.")
        return {}

    logger.info(f"Analysing noise in {len(v_images)} sub-bands …")
    noise_data = []
    for v_img in v_images:
        try:
            freq_str = v_img.split('/')[-4].replace('MHz', '')
            freq = float(freq_str)
            rms = get_inner_rms(v_img)
            if np.isfinite(rms) and rms > 0:
                noise_data.append({'freq': freq, 'rms': rms,
                                   'weight': 1.0 / (rms ** 2)})
        except Exception:
            continue

    if not noise_data:
        logger.warning("No valid noise measurements — aborting stacking.")
        return {}

    df_noise = pd.DataFrame(noise_data)
    df_noise.to_csv(os.path.join(wb_dir, "thermal_noise.csv"), index=False)

    # --- 2. Define stacking targets ---
    transient_targets = [
        ('I', 'deep',  'Taper', 'Robust-0-',  '_dewarped', False, 'Robust-0'),
        ('I', '10min', 'Taper', '10min',       '_dewarped', False, 'Robust-0'),
        ('V', 'deep',  'Taper', 'Deep',        '_dewarped', False, 'Robust-0'),
        ('V', '10min', 'Taper', '10min',       '_dewarped', False, 'Robust-0'),
    ]
    science_targets = [
        ('I', 'deep', 'NoTaper', 'Robust-0-',   '_dewarped', False, 'Robust-0'),
        ('I', 'deep', 'NoTaper', 'Robust-0.75-', '_dewarped', False, 'Robust-0.75'),
    ]

    wideband_files = {}

    # --- 3. Stack per band ---
    for b_name, (f_min, f_max) in WIDEBAND_BANDS.items():
        subset = df_noise[(df_noise['freq'] >= f_min) & (df_noise['freq'] < f_max)]
        if subset.empty:
            continue

        logger.info(f"--- Stacking {b_name} ({f_min}–{f_max} MHz, "
                     f"{len(subset)} sub-bands) ---")

        for targets, is_science in [(transient_targets, False),
                                     (science_targets, True)]:
            for pol, cat, taper_kw, suffix_kw, file_ext, is_raw, robust_label in targets:
                img_list, weights = [], []
                for _, row in subset.iterrows():
                    freq_dir = os.path.join(
                        run_dir, f"{int(row['freq'])}MHz", pol, cat)
                    if not os.path.isdir(freq_dir):
                        continue

                    candidates = glob.glob(
                        os.path.join(freq_dir, f"*{suffix_kw}*{file_ext}*.fits"))
                    if not is_raw:
                        candidates = [
                            f for f in candidates
                            if "pbcorr" not in os.path.basename(f)
                            and "_dewarped_dewarped" not in os.path.basename(f)]
                    else:
                        candidates = [
                            f for f in candidates
                            if "pbcorr" not in os.path.basename(f)
                            and "dewarped" not in os.path.basename(f)]

                    if taper_kw == 'NoTaper':
                        candidates = [f for f in candidates
                                      if 'NoTaper' in os.path.basename(f)]
                    else:
                        candidates = [f for f in candidates
                                      if 'NoTaper' not in os.path.basename(f)]

                    if suffix_kw == 'Robust-0-':
                        candidates = [f for f in candidates
                                      if 'Robust-0.75' not in os.path.basename(f)]

                    if not candidates:
                        # Fallback: raw images if no dewarped
                        raw_pat = os.path.join(
                            freq_dir, f"*{suffix_kw}*image*.fits")
                        candidates = glob.glob(raw_pat)
                        candidates = [
                            f for f in candidates
                            if "pbcorr" not in os.path.basename(f)
                            and "dewarped" not in os.path.basename(f)]
                        if taper_kw == 'NoTaper':
                            candidates = [f for f in candidates
                                          if 'NoTaper' in os.path.basename(f)]
                        else:
                            candidates = [f for f in candidates
                                          if 'NoTaper' not in os.path.basename(f)]
                        if suffix_kw == 'Robust-0-':
                            candidates = [
                                f for f in candidates
                                if 'Robust-0.75' not in os.path.basename(f)]

                    if candidates:
                        if cat == '10min':
                            for c in sorted(candidates):
                                img_list.append(c)
                                weights.append(row['weight'])
                        else:
                            img_list.append(sorted(candidates)[0])
                            weights.append(row['weight'])

                if not img_list:
                    continue

                taper_str = "Taper" if taper_kw != 'NoTaper' else "NoTaper"

                if cat == '10min':
                    interval_groups = {}
                    for img, w in zip(img_list, weights):
                        bn = os.path.basename(img)
                        m = re.search(r'(t\d{4})', bn)
                        key = m.group(1) if m else 'all'
                        interval_groups.setdefault(key, ([], []))
                        interval_groups[key][0].append(img)
                        interval_groups[key][1].append(w)

                    for int_key, (int_imgs, int_w) in sorted(
                            interval_groups.items()):
                        out_name = (f"Wideband_{b_name}_{pol}_{cat}_"
                                    f"{taper_str}_{robust_label}_{int_key}.fits")
                        out_path = os.path.join(wb_dir, out_name)
                        history = (f"Wideband {b_name} ({f_min}–{f_max} MHz) "
                                   f"{pol} {cat} {taper_str} {robust_label} "
                                   f"{int_key}")
                        if _stack_images(int_imgs, int_w, out_path, history):
                            logger.info(f"  Stacked: {out_name} "
                                        f"({len(int_imgs)} images)")
                            wideband_files[(b_name, pol, cat, taper_str,
                                            robust_label, int_key)] = out_path
                else:
                    out_name = (f"Wideband_{b_name}_{pol}_{cat}_"
                                f"{taper_str}_{robust_label}.fits")
                    out_path = os.path.join(wb_dir, out_name)
                    history = (f"Wideband {b_name} ({f_min}–{f_max} MHz) "
                               f"{pol} {cat} {taper_str} {robust_label}")
                    if _stack_images(img_list, weights, out_path, history):
                        logger.info(f"  Stacked: {out_name} "
                                    f"({len(img_list)} images)")
                        wideband_files[(b_name, pol, cat, taper_str,
                                        robust_label)] = out_path

    # --- 4. 3-colour PNGs ---
    logger.info("Generating 3-colour PNGs …")
    attachment_files = []

    r_mixed = wideband_files.get(
        ('Red', 'I', 'deep', 'NoTaper', 'Robust-0.75'))
    g_mixed = wideband_files.get(
        ('Green', 'I', 'deep', 'NoTaper', 'Robust-0'))
    b_mixed = wideband_files.get(
        ('Blue', 'I', 'deep', 'NoTaper', 'Robust-0'))
    if any(p is not None for p in [r_mixed, g_mixed, b_mixed]):
        png_path = os.path.join(
            wb_dir, "Wideband_I_deep_NoTaper_mixed_3color.png")
        _make_3color_png(r_mixed, g_mixed, b_mixed, png_path,
                         title="Wideband I — Red:R-0.75 Green/Blue:R0")
        attachment_files.append(png_path)

    for robust_str in ['Robust-0', 'Robust-0.75']:
        r_p = wideband_files.get(('Red', 'I', 'deep', 'NoTaper', robust_str))
        g_p = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', robust_str))
        b_p = wideband_files.get(('Blue', 'I', 'deep', 'NoTaper', robust_str))
        if any(p is not None for p in [r_p, g_p, b_p]):
            png_name = f"Wideband_I_deep_NoTaper_{robust_str}_3color.png"
            png_path = os.path.join(wb_dir, png_name)
            _make_3color_png(r_p, g_p, b_p, png_path,
                             title=f"Wideband I NoTaper {robust_str}")
            attachment_files.append(png_path)

    # --- 5. Wideband transient search ---
    try:
        from orca.transform.transient_search import run_test as _run_test
    except ImportError:
        _run_test = None

    if _run_test and catalog_path:
        det_dir = os.path.join(run_dir, "detections")
        os.makedirs(det_dir, exist_ok=True)

        for b_name in WIDEBAND_BANDS:
            wb_i_deep = wideband_files.get(
                (b_name, 'I', 'deep', 'Taper', 'Robust-0'))

            # V deep
            wb_v_deep = wideband_files.get(
                (b_name, 'V', 'deep', 'Taper', 'Robust-0'))
            if wb_v_deep:
                logger.info(f"  Transient: WB {b_name} V deep …")
                try:
                    _run_test(None, wb_v_deep, wb_i_deep, catalog_path,
                              output_dir=det_dir, mode='V')
                except Exception as e:
                    logger.error(f"  WB V transient failed ({b_name}): {e}")

            # V 10min per interval
            for k in sorted(k for k in wideband_files
                            if len(k) >= 6 and k[:3] == (b_name, 'V', '10min')):
                try:
                    _run_test(None, wideband_files[k], wb_i_deep, catalog_path,
                              output_dir=det_dir, mode='V')
                except Exception as e:
                    logger.error(f"  WB V 10min failed ({b_name}): {e}")

            # I deep − 10min subtraction
            wb_i_deep_ref = wideband_files.get(
                (b_name, 'I', 'deep', 'Taper', 'Robust-0'))
            for k in sorted(k for k in wideband_files
                            if len(k) >= 6 and k[:3] == (b_name, 'I', '10min')):
                if wb_i_deep_ref:
                    try:
                        _run_test(wb_i_deep_ref, wideband_files[k],
                                  wb_i_deep_ref, catalog_path,
                                  output_dir=det_dir, mode='I')
                    except Exception as e:
                        logger.error(
                            f"  WB I transient failed ({b_name}): {e}")
    elif not catalog_path:
        logger.info("No catalog — skipping wideband transient search.")

    return wideband_files


# ---------------------------------------------------------------------------
#  Step 2 — Gather detections from all subbands
# ---------------------------------------------------------------------------

def gather_detections(run_dir):
    """Collect target photometry, transient, and solar system detections."""
    logger.info("Gathering detections …")
    report_lines = []
    attachment_files = []

    # A) Target photometry detections
    det_csvs = glob.glob(
        os.path.join(run_dir, "*MHz", "detections", "*_photometry.csv"))
    det_csvs.extend(glob.glob(
        os.path.join(run_dir, "Wideband", "*_photometry.csv")))
    for csv_path in det_csvs:
        try:
            df = pd.read_csv(csv_path)
            if 'Detection' in df.columns and df['Detection'].any():
                for _, row in df[df['Detection'] == True].iterrows():
                    tgt = row.get('Target', 'Unknown')
                    freq = row.get('Freq_MHz', 0)
                    flux = row.get('I_Src_Jy', row.get('I_Flux_Jy', 0))
                    report_lines.append(
                        f"DETECTED: {tgt} @ {freq}MHz ({flux:.3f} Jy)")
                    png = csv_path.replace(".csv", ".png")
                    if os.path.exists(png):
                        attachment_files.append(png)
        except Exception:
            pass

    # B) Transient detections
    for stokes in ['I', 'V']:
        dirs = glob.glob(os.path.join(
            run_dir, "*MHz", "detections", "transients", stokes, "*"))
        dirs.extend(glob.glob(os.path.join(
            run_dir, "detections", "transients", stokes, "*")))
        for jdir in dirs:
            if not os.path.isdir(jdir):
                continue
            jname = os.path.basename(jdir)
            n_files = len(os.listdir(jdir))
            if n_files > 0:
                report_lines.append(
                    f"TRANSIENT: Stokes {stokes} {jname} ({n_files} files)")
                pngs = sorted(glob.glob(os.path.join(jdir, "*.png")))
                if pngs and len(attachment_files) < 20:
                    attachment_files.append(pngs[0])

    # C) Solar system detections
    ss_dirs = glob.glob(os.path.join(
        run_dir, "*MHz", "detections", "SolarSystem", "*"))
    ss_dirs.extend(glob.glob(os.path.join(
        run_dir, "detections", "SolarSystem", "*")))
    for ss_dir in ss_dirs:
        if os.path.isdir(ss_dir):
            body = os.path.basename(ss_dir)
            n_files = len(os.listdir(ss_dir))
            if n_files > 0:
                report_lines.append(
                    f"SOLAR SYSTEM: {body} ({n_files} files)")

    return report_lines, attachment_files


# ---------------------------------------------------------------------------
#  Step 3 — Email report
# ---------------------------------------------------------------------------

def send_email_report(run_dir, report_lines, attachment_files):
    """Send HTML email summary with attachments."""
    creds = None
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE) as f:
                creds = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read secrets: {e}")

    msg = MIMEMultipart()
    msg['From'] = creds['email'] if creds else "pipeline@astro.caltech.edu"
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"OVRO-LWA Results: {os.path.basename(run_dir)}"

    body = "<h3>Pipeline Run Completed</h3>"
    body += f"<p>Run Directory: {run_dir}</p>"

    noise_csv = os.path.join(run_dir, "Wideband", "thermal_noise.csv")
    if os.path.exists(noise_csv):
        try:
            df = pd.read_csv(noise_csv)
            body += "<h4>Thermal Noise Profile</h4><ul>"
            body += f"<li>Median RMS: {df['rms'].median() * 1000:.1f} mJy</li>"
            body += (f"<li>Min RMS: {df['rms'].min() * 1000:.1f} mJy "
                     f"(@ {df.loc[df['rms'].idxmin()]['freq']:.0f} MHz)</li>")
            body += "</ul>"
            attachment_files.append(noise_csv)
        except Exception:
            pass

    body += "<h4>Detections</h4>"
    if report_lines:
        body += "<ul>" + "".join(
            f"<li>{l}</li>" for l in report_lines) + "</ul>"
    else:
        body += "<p>No significant detections found.</p>"

    msg.attach(MIMEText(body, 'html'))

    total_size = 0
    for f in list(set(attachment_files)):
        if os.path.exists(f):
            s = os.path.getsize(f)
            if total_size + s < 24 * 1024 * 1024:
                total_size += s
                with open(f, "rb") as att:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(att.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                f"attachment; filename={os.path.basename(f)}")
                msg.attach(part)

    try:
        if creds:
            s = smtplib.SMTP(creds['server'], creds['port'])
            s.starttls()
            s.login(creds['email'], creds['password'])
        else:
            s = smtplib.SMTP('localhost')
        s.send_message(msg)
        s.quit()
        logger.info(f"Email sent to {EMAIL_RECIPIENT}.")
    except Exception as e:
        logger.error(f"Email failed: {e}")


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def run_post_processing(run_dir, catalog_path=None):
    """Run the full Phase 3 post-processing pipeline.

    Args:
        run_dir: Top-level Lustre run directory containing per-subband folders.
        catalog_path: Source catalog CSV for transient search masking.
    """
    os.makedirs(OUTPUT_CAT_DIR, exist_ok=True)

    # 1. Flux check (across all subbands)
    logger.info("Running Flux Check …")
    try:
        from orca.transform.flux_check_cutout import run_flux_check
        run_flux_check(run_dir, logger=logger)
    except Exception as e:
        logger.error(f"Flux check failed: {e}")

    # 2. Wideband stacking + transient search
    run_wideband_stacking(run_dir, catalog_path)

    # 3. Wideband solar system photometry
    try:
        from orca.transform.solar_system_cutout import (
            process_wideband_solar_system,
        )
        process_wideband_solar_system(run_dir, logger=logger)
    except ImportError:
        logger.warning("Wideband solar system function not available.")
    except Exception as e:
        logger.error(f"Wideband solar system failed: {e}")
        traceback.print_exc()

    # 4. Gather & report
    report_lines, att_files = gather_detections(run_dir)
    send_email_report(run_dir, report_lines, att_files)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    parser = argparse.ArgumentParser(
        description="OVRO-LWA Phase 3 wideband science aggregation")
    parser.add_argument("--run_dir", required=True,
                        help="Lustre run directory with per-subband folders")
    parser.add_argument("--catalog",
                        help="Source catalog CSV for transient masking")
    args = parser.parse_args()

    run_post_processing(args.run_dir, catalog_path=args.catalog)
