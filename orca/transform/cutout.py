"""Target photometry with ionospheric-aware search and confusing source masking.

Ported from ExoPipe/cutout.py into the orca package for use with the
Celery pipeline.  All ``pipeline_config`` imports are replaced by
``orca.resources.subband_config``.
"""

import os
import glob
import re
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std

# --- CASA TASKS ---
try:
    from casatasks import imfit, imstat
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

# --- OBSERVATORY ---
from orca.resources.subband_config import OVRO_LOC

warnings.simplefilter('ignore', category=FITSFixedWarning)

# --- CONFIG ---
DETECTION_SIGMA = 5.0
MIN_ELEVATION = 30.0  # degrees — skip targets below this elevation
BEAM_SEARCH_MULTIPLIER = 1.0
IONOSPHERE_PAD_ARCSEC_85MHZ = 120.0  # 2 arcmin at 85 MHz (top of band)
IONOSPHERE_REF_FREQ_MHZ = 85.0       # Reference frequency for scaling
DEFAULT_BEAM_ARCSEC = 300.0
CUTOUT_SIZE = 2.0 * u.deg
FIT_SIGMA_THRESHOLD = 3.0    # Only run imfit if peak exceeds this many sigma


def get_timestamp_from_header(header):
    try:
        t = Time(header['DATE-OBS'], format='isot', scale='utc')
        return t.datetime.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return "UnknownDate"


def get_beam_size_arcsec(header):
    if 'BMAJ' in header:
        return header['BMAJ'] * 3600.0
    return DEFAULT_BEAM_ARCSEC


def robust_measure_flux(data, cutout_wcs, sky_coord, beam_arcsec, freq_mhz,
                        fits_path=None, confusing_sources=None):
    """Measure flux at source position and find peak within beam + ionospheric padding.

    Args:
        data:        2D numpy array (cutout pixel data, already in memory)
        cutout_wcs:  astropy WCS (celestial, 2D) from Cutout2D
        sky_coord:   SkyCoord of the target
        beam_arcsec: Beam FWHM in arcseconds
        freq_mhz:    Observing frequency in MHz (for ionospheric scaling)
        fits_path:   Optional path to cutout FITS for CASA imfit
        confusing_sources: Optional list of SkyCoord for known confusing sources
                         to mask within one beam FWHM before measurement.

    Returns dict with keys:
        src_flux, peak_flux, peak_coord, offset_arcmin, rms,
        fit_flux, fit_err, method, n_confusers_masked
    """
    result = {
        'src_flux': np.nan, 'peak_flux': np.nan, 'peak_coord': None,
        'offset_arcmin': np.nan, 'rms': np.nan,
        'fit_flux': None, 'fit_err': None, 'method': 'Failed',
        'n_confusers_masked': 0
    }
    try:
        cx, cy = cutout_wcs.world_to_pixel(sky_coord)
        cx, cy = float(cx), float(cy)
        ny, nx = data.shape

        # Compute true angular pixel scale LOCAL to the target position
        sky_center = cutout_wcs.pixel_to_world(cx, cy)
        sky_dx = cutout_wcs.pixel_to_world(cx + 1, cy)
        sky_dy = cutout_wcs.pixel_to_world(cx, cy + 1)
        scale_x = sky_center.separation(sky_dx).arcsec
        scale_y = sky_center.separation(sky_dy).arcsec
        cdelt = np.sqrt(scale_x * scale_y)  # geometric mean, arcsec/pixel

        # Flux at exact source position (nearest pixel)
        ix_src, iy_src = int(round(cx)), int(round(cy))
        if 0 <= ix_src < nx and 0 <= iy_src < ny:
            result['src_flux'] = float(data[iy_src, ix_src])

        # Search region: beam + ionospheric padding
        iono_pad = IONOSPHERE_PAD_ARCSEC_85MHZ * (IONOSPHERE_REF_FREQ_MHZ / freq_mhz)
        search_arcsec = beam_arcsec * BEAM_SEARCH_MULTIPLIER + iono_pad
        radius_pix = int(np.ceil(search_arcsec / cdelt))

        x1 = max(0, ix_src - radius_pix)
        x2 = min(nx, ix_src + radius_pix + 1)
        y1 = max(0, iy_src - radius_pix)
        y2 = min(ny, iy_src + radius_pix + 1)

        print(f"    [flux] cdelt={cdelt:.1f}\"/pix (scale_x={scale_x:.1f}, scale_y={scale_y:.1f}) "
              f"beam={beam_arcsec:.0f}\" iono={iono_pad:.0f}\" "
              f"search={search_arcsec:.0f}\" radius={radius_pix}pix "
              f"src=({ix_src},{iy_src}) box=[{x1}:{x2},{y1}:{y2}] img={nx}x{ny}",
              flush=True)

        if x1 >= x2 or y1 >= y2:
            result['method'] = 'OutOfBounds'
            return result

        stamp = data[y1:y2, x1:x2]

        # Apply circular mask: only consider pixels within radius_pix of source
        sy, sx = np.ogrid[0:stamp.shape[0], 0:stamp.shape[1]]
        cx_stamp = ix_src - x1
        cy_stamp = iy_src - y1
        dist_sq = (sx - cx_stamp)**2 + (sy - cy_stamp)**2
        circular_mask = dist_sq <= radius_pix**2

        masked_stamp = np.where(circular_mask, stamp, np.nan)

        # Mask known confusing sources within one beam FWHM
        n_masked = 0
        if confusing_sources:
            beam_radius_pix = beam_arcsec / cdelt  # 1 FWHM in pixels
            for conf_coord in confusing_sources:
                try:
                    conf_cx, conf_cy = cutout_wcs.world_to_pixel(conf_coord)
                    conf_cx_stamp = float(conf_cx) - x1
                    conf_cy_stamp = float(conf_cy) - y1
                    conf_dist_sq = (sx - conf_cx_stamp)**2 + (sy - conf_cy_stamp)**2
                    conf_mask = conf_dist_sq <= beam_radius_pix**2
                    n_before = np.count_nonzero(np.isfinite(masked_stamp))
                    masked_stamp = np.where(conf_mask, np.nan, masked_stamp)
                    n_after = np.count_nonzero(np.isfinite(masked_stamp))
                    if n_before > n_after:
                        n_masked += 1
                        print(f"    [flux] Masked confuser at ({conf_coord.ra.deg:.3f}, "
                              f"{conf_coord.dec.deg:.3f}): {n_before - n_after} pixels")
                except Exception:
                    pass
        result['n_confusers_masked'] = n_masked

        rms = float(mad_std(masked_stamp, ignore_nan=True))
        result['rms'] = rms

        # Find peak pixel in circular search region
        peak_idx = np.nanargmax(masked_stamp)
        py_stamp, px_stamp = np.unravel_index(peak_idx, stamp.shape)
        px_full = x1 + px_stamp
        py_full = y1 + py_stamp
        result['peak_flux'] = float(data[py_full, px_full])

        peak_sky = cutout_wcs.pixel_to_world(float(px_full), float(py_full))
        result['peak_coord'] = peak_sky
        result['offset_arcmin'] = float(sky_coord.separation(peak_sky).arcmin)

        print(f"    [flux] peak at pix ({px_full},{py_full}) = {result['peak_flux']*1000:.1f} mJy, "
              f"offset={result['offset_arcmin']:.1f}'")

        result['method'] = 'Peak'

        # If significant detection and CASA available, try Gaussian fit on saved FITS
        if fits_path and rms > 0 and np.isfinite(rms):
            snr = result['peak_flux'] / rms
            if snr >= FIT_SIGMA_THRESHOLD and CASA_AVAILABLE:
                box_str = f"{x1},{y1},{x2-1},{y2-1}"
                try:
                    fit = imfit(imagename=fits_path, box=box_str)
                    if (fit and fit.get('converged', False)
                            and 'results' in fit
                            and 'component0' in fit['results']):
                        result['fit_flux'] = fit['results']['component0']['flux']['value'][0]
                        result['fit_err'] = fit['results']['component0']['flux']['error'][0]
                        result['method'] = 'Gaussian'
                except Exception:
                    pass

        return result

    except Exception as e:
        print(f"    [flux] Exception: {e}")
        return result


def _find_best_image(pol_dir, stokes_pattern, fallback_pol_dir=None):
    """Find best available image: dewarped pbcorr → pbcorr → raw image."""
    for search_dir, suffix in [(pol_dir, ""), (fallback_pol_dir, "(fallback)")]:
        if search_dir is None or (suffix and search_dir == pol_dir):
            continue
        # 1. dewarped pbcorr
        hits = glob.glob(os.path.join(search_dir, f"*{stokes_pattern}*pbcorr_dewarped*.fits"))
        if hits:
            return sorted(hits)[0], f"dewarped_pbcorr{suffix}"
        # 2. pbcorr (non-dewarped)
        hits = [f for f in glob.glob(os.path.join(search_dir, f"*{stokes_pattern}*pbcorr*.fits"))
                if "dewarped" not in f]
        if hits:
            return sorted(hits)[0], f"pbcorr{suffix}"
        # 3. raw image (no pbcorr)
        hits = [f for f in glob.glob(os.path.join(search_dir, f"*{stokes_pattern}*image*.fits"))
                if "pbcorr" not in f and "dewarped" not in f]
        if hits:
            return sorted(hits)[0], f"raw{suffix}"
    return None, "none"


def find_all_images(run_dir, fallback_dir=None):
    """Find all image sets for target cutouts.

    Returns dict with:
        'freq': float,
        'deep_i': path, 'deep_v': path,
        'snapshots_10min': list of (i_path, v_path, interval_tag) tuples
    Or empty dict if deep pair not found.
    """
    fb_i_deep = os.path.join(fallback_dir, "I", "deep") if fallback_dir else None
    fb_v_deep = os.path.join(fallback_dir, "V", "deep") if fallback_dir else None

    i_deep, i_type = _find_best_image(
        os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0.75", fb_i_deep)
    if i_deep is None:
        i_deep, i_type = _find_best_image(
            os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0", fb_i_deep)

    v_deep, v_type = _find_best_image(
        os.path.join(run_dir, "V", "deep"), "V-Taper-Deep", fb_v_deep)

    if i_deep is None or v_deep is None:
        return {}

    freq = 0.0
    fname = os.path.basename(i_deep)
    if "MHz" in fname:
        try:
            freq = float(fname.split('MHz')[0].split('-')[0].split('_')[-1])
        except (ValueError, IndexError):
            pass

    print(f"  [Cutout] I deep ({i_type}): {os.path.basename(i_deep)}")
    print(f"  [Cutout] V deep ({v_type}): {os.path.basename(v_deep)}")

    result = {
        'freq': freq,
        'deep_i': i_deep,
        'deep_v': v_deep,
        'snapshots_10min': []
    }

    for search_dir in [run_dir, fallback_dir]:
        if search_dir is None:
            continue

        i_10min_dir = os.path.join(search_dir, "I", "10min")
        v_10min_dir = os.path.join(search_dir, "V", "10min")

        if not os.path.isdir(i_10min_dir):
            continue

        i_snaps = glob.glob(os.path.join(i_10min_dir, "*Taper*10min*pbcorr_dewarped*.fits"))
        if not i_snaps:
            i_snaps = [f for f in glob.glob(os.path.join(i_10min_dir, "*Taper*10min*pbcorr*.fits"))
                       if "dewarped" not in f]
        if not i_snaps:
            i_snaps = [f for f in glob.glob(os.path.join(i_10min_dir, "*Taper*10min*image*.fits"))
                       if "pbcorr" not in f and "dewarped" not in f]
        if not i_snaps:
            continue

        for i_snap in sorted(i_snaps):
            bn = os.path.basename(i_snap)
            m = re.search(r'(t\d{4})', bn)
            int_tag = m.group(1) if m else None
            if int_tag is None:
                continue

            v_snaps = glob.glob(os.path.join(v_10min_dir, f"*Taper*10min*{int_tag}*pbcorr_dewarped*.fits"))
            if not v_snaps:
                v_snaps = [f for f in glob.glob(os.path.join(v_10min_dir, f"*Taper*10min*{int_tag}*pbcorr*.fits"))
                           if "dewarped" not in f]
            if not v_snaps:
                v_snaps = [f for f in glob.glob(os.path.join(v_10min_dir, f"*Taper*10min*{int_tag}*image*.fits"))
                           if "pbcorr" not in f and "dewarped" not in f]
            if v_snaps:
                result['snapshots_10min'].append((i_snap, v_snaps[0], int_tag))

        if result['snapshots_10min']:
            break

    if result['snapshots_10min']:
        print(f"  [Cutout] Found {len(result['snapshots_10min'])} 10min snapshot pairs")

    return result


def find_image_pairs(run_dir, fallback_dir=None):
    """Legacy wrapper: returns list of (freq, i_path, v_path) for deep images only."""
    info = find_all_images(run_dir, fallback_dir)
    if not info:
        return []
    return [(info['freq'], info['deep_i'], info['deep_v'])]


def process_target(run_dir, target_name, coord, sample_name, base_out_dir, detections_dir,
                   fallback_dir=None, detection_stokes='IV', confusing_sources=None):
    """Extract cutouts for a target from deep and 10min images.

    Deep Stokes I: uses confusing source masking if sources specified.
    10min Stokes I: uses (10min - deep) difference image — no masking needed.
    Stokes V: measured directly for both deep and 10min.
    """
    img_info = find_all_images(run_dir, fallback_dir=fallback_dir)
    if not img_info:
        return

    freq = img_info['freq']
    safe_target = target_name.replace(" ", "_")
    out_dir = os.path.join(base_out_dir, sample_name, safe_target)
    os.makedirs(out_dir, exist_ok=True)
    det_target_dir = os.path.join(detections_dir, sample_name, safe_target)
    os.makedirs(detections_dir, exist_ok=True)

    summary_data = []

    def _process_one(i_path, v_path, category, interval_tag=None,
                     i_diff_data=None, i_diff_wcs=None, i_diff_header=None):
        nonlocal summary_data
        try:
            with fits.open(i_path) as h_i:
                data_i_raw = h_i[0].data.squeeze()
                head_i = h_i[0].header
                wcs_i = WCS(head_i).celestial
                ts_str = get_timestamp_from_header(head_i)
                beam_arcsec = get_beam_size_arcsec(head_i)

                if freq == 0.0:
                    nonlocal_freq = head_i.get('CRVAL3', 0) / 1e6
                else:
                    nonlocal_freq = freq

            # Horizon check
            try:
                obs_time = Time(head_i['DATE-OBS'], format='isot', scale='utc')
                duration = float(head_i.get('DURATION', 0.0))
                mid_time = obs_time + duration * 0.5 * u.s
                altaz = coord.transform_to(AltAz(obstime=mid_time, location=OVRO_LOC))
                el = altaz.alt.deg
                if el < MIN_ELEVATION:
                    return
            except Exception:
                pass

            with fits.open(v_path) as h_v:
                data_v = h_v[0].data.squeeze()
                head_v = h_v[0].header

            if i_diff_data is not None:
                data_i_measure = i_diff_data
                wcs_i_measure = i_diff_wcs
                i_category_label = f"{category}_diff"
            else:
                data_i_measure = data_i_raw
                wcs_i_measure = wcs_i
                i_category_label = category

            cut_i = Cutout2D(data_i_measure, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                             wcs=wcs_i_measure, mode='trim')
            cut_v = Cutout2D(data_v, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                             wcs=WCS(head_v).celestial, mode='trim')

            tag = f"_{interval_tag}" if interval_tag else ""
            fname_i = f"{safe_target}_{nonlocal_freq:.0f}MHz_{category}{tag}_{ts_str}_I.fits"
            fname_v = f"{safe_target}_{nonlocal_freq:.0f}MHz_{category}{tag}_{ts_str}_V.fits"

            def make_cutout_header(original_header, cutout_wcs):
                h = original_header.copy()
                for key in list(h.keys()):
                    if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                        'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                        del h[key]
                h['NAXIS'] = 2
                h.update(cutout_wcs.to_header())
                return h

            head_cutout_i = make_cutout_header(
                i_diff_header if i_diff_header is not None else head_i, cut_i.wcs)
            head_cutout_v = make_cutout_header(head_v, cut_v.wcs)

            if i_diff_data is not None:
                head_cutout_i['HISTORY'] = 'Difference image: 10min snapshot minus deep reference'

            fits.writeto(os.path.join(out_dir, fname_i), cut_i.data, head_cutout_i, overwrite=True)
            fits.writeto(os.path.join(out_dir, fname_v), cut_v.data, head_cutout_v, overwrite=True)

            i_fits_path = os.path.join(out_dir, fname_i)
            v_fits_path = os.path.join(out_dir, fname_v)

            i_confusers = confusing_sources if (i_diff_data is None) else None
            i_res = robust_measure_flux(cut_i.data, cut_i.wcs, coord, beam_arcsec,
                                        nonlocal_freq, fits_path=i_fits_path,
                                        confusing_sources=i_confusers)
            v_res = robust_measure_flux(cut_v.data, cut_v.wcs, coord, beam_arcsec,
                                        nonlocal_freq, fits_path=v_fits_path)

            # Diagnostic Plot
            fig = plt.figure(figsize=(12, 6))

            for panel_idx, (ax_proj, cut, res, stokes) in enumerate([
                (cut_i.wcs, cut_i, i_res, 'I'),
                (cut_v.wcs, cut_v, v_res, 'V'),
            ]):
                ax = fig.add_subplot(1, 2, panel_idx + 1, projection=ax_proj)
                vmin, vmax = np.nanpercentile(cut.data, [1, 99.5])
                ax.imshow(cut.data, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

                ax.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)

                if res['peak_coord'] is not None:
                    ax.plot_coord(res['peak_coord'], 'D', color='yellow',
                                  ms=10, markeredgewidth=1.5, fillstyle='none')

                if stokes == 'I' and i_diff_data is None and confusing_sources:
                    for conf_c in confusing_sources:
                        ax.plot_coord(conf_c, 'o', color='red',
                                      ms=12, markeredgewidth=2, fillstyle='none')

                i_label = "I (diff)" if i_diff_data is not None else "I"
                title_stokes = i_label if stokes == 'I' else 'V'
                ax.set_title(f"Stokes {title_stokes} ({nonlocal_freq:.0f} MHz) {category}{tag}")

                lines = []
                lines.append(f"Src:  {res['src_flux']*1000:.1f} mJy")
                lines.append(f"Peak: {res['peak_flux']*1000:.1f} mJy")
                lines.append(f"RMS:  {res['rms']*1000:.1f} mJy")
                if res['fit_flux'] is not None:
                    lines.append(f"Fit:  {res['fit_flux']*1000:.1f} ± {res['fit_err']*1000:.1f} mJy")
                lines.append(f"Offset: {res['offset_arcmin']:.1f}'")
                if res.get('n_confusers_masked', 0) > 0:
                    lines.append(f"Masked: {res['n_confusers_masked']} source(s)")
                lines.append(f"({res['method']})")

                txt = "\n".join(lines)
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, color='white',
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(facecolor='black', alpha=0.6))

            png_name = f"{safe_target}_{nonlocal_freq:.0f}MHz_{category}{tag}_{ts_str}_Diagnostic.png"
            png_path = os.path.join(out_dir, png_name)
            plt.tight_layout()
            plt.savefig(png_path, dpi=100)
            plt.close()

            # Detection logic
            is_detection = False
            i_peak_snr = i_res['peak_flux'] / i_res['rms'] if i_res['rms'] > 0 else 0
            v_peak_snr = v_res['peak_flux'] / v_res['rms'] if v_res['rms'] > 0 else 0

            if detection_stokes == 'I':
                is_detection = i_peak_snr > DETECTION_SIGMA
            elif detection_stokes == 'V':
                is_detection = abs(v_peak_snr) > DETECTION_SIGMA
            else:
                is_detection = (i_peak_snr > DETECTION_SIGMA) or (abs(v_peak_snr) > DETECTION_SIGMA)

            if is_detection:
                os.makedirs(det_target_dir, exist_ok=True)
                shutil.copy(png_path, os.path.join(det_target_dir, png_name))

            summary_data.append({
                'Target': target_name, 'Freq_MHz': nonlocal_freq, 'Date': ts_str,
                'Category': category,
                'Interval': interval_tag if interval_tag else '',
                'I_Type': i_category_label,
                'I_Src_Jy': i_res['src_flux'], 'I_Peak_Jy': i_res['peak_flux'],
                'I_RMS_Jy': i_res['rms'], 'I_Offset_arcmin': i_res['offset_arcmin'],
                'I_Fit_Jy': i_res['fit_flux'], 'I_FitErr_Jy': i_res['fit_err'],
                'I_Method': i_res['method'],
                'I_Confusers_Masked': i_res.get('n_confusers_masked', 0),
                'V_Src_Jy': v_res['src_flux'], 'V_Peak_Jy': v_res['peak_flux'],
                'V_RMS_Jy': v_res['rms'], 'V_Offset_arcmin': v_res['offset_arcmin'],
                'V_Fit_Jy': v_res['fit_flux'], 'V_FitErr_Jy': v_res['fit_err'],
                'V_Method': v_res['method'],
                'Detection_Stokes': detection_stokes,
                'Detection': is_detection
            })

        except Exception as e:
            plt.close('all')
            import traceback
            print(f"  [Cutout ERROR] {target_name} {category}{tag if interval_tag else ''}: {e}")
            traceback.print_exc()

    # ===== DEEP images (1hr) =====
    _process_one(img_info['deep_i'], img_info['deep_v'], 'deep')

    # ===== 10min snapshots (differenced I) =====
    if img_info['snapshots_10min']:
        with fits.open(img_info['deep_i']) as h_ref:
            deep_i_data = h_ref[0].data.squeeze()
            deep_i_header = h_ref[0].header
            deep_i_wcs = WCS(deep_i_header).celestial

        for i_snap_path, v_snap_path, int_tag in img_info['snapshots_10min']:
            try:
                with fits.open(i_snap_path) as h_snap:
                    snap_i_data = h_snap[0].data.squeeze()
                    snap_i_header = h_snap[0].header
                    snap_i_wcs = WCS(snap_i_header).celestial

                if snap_i_data.shape == deep_i_data.shape:
                    diff_data = snap_i_data - deep_i_data
                    _process_one(i_snap_path, v_snap_path, '10min',
                                 interval_tag=int_tag,
                                 i_diff_data=diff_data,
                                 i_diff_wcs=snap_i_wcs,
                                 i_diff_header=snap_i_header)
                else:
                    print(f"  [Cutout WARN] Shape mismatch for {int_tag}: "
                          f"snap={snap_i_data.shape} vs deep={deep_i_data.shape}")
                    _process_one(i_snap_path, v_snap_path, '10min',
                                 interval_tag=int_tag)
            except Exception as e:
                print(f"  [Cutout ERROR] 10min {int_tag}: {e}")

    # Save Summary CSV
    if summary_data:
        csv_name = f"{safe_target}_photometry.csv"
        csv_path = os.path.join(out_dir, csv_name)
        pd.DataFrame(summary_data).to_csv(csv_path, index=False)
        if any(d['Detection'] for d in summary_data):
            os.makedirs(det_target_dir, exist_ok=True)
            shutil.copy(csv_path, os.path.join(det_target_dir, csv_name))


def load_targets(filepath):
    """Load targets from CSV. Returns list of (name, coord, detection_stokes, confusers)."""
    try:
        df = pd.read_csv(filepath)
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=r'\s+')
    except Exception:
        df = pd.read_csv(filepath, sep=r'\s+')

    cols = {c.lower(): c for c in df.columns}
    name_col = next((k for k in ['common_name', 'name', 'source', 'id'] if k in cols), None)
    ra_col = next((k for k in ['ra_current', 'ra_deg', 'ra'] if k in cols), None)
    dec_col = next((k for k in ['dec_current', 'dec_deg', 'dec'] if k in cols), None)
    det_col = next((k for k in ['detection_stokes', 'det_stokes', 'stokes'] if k in cols), None)
    conf_col = next((k for k in ['confusing_sources', 'confusers', 'confuser_coords'] if k in cols), None)

    if not (ra_col and dec_col):
        return []
    targets = []
    for _, row in df.iterrows():
        n = str(row[cols[name_col]]) if name_col else f"Target_{_}"
        try:
            r, d = row[cols[ra_col]], row[cols[dec_col]]
            if isinstance(r, (float, int)):
                c = SkyCoord(r, d, unit='deg')
            else:
                c = SkyCoord(r, d, unit=(u.hourangle, u.deg))
            det_stokes = 'IV'
            if det_col:
                val = str(row[cols[det_col]]).strip().upper()
                if val in ('I', 'V', 'IV', 'VI', 'BOTH'):
                    det_stokes = 'IV' if val in ('IV', 'VI', 'BOTH') else val
            confusing_sources = []
            if conf_col:
                conf_str = str(row[cols[conf_col]]).strip()
                if conf_str and conf_str.lower() not in ('nan', '', 'none'):
                    for pair in conf_str.split(';'):
                        pair = pair.strip()
                        if ',' in pair:
                            try:
                                cra, cdec = pair.split(',')
                                confusing_sources.append(SkyCoord(float(cra), float(cdec), unit='deg'))
                            except (ValueError, TypeError):
                                pass
            targets.append((n, c, det_stokes, confusing_sources))
        except Exception:
            pass
    return targets
