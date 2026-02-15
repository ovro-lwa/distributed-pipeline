"""Ionospheric dewarping via VLSSr cross-match.

Ported from ExoPipe/extractor_pb_75.py into the orca package.

Computes pixel-level warp screens by cross-matching extracted LWA sources
against the VLSSr catalog and interpolating positional offsets over the
full image grid using scipy.interpolate.griddata.

Functions
---------
parse_vlssr_text
    Parse the OVRO-LWA VLSSr text catalog into a dict.
load_ref_catalog
    Load a reference catalog (FITS or VLSSr text format).
generate_warp_screens
    Build 2-D warp screens from LWA↔catalog cross-match.
apply_warp
    Apply pre-computed warp screens to an image array.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

import logging

logger = logging.getLogger(__name__)

# --- Cross-match configuration ---
MATCH_RADIUS = 5.0 * u.arcmin
DEFAULT_ALPHA = -0.7                  # Typical synchrotron spectral index
FLUX_RATIO_MIN = 0.80
FLUX_RATIO_MAX = 1.20
VLSS_MAX_SIZE_DEG = 75.0 / 3600.0
NVSS_MAX_SIZE_DEG = 45.0 / 3600.0
LWA_SIZE_TOLERANCE = 1.20


# ---------------------------------------------------------------------------
#  Catalog loading
# ---------------------------------------------------------------------------

def parse_vlssr_text(path):
    """Parse the OVRO-LWA VLSSr text catalog.

    The text format has columns: RAh RAm RAs DECd DECm DECs ... flux ...
    where flux is in Jy and positions are sexagesimal.

    Args:
        path: Path to the VLSSr text file.

    Returns:
        Dict with keys 'coords', 'flux', 'maj', 'ra', 'dec', 'type',
        or None on failure.
    """
    coords, fluxes, maj_axes = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        try:
            ra_str = f"{parts[0]}h{parts[1]}m{parts[2]}s"
            dec_str = f"{parts[3]}d{parts[4]}m{parts[5]}s"
            flux_val = float(parts[7].replace('<', ''))

            maj_deg = 0.0
            if len(parts) >= 11:
                maj_deg = float(parts[9].replace('<', '')) / 3600.0

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                coords.append(SkyCoord(ra_str, dec_str, frame='icrs'))

            fluxes.append(flux_val)
            maj_axes.append(maj_deg)
        except Exception:
            continue

    if not coords:
        logger.warning("Failed to parse any sources from VLSSr text file.")
        return None

    return {
        'coords': SkyCoord(coords),
        'flux': np.array(fluxes),
        'maj': np.array(maj_axes),
        'ra': np.array([c.ra.deg for c in coords]),
        'dec': np.array([c.dec.deg for c in coords]),
        'type': 'VLSS',
    }


def load_ref_catalog(path, name="VLSSr"):
    """Load a reference catalog (FITS table or VLSSr text).

    Args:
        path: Filesystem path to the catalog.
        name: Catalog name (used to identify format).

    Returns:
        Dict with keys 'coords', 'flux', 'maj', 'ra', 'dec', 'type',
        or None on failure.
    """
    if not os.path.exists(path):
        logger.error(f"Catalog file not found: {path}")
        return None

    logger.info(f"[{name}] Loading catalog {path}...")
    is_fits = path.lower().endswith(('.fits', '.fit'))
    cat_type = 'VLSS' if 'VLSS' in name.upper() else 'NVSS'

    if not is_fits and 'VLSS' in name.upper():
        return parse_vlssr_text(path)

    if is_fits:
        try:
            with fits.open(path) as hdul:
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                tab = Table(data)
        except Exception:
            return None

        cols = tab.colnames
        ra_col = next((c for c in cols if 'RA' in c.upper()), None)
        dec_col = next((c for c in cols if 'DEC' in c.upper()), None)
        flux_col = next(
            (c for c in cols
             if any(x in c.upper() for x in ['S1400', 'S74', 'PEAK', 'INT'])
             and 'ERR' not in c.upper()),
            None,
        )
        maj_col = next(
            (c for c in cols if 'MAJ' in c.upper() and 'ERR' not in c.upper()),
            None,
        )

        if all([ra_col, dec_col, flux_col]):
            mask = (~np.isnan(tab[ra_col]) & ~np.isnan(tab[dec_col])
                    & ~np.isnan(tab[flux_col]))
            tab = tab[mask]

            maj_vals = np.zeros(len(tab))
            if maj_col:
                maj_vals = np.array(tab[maj_col])
                if len(maj_vals) > 0 and np.median(maj_vals) > 1.0:
                    maj_vals /= 3600.0

            flux_vals = np.array(tab[flux_col])
            if np.nanmedian(flux_vals) > 50:
                flux_vals /= 1000.0

            return {
                'ra': np.array(tab[ra_col]),
                'dec': np.array(tab[dec_col]),
                'flux': flux_vals,
                'maj': maj_vals,
                'coords': SkyCoord(tab[ra_col], tab[dec_col], unit='deg'),
                'type': cat_type,
            }
    return None


# ---------------------------------------------------------------------------
#  Warp screen generation
# ---------------------------------------------------------------------------

def _plot_distortion_map(mag, sx, sy, nx, title, outname):
    """Save a diagnostic distortion-magnitude + quiver plot."""
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    im = plt.imshow(mag, origin='lower', cmap='inferno')
    plt.colorbar(im, label='Shift (arcmin)')
    plt.title(f"Distortion: {title}")

    plt.subplot(122)
    step = max(nx // 40, 1)

    sy_sub = sy[::step, ::step]
    sx_sub = sx[::step, ::step]

    h, w = sx_sub.shape
    x_idx = np.arange(0, w * step, step)[:w]
    y_idx = np.arange(0, h * step, step)[:h]
    X, Y = np.meshgrid(x_idx, y_idx)

    if np.ma.is_masked(sx_sub):
        sx_sub = sx_sub.filled(np.nan)
    if np.ma.is_masked(sy_sub):
        sy_sub = sy_sub.filled(np.nan)

    min_y = min(Y.shape[0], sx_sub.shape[0])
    min_x = min(X.shape[1], sx_sub.shape[1])

    plt.imshow(mag, origin='lower', cmap='gray', alpha=0.3)
    plt.quiver(
        X[:min_y, :min_x], Y[:min_y, :min_x],
        sx_sub[:min_y, :min_x], sy_sub[:min_y, :min_x],
        color='cyan', scale=None, scale_units='xy',
    )

    plt.savefig(outname, dpi=150)
    plt.close()


def generate_warp_screens(
    img_sources, cat_data, wcs_obj, shape,
    freq_mhz, cat_freq_mhz,
    lwa_bmaj_deg, clip_sigma,
    base_name="",
):
    """Build 2-D ionospheric warp screens from LWA↔catalog cross-match.

    Args:
        img_sources: DataFrame with columns 'ra', 'dec', 'flux_peak_I_app',
            optionally 'maj'.
        cat_data: Dict from :func:`load_ref_catalog`.
        wcs_obj: ``astropy.wcs.WCS`` (celestial) of the reference image.
        shape: (ny, nx) of the image.
        freq_mhz: Observing frequency (MHz).
        cat_freq_mhz: Reference catalog frequency (MHz), e.g. 74.
        lwa_bmaj_deg: Beam FWHM of the LWA image (degrees).
        clip_sigma: Sigma-clipping threshold (not currently applied —
            kept for API compatibility).
        base_name: Prefix for diagnostic file names.

    Returns:
        Tuple ``(screen_x, screen_y, anchors_diag, raw_coords)``
        where screens are 2-D arrays of pixel offsets, or
        ``(None, None, None, None)`` on failure.
    """
    diag_dir = "Dewarp_Diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    cat_limit = (VLSS_MAX_SIZE_DEG if cat_data['type'] == 'VLSS'
                 else NVSS_MAX_SIZE_DEG)

    # 1. Filter LWA: keep unresolved sources
    if 'maj' in img_sources.columns:
        mask_point_lwa = img_sources['maj'] < (lwa_bmaj_deg * LWA_SIZE_TOLERANCE)
    else:
        mask_point_lwa = np.ones(len(img_sources), dtype=bool)

    lwa_unresolved = img_sources[mask_point_lwa].copy()
    lwa_unresolved.sort_values(
        by='flux_peak_I_app', ascending=False, inplace=True)
    if lwa_unresolved.empty:
        return None, None, None, None

    # 2. Filter catalog
    if 'maj' in cat_data:
        mask_cat_point = cat_data['maj'] < cat_limit
    else:
        mask_cat_point = np.ones(len(cat_data['ra']), dtype=bool)

    cat_coords_filt = cat_data['coords'][mask_cat_point]
    cat_flux_filt = cat_data['flux'][mask_cat_point]
    cat_ra_filt = cat_data['ra'][mask_cat_point]
    cat_dec_filt = cat_data['dec'][mask_cat_point]

    # 3. Cross-match
    c_img = SkyCoord(
        lwa_unresolved['ra'].values * u.deg,
        lwa_unresolved['dec'].values * u.deg,
    )
    idx, d2d, _ = c_img.match_to_catalog_sky(cat_coords_filt)

    # 4. Flux consistency check with frequency-dependent tolerance
    predicted_flux = cat_flux_filt[idx] * (
        (freq_mhz / cat_freq_mhz) ** DEFAULT_ALPHA)
    lwa_flux = lwa_unresolved['flux_peak_I_app'].values
    flux_ratio = lwa_flux / predicted_flux

    log_freq_ratio = abs(np.log10(freq_mhz / cat_freq_mhz))
    tolerance_factor = max(1.0, 1.0 + 3.0 * log_freq_ratio)
    eff_flux_min = FLUX_RATIO_MIN / tolerance_factor
    eff_flux_max = FLUX_RATIO_MAX * tolerance_factor

    mask_pos = d2d < MATCH_RADIUS
    mask_flux = (flux_ratio > eff_flux_min) & (flux_ratio < eff_flux_max)
    mask_warp = mask_pos & mask_flux

    n_warp = int(np.sum(mask_warp))
    n_diag = int(np.sum(mask_pos))

    logger.info(
        f"[Dewarp] freq={freq_mhz:.1f} MHz, cat={cat_freq_mhz:.0f} MHz, "
        f"tolerance=[{eff_flux_min:.2f}, {eff_flux_max:.2f}], "
        f"anchors={n_warp}, diag_matches={n_diag}"
    )

    if n_warp < 6:
        logger.warning("Too few warp anchors (<6). Skipping dewarping.")
        return None, None, None, None

    anchors_warp = {
        'ra_lwa': lwa_unresolved['ra'].values[mask_warp],
        'dec_lwa': lwa_unresolved['dec'].values[mask_warp],
        'ra_cat': cat_ra_filt[idx[mask_warp]],
        'dec_cat': cat_dec_filt[idx[mask_warp]],
    }

    anchors_diag = {
        'ra_lwa': lwa_unresolved['ra'].values[mask_pos],
        'dec_lwa': lwa_unresolved['dec'].values[mask_pos],
        'flux_lwa': lwa_flux[mask_pos],
        'flux_cat': predicted_flux[mask_pos],
    }

    # 5. Build interpolated warp screens
    mx, my = wcs_obj.all_world2pix(
        anchors_warp['ra_lwa'], anchors_warp['dec_lwa'], 0)
    tx, ty = wcs_obj.all_world2pix(
        anchors_warp['ra_cat'], anchors_warp['dec_cat'], 0)

    ny, nx = shape
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]

    screen_x = griddata(
        (ty, tx), mx - tx, (grid_y, grid_x),
        method='linear', fill_value=0.0,
    )
    screen_y = griddata(
        (ty, tx), my - ty, (grid_y, grid_x),
        method='linear', fill_value=0.0,
    )

    # Fill NaN edges with nearest-neighbour
    mask_nan = np.isnan(screen_x)
    if np.any(mask_nan):
        interp_nn_x = NearestNDInterpolator(list(zip(ty, tx)), mx - tx)
        interp_nn_y = NearestNDInterpolator(list(zip(ty, tx)), my - ty)
        screen_x[mask_nan] = interp_nn_x(grid_y[mask_nan], grid_x[mask_nan])
        screen_y[mask_nan] = interp_nn_y(grid_y[mask_nan], grid_x[mask_nan])

    # 6. Diagnostic plot
    try:
        scales = wcs_obj.proj_plane_pixel_scales()
        scale_deg = np.mean([scales[0].value, scales[1].value])
        shift_mag = np.sqrt(screen_x ** 2 + screen_y ** 2) * scale_deg * 60.0
        plot_out = os.path.join(
            diag_dir, f"distortion_map_raw_{base_name}.png")
        _plot_distortion_map(shift_mag, screen_x, screen_y, nx, "Raw", plot_out)
    except Exception as e:
        logger.warning(f"Distortion plot failed: {e}")

    return screen_x, screen_y, anchors_diag, (mx, my, tx, ty)


def apply_warp(data, screen_x, screen_y):
    """Apply pre-computed warp screens to an image array.

    Args:
        data: 2-D numpy array of image pixels.
        screen_x: X-offset screen (same shape as *data*).
        screen_y: Y-offset screen (same shape as *data*).

    Returns:
        Dewarped 2-D array, or None on shape mismatch.
    """
    ny, nx = data.shape
    sy, sx_s = screen_x.shape
    if (ny != sy) or (nx != sx_s):
        logger.error(
            f"Warp shape mismatch: image {data.shape} vs screen {screen_x.shape}")
        return None
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]
    return map_coordinates(
        data,
        [grid_y + screen_y, grid_x + screen_x],
        order=1, mode='constant', cval=0.0,
    )
