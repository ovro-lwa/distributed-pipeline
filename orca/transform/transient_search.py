"""OVRO-LWA Transient Search Module.

Ported from ExoPipe/transient_search.py into the orca package for use
with the Celery pipeline.  All ``pipeline_config`` imports are replaced
by ``orca.resources.subband_config``.

Stokes V: Search each image directly (NO reference subtraction).
Stokes I: Subtract deep from each 10min snapshot.

``run_test()`` returns a list of detection dicts (with ``snr`` key) so
the caller can sort, filter, and apply a quality gate.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.stats import mad_std
from astropy.nddata import Cutout2D
from scipy.ndimage import label, zoom, binary_dilation, distance_transform_edt
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# --- OBSERVATORY ---
from orca.resources.subband_config import OVRO_LOC

# --- CONFIGURATION PROFILES ---

# 1. STOKES I (High Contamination Mode)
CONFIG_I = {
    'MIN_ELEVATION': 30.0,
    'A_TEAM': {
        'Cas A': {'coord': SkyCoord('23h23m24s', '+58d48m54s'), 'radius': 15.0 * u.deg},
        'Cyg A': {'coord': SkyCoord('19h59m28s', '+40d44m02s'), 'radius': 15.0 * u.deg},
        'Tau A': {'coord': SkyCoord('05h34m31s', '+22d00m52s'), 'radius': 10.0 * u.deg},
        'Vir A': {'coord': SkyCoord('12h30m49s', '+12d23m28s'), 'radius': 10.0 * u.deg}
    },
    'MASK_TIERS': [
        [100.0, 2.0],
        [30.0,  1.0],
    ],
    'CANDIDATE_SIGMA': 7.0,
    'PARTNER_SIGMA': 4.0,
    'RMS_BOX_SIZE': 32,
    'BILOBE_RADIUS': 30.0 * u.arcmin,
    'FLUX_RATIO_MIN': 0.2,
    'FLUX_RATIO_MAX': 5.0,
    'CATALOG_MATCH_RADIUS': 2.0 * u.arcmin
}

# 2. STOKES V (Low Contamination Mode)
CONFIG_V = {
    'MIN_ELEVATION': 30.0,
    'A_TEAM': {
        'Cas A': {'coord': SkyCoord('23h23m24s', '+58d48m54s'), 'radius': 5.0 * u.deg},
        'Cyg A': {'coord': SkyCoord('19h59m28s', '+40d44m02s'), 'radius': 5.0 * u.deg},
        'Tau A': {'coord': SkyCoord('05h34m31s', '+22d00m52s'), 'radius': 5.0 * u.deg},
        'Vir A': {'coord': SkyCoord('12h30m49s', '+12d23m28s'), 'radius': 5.0 * u.deg},
        'Hydra A': {'coord': SkyCoord('09h18m05.7s', '-12d05m44s'), 'radius': 3.0 * u.deg}
    },
    'MASK_TIERS': [
        [500.0, 2.0],
        [100.0, 1.0],
    ],
    'CANDIDATE_SIGMA': 5.0,
    'PARTNER_SIGMA': 3.0,
    'RMS_BOX_SIZE': 32,
    'BILOBE_RADIUS': 30.0 * u.arcmin,
    'FLUX_RATIO_MIN': 0.2,
    'FLUX_RATIO_MAX': 5.0,
    'CATALOG_MATCH_RADIUS': 2.0 * u.arcmin
}

CUTOUT_SIZE = 2.0 * u.deg
MAX_CUTOUTS = 5


def _truncated_jname(coord):
    """Generate truncated J-name: J0553+31 format."""
    ra_str = coord.ra.to_string(unit=u.hour, sep='', precision=0, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep='', precision=0, alwayssign=True, pad=True)
    ra_short = ra_str[:4]
    dec_short = dec_str[:3]
    return f"J{ra_short}{dec_short}"


def load_catalog(path):
    """Load a source catalog for cross-matching."""
    if not path or not os.path.exists(path):
        return None, None
    try:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=r'\s+')
        df.columns = [c.lower() for c in df.columns]
        ra_key, dec_key, name_key = 'ra_current', 'dec_current', 'common_name'
        if ra_key not in df.columns or dec_key not in df.columns:
            return None, None
        coords = SkyCoord(df[ra_key].values, df[dec_key].values, unit='deg')
        names = df[name_key].values
        print(f"[Catalog] Loaded {len(df)} sources.")
        return coords, names
    except Exception:
        return None, None


def get_midpoint_time(header):
    """Get observation midpoint from FITS header."""
    try:
        t_start = Time(header.get('DATE-OBS'), format='isot', scale='utc')
        duration_sec = float(header.get('DURATION', 0.0))
        t_mid = t_start + TimeDelta(duration_sec * 0.5, format='sec')
        return t_mid
    except Exception:
        return Time(header.get('DATE-OBS', Time.now().isot), format='isot', scale='utc')


def _target_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def create_static_masks(header_tgt, header_ref, wcs, shape, cfg):
    """Create horizon and A-team masks."""
    h, w = shape
    y, x = np.indices((h, w))
    ra, dec = wcs.all_pix2world(x, y, 0)
    coords = SkyCoord(ra, dec, unit='deg')

    min_el = cfg['MIN_ELEVATION']

    time_tgt = get_midpoint_time(header_tgt)
    altaz_tgt = coords.transform_to(AltAz(obstime=time_tgt, location=OVRO_LOC))
    mask_tgt = altaz_tgt.alt.deg < min_el

    mask_ref = np.zeros(shape, dtype=bool)
    if header_ref is not None:
        time_ref = get_midpoint_time(header_ref)
        altaz_ref = coords.transform_to(AltAz(obstime=time_ref, location=OVRO_LOC))
        mask_ref = altaz_ref.alt.deg < min_el

    horizon_mask = mask_tgt | mask_ref

    ateam_mask = np.zeros(shape, dtype=bool)
    for name, info in cfg['A_TEAM'].items():
        sep = coords.separation(info['coord'])
        ateam_mask |= (sep < info['radius'])

    return horizon_mask, ateam_mask, time_tgt


def create_bright_source_mask(data_ref, wcs, header_for_beam, cfg, search_rms):
    """Adaptive masking: mask where Flux_Ref > (Tier * Sigma_Search)."""
    try:
        bmaj = header_for_beam.get('BMAJ', 0.0)
        cdelt = np.abs(wcs.wcs.cdelt[0])
        beam_pix = bmaj / cdelt if bmaj > 0 else 5.0
    except Exception:
        beam_pix = 5.0

    final_mask = np.zeros_like(data_ref, dtype=bool)

    for tier in cfg['MASK_TIERS']:
        sigma_mult, radius_beams = tier
        flux_thresh = sigma_mult * search_rms
        tier_pixels = data_ref > flux_thresh

        if np.any(tier_pixels):
            radius_pix = int(beam_pix * radius_beams)
            struct = np.ones((radius_pix * 2 + 1, radius_pix * 2 + 1))
            dilated = binary_dilation(tier_pixels, structure=struct)
            final_mask |= dilated

    return final_mask


def calculate_local_rms_map(data, mask, cfg):
    """Compute a local RMS map using boxcar MAD."""
    box_size = cfg['RMS_BOX_SIZE']
    h, w = data.shape
    work_data = data.copy()
    work_data[mask] = np.nan
    ny, nx = int(np.ceil(h / box_size)), int(np.ceil(w / box_size))
    grid_rms = np.zeros((ny, nx))
    print(f"  Calculating Local RMS Map ({ny}x{nx} grid, Box={box_size})...")

    for i in range(ny):
        for j in range(nx):
            y1, y2 = i * box_size, min((i + 1) * box_size, h)
            x1, x2 = j * box_size, min((j + 1) * box_size, w)
            chunk = work_data[y1:y2, x1:x2]
            val = mad_std(chunk, ignore_nan=True)
            grid_rms[i, j] = val if not (np.isnan(val) or val == 0) else np.nan

    valid_mask = np.isfinite(grid_rms)
    if not np.any(valid_mask):
        grid_rms[:] = 1.0
    elif not np.all(valid_mask):
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        grid_rms = grid_rms[tuple(indices)]

    rms_map = zoom(grid_rms, (h / ny, w / nx), order=1)

    if rms_map.shape != (h, w):
        new_map = np.zeros((h, w))
        sh, sw = min(h, rms_map.shape[0]), min(w, rms_map.shape[1])
        new_map[:sh, :sw] = rms_map[:sh, :sw]
        rms_map = new_map

    cy, cx = h // 2, w // 2
    half_box = 256
    center_chunk = work_data[cy - half_box:cy + half_box, cx - half_box:cx + half_box]
    floor_rms = mad_std(center_chunk, ignore_nan=True)

    if np.isfinite(floor_rms) and floor_rms > 0:
        print(f"  Enforcing RMS Floor: {floor_rms:.4f} Jy")
        rms_map = np.maximum(rms_map, floor_rms)

    return rms_map


def extract_peaks_flex(data, mask, threshold_sigma, rms_map, wcs, obs_time):
    """Extract peaks above (or below) a sigma threshold."""
    if threshold_sigma > 0:
        threshold_map = threshold_sigma * rms_map
        work_data = data.copy()
        work_data[mask] = -np.inf
        blobs = work_data > threshold_map
    else:
        threshold_map = abs(threshold_sigma) * rms_map
        work_data = data.copy()
        work_data[mask] = np.inf
        blobs = work_data < -threshold_map

    labeled, num = label(blobs)
    results = []

    if num > 0:
        objects = scipy.ndimage.find_objects(labeled)
        for i, slice_obj in enumerate(objects):
            sub_data = work_data[slice_obj]
            if threshold_sigma > 0:
                my, mx = np.unravel_index(np.nanargmax(sub_data), sub_data.shape)
            else:
                my, mx = np.unravel_index(np.nanargmin(sub_data), sub_data.shape)

            gy, gx = slice_obj[0].start + my, slice_obj[1].start + mx
            peak_val = work_data[gy, gx]
            local_rms = rms_map[gy, gx]

            try:
                coord = wcs.pixel_to_world(gx, gy)
                if np.isnan(coord.ra.deg) or np.isnan(coord.dec.deg):
                    continue
                altaz = coord.transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
                el = altaz.alt.deg

                results.append({
                    'coord': coord,
                    'flux': peak_val,
                    'snr': peak_val / local_rms,
                    'el': el,
                    'pix_x': int(gx),
                    'pix_y': int(gy),
                })
            except Exception:
                continue
    return results


def filter_bilobed_artifacts(candidates, partners, cfg):
    """Remove candidates that have a matching opposite-sign partner."""
    clean = []
    artifacts = []
    if not partners:
        return candidates, []

    radius = cfg['BILOBE_RADIUS']
    fr_min = cfg['FLUX_RATIO_MIN']
    fr_max = cfg['FLUX_RATIO_MAX']

    partner_coords = SkyCoord([p['coord'] for p in partners])

    for c in candidates:
        idx, d2d, _ = c['coord'].match_to_catalog_sky(partner_coords)
        if d2d < radius:
            partner = partners[idx]
            flux_ratio = abs(partner['flux'] / c['flux'])
            if fr_min < flux_ratio < fr_max:
                artifacts.append(c)
                continue
        clean.append(c)
    return clean, artifacts


def _make_cutout_header(original_header, cutout_wcs, candidate, stokes, basename_tag):
    """Build a cutout FITS header with metadata."""
    h = original_header.copy()
    for key in list(h.keys()):
        if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
            del h[key]
    h['NAXIS'] = 2
    h.update(cutout_wcs.to_header())
    coord = candidate['coord']
    h['DET_RA'] = (coord.ra.deg, '[deg] Detection RA (ICRS)')
    h['DET_DEC'] = (coord.dec.deg, '[deg] Detection Dec (ICRS)')
    h['DET_FLUX'] = (candidate['flux'], '[Jy] Detection peak flux')
    h['DET_SNR'] = (candidate['snr'], 'Detection SNR')
    h['DET_EL'] = (candidate['el'], '[deg] Elevation at detection')
    h['DET_POL'] = (stokes, 'Stokes parameter of search')
    h['DET_SRC'] = (basename_tag, 'Source image identifier')
    h['CUTSIZE'] = (CUTOUT_SIZE.value, '[deg] Cutout angular size')
    return h


def _make_cutout_v(candidate, data_v, wcs_v, data_i, wcs_i, header_v, header_i, out_dir, basename_tag):
    """Stokes V cutout: I and V FITS + side-by-side diagnostic PNG."""
    coord = candidate['coord']
    try:
        cut_v = Cutout2D(data_v, coord, (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs_v, mode='trim')
        cut_i = Cutout2D(data_i, coord, (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs_i, mode='trim')
    except Exception:
        return None

    ra_str = coord.ra.to_string(unit=u.hour, sep='', precision=0, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep='', precision=0, alwayssign=True, pad=True)
    full_jname = f"J{ra_str}{dec_str}".replace(' ', '')
    short_jname = _truncated_jname(coord)

    jname_dir = os.path.join(out_dir, "transients", "V", short_jname)
    os.makedirs(jname_dir, exist_ok=True)

    head_v_cut = _make_cutout_header(header_v, cut_v.wcs, candidate, 'V', basename_tag)
    head_i_cut = _make_cutout_header(header_i if header_i is not None else header_v,
                                      cut_i.wcs, candidate, 'I_ref', basename_tag)
    fits_v_name = f"det_V_{basename_tag}_{full_jname}_V.fits"
    fits_i_name = f"det_V_{basename_tag}_{full_jname}_I.fits"
    fits.writeto(os.path.join(jname_dir, fits_v_name), cut_v.data, head_v_cut, overwrite=True)
    fits.writeto(os.path.join(jname_dir, fits_i_name), cut_i.data, head_i_cut, overwrite=True)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection=cut_i.wcs)
    vmin_i, vmax_i = np.nanpercentile(cut_i.data, [1, 99.5])
    ax1.imshow(cut_i.data, origin='lower', cmap='inferno', vmin=vmin_i, vmax=vmax_i)
    ax1.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax1.set_title("Stokes I (reference)")
    try:
        cx_i, cy_i = cut_i.wcs.world_to_pixel(coord)
        ix, iy = int(round(cx_i)), int(round(cy_i))
        if 0 <= iy < cut_i.data.shape[0] and 0 <= ix < cut_i.data.shape[1]:
            i_flux = cut_i.data[iy, ix]
        else:
            i_flux = np.nan
    except Exception:
        i_flux = np.nan
    i_flux_str = f"{i_flux * 1000:.1f} mJy" if np.isfinite(i_flux) else "N/A"
    ax1.text(0.05, 0.95, f"Flux: {i_flux_str}",
             transform=ax1.transAxes, color='white', fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

    ax2 = fig.add_subplot(122, projection=cut_v.wcs)
    vmin_v, vmax_v = np.nanpercentile(cut_v.data, [1, 99.5])
    ax2.imshow(cut_v.data, origin='lower', cmap='inferno', vmin=vmin_v, vmax=vmax_v)
    ax2.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax2.set_title("Stokes V (search)")
    v_flux_str = f"{candidate['flux'] * 1000:.1f} mJy" if np.isfinite(candidate['flux']) else "N/A"
    snr_str = f"{candidate['snr']:.1f}" if np.isfinite(candidate['snr']) else "N/A"
    ax2.text(0.05, 0.95, f"Flux: {v_flux_str}\nSNR: {snr_str}",
             transform=ax2.transAxes, color='white', fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

    png_name = f"det_V_{basename_tag}_{full_jname}.png"
    png_path = os.path.join(jname_dir, png_name)
    plt.tight_layout()
    plt.savefig(png_path, dpi=100)
    plt.close()
    return png_path


def _make_cutout_i(candidate, data_diff, wcs_diff, data_ref, wcs_ref, header_tgt, header_ref, out_dir, basename_tag):
    """Stokes I cutout: reference + diff FITS + side-by-side diagnostic PNG."""
    coord = candidate['coord']
    try:
        cut_diff = Cutout2D(data_diff, coord, (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs_diff, mode='trim')
        cut_ref = Cutout2D(data_ref, coord, (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs_ref, mode='trim')
    except Exception:
        return None

    ra_str = coord.ra.to_string(unit=u.hour, sep='', precision=0, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep='', precision=0, alwayssign=True, pad=True)
    full_jname = f"J{ra_str}{dec_str}".replace(' ', '')
    short_jname = _truncated_jname(coord)

    jname_dir = os.path.join(out_dir, "transients", "I", short_jname)
    os.makedirs(jname_dir, exist_ok=True)

    head_diff_cut = _make_cutout_header(header_tgt, cut_diff.wcs, candidate, 'I_diff', basename_tag)
    head_diff_cut['HISTORY'] = 'Difference image: 10min snapshot minus deep reference'
    head_ref_cut = _make_cutout_header(
        header_ref if header_ref is not None else header_tgt,
        cut_ref.wcs, candidate, 'I_ref', basename_tag)

    fits_diff_name = f"det_I_{basename_tag}_{full_jname}_diff.fits"
    fits_ref_name = f"det_I_{basename_tag}_{full_jname}_ref.fits"
    fits.writeto(os.path.join(jname_dir, fits_diff_name), cut_diff.data, head_diff_cut, overwrite=True)
    fits.writeto(os.path.join(jname_dir, fits_ref_name), cut_ref.data, head_ref_cut, overwrite=True)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection=cut_ref.wcs)
    vmin_r, vmax_r = np.nanpercentile(cut_ref.data, [1, 99.5])
    ax1.imshow(cut_ref.data, origin='lower', cmap='inferno', vmin=vmin_r, vmax=vmax_r)
    ax1.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax1.set_title("Stokes I (deep reference)")
    try:
        cx_r, cy_r = cut_ref.wcs.world_to_pixel(coord)
        ix, iy = int(round(cx_r)), int(round(cy_r))
        if 0 <= iy < cut_ref.data.shape[0] and 0 <= ix < cut_ref.data.shape[1]:
            ref_flux = cut_ref.data[iy, ix]
        else:
            ref_flux = np.nan
    except Exception:
        ref_flux = np.nan
    ref_str = f"{ref_flux * 1000:.1f} mJy" if np.isfinite(ref_flux) else "N/A"
    ax1.text(0.05, 0.95, f"Ref Flux: {ref_str}",
             transform=ax1.transAxes, color='white', fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

    ax2 = fig.add_subplot(122, projection=cut_diff.wcs)
    vmin_d, vmax_d = np.nanpercentile(cut_diff.data, [1, 99.5])
    ax2.imshow(cut_diff.data, origin='lower', cmap='inferno', vmin=vmin_d, vmax=vmax_d)
    ax2.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax2.set_title("Stokes I (10min − deep)")
    diff_str = f"{candidate['flux'] * 1000:.1f} mJy" if np.isfinite(candidate['flux']) else "N/A"
    snr_str = f"{candidate['snr']:.1f}" if np.isfinite(candidate['snr']) else "N/A"
    ax2.text(0.05, 0.95, f"Diff Flux: {diff_str}\nSNR: {snr_str}",
             transform=ax2.transAxes, color='white', fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

    png_name = f"det_I_{basename_tag}_{full_jname}.png"
    png_path = os.path.join(jname_dir, png_name)
    plt.tight_layout()
    plt.savefig(png_path, dpi=100)
    plt.close()
    return png_path


def run_test(ref_path, tgt_path, ref_i_path, catalog_path, output_dir=None, mode=None):
    """Main entry point for transient search.

    Args:
        ref_path:     Reference image for subtraction (1hr I deep). None for Stokes V.
        tgt_path:     Target image to search (10min snapshot or deep V).
        ref_i_path:   Reference Stokes I image for bright-source masking.
        catalog_path: Path to source catalog CSV.
        output_dir:   Directory for cutouts and debug FITS.
        mode:         Force 'V' or 'I' mode. If None, auto-detected from path.

    Returns:
        List of detection dicts sorted by descending |SNR|.
    """
    if mode is not None:
        is_v = mode.upper() == 'V'
    else:
        path_parts = os.path.normpath(tgt_path).split(os.sep)
        is_v = 'V' in path_parts

    if is_v:
        cfg = CONFIG_V
        mode_name = "STOKES V (Low Contamination)"
    else:
        cfg = CONFIG_I
        mode_name = "STOKES I (High Contamination)"

    print(f"--- Bi-Lobed Transient Search [{mode_name}] ---")
    print(f"    Min El: {cfg['MIN_ELEVATION']} deg | Threshold: {cfg['CANDIDATE_SIGMA']} sigma")

    if ref_path:
        print(f"Ref (1hr): {os.path.basename(ref_path)}")
    else:
        print(f"Ref (1hr): [None] (Direct V Search Mode)")

    print(f"Tgt (10min): {os.path.basename(tgt_path)}")

    basename_tag = _target_basename(tgt_path)

    with fits.open(tgt_path) as hdul_tgt:
        d_tgt = hdul_tgt[0].data.squeeze()
        h_tgt = hdul_tgt[0].header
        wcs_tgt = WCS(h_tgt).celestial

    d_ref_sub = None
    h_ref_file = None
    if ref_path and os.path.exists(ref_path):
        with fits.open(ref_path) as hdul_ref:
            d_ref_sub = hdul_ref[0].data.squeeze()
            h_ref_file = hdul_ref[0].header

        if d_ref_sub.shape != d_tgt.shape:
            print("ERROR: Grid mismatch (Ref vs Tgt).")
            return []

    header_for_mask = h_tgt

    if is_v:
        search_map = d_tgt
        if ref_i_path and os.path.exists(ref_i_path):
            print(f"  Using provided I image for Bright Source Mask: {os.path.basename(ref_i_path)}")
            with fits.open(ref_i_path) as hdul_ref_i:
                ref_data_for_mask = hdul_ref_i[0].data.squeeze()
                header_for_mask = hdul_ref_i[0].header
        else:
            print("  WARNING: No I image provided for masking. Using Stokes V image.")
            ref_data_for_mask = d_tgt
    else:
        if d_ref_sub is None:
            print("ERROR: Stokes I requires a Reference image for subtraction.")
            return []
        search_map = d_tgt - d_ref_sub
        ref_data_for_mask = d_ref_sub
        if h_ref_file is not None:
            header_for_mask = h_ref_file

    search_rms = mad_std(search_map, ignore_nan=True)
    print(f"  Global Search RMS (Robust): {search_rms:.5f} Jy")

    print("Generating Masks...")
    horizon_mask, ateam_mask, time_tgt_mid = create_static_masks(
        h_tgt, h_ref_file, wcs_tgt, d_tgt.shape, cfg
    )

    bright_mask = create_bright_source_mask(
        ref_data_for_mask, wcs_tgt, header_for_mask, cfg, search_rms
    )
    final_mask = horizon_mask | ateam_mask | bright_mask

    try:
        scales = wcs_tgt.proj_plane_pixel_scales()
        pix_area_deg2 = (scales[0].value * scales[1].value)
        visible_sky_pix = np.sum(~horizon_mask)
        visible_sky_deg2 = visible_sky_pix * pix_area_deg2
        ateam_pix = np.sum(ateam_mask & (~horizon_mask))
        bright_pix = np.sum(bright_mask & (~horizon_mask) & (~ateam_mask))
        total_masked_pix = np.sum(final_mask & (~horizon_mask))

        print("-" * 65)
        print(f"Masked Area Statistics (Union Mask > {cfg['MIN_ELEVATION']} deg):")
        print(f"  Total Available Area: {visible_sky_deg2:.2f} sq. deg")
        print(f"  A-Team Mask:          {ateam_pix / visible_sky_pix * 100:6.2f}%")
        print(f"  Bright Source Mask:   {bright_pix / visible_sky_pix * 100:6.2f}%")
        print(f"  Total Science Lost:   {total_masked_pix / visible_sky_pix * 100:6.2f}%")
        print("-" * 65)
    except Exception as e:
        print(f"Could not calc area stats: {e}")

    rms_map = calculate_local_rms_map(search_map, final_mask, cfg)

    cand_sig = cfg['CANDIDATE_SIGMA']
    part_sig = cfg['PARTNER_SIGMA']

    print(f"Detecting Candidates (> {cand_sig} sigma)...")
    pos_cands_high = extract_peaks_flex(search_map, final_mask, cand_sig, rms_map, wcs_tgt, time_tgt_mid)

    if is_v:
        neg_cands_high = extract_peaks_flex(search_map, final_mask, -cand_sig, rms_map, wcs_tgt, time_tgt_mid)
    else:
        neg_cands_high = []

    print(f"Detecting Partners (> {part_sig} sigma)...")
    pos_cands_low = extract_peaks_flex(search_map, final_mask, part_sig, rms_map, wcs_tgt, time_tgt_mid)

    if is_v:
        neg_cands_low = extract_peaks_flex(search_map, final_mask, -part_sig, rms_map, wcs_tgt, time_tgt_mid)
    else:
        neg_cands_low = []

    print(f"  Candidates: {len(pos_cands_high)} (+) / {len(neg_cands_high)} (-)")

    print("Filtering Artifacts...")
    real_pos, art_pos = filter_bilobed_artifacts(pos_cands_high, neg_cands_low, cfg)
    real_neg, art_neg = filter_bilobed_artifacts(neg_cands_high, pos_cands_low, cfg)

    all_real = real_pos + real_neg
    print(f"  Rejected {len(art_pos) + len(art_neg)} pairs. Remaining: {len(all_real)}")

    cat_coords, cat_names = load_catalog(catalog_path)

    detections = []
    for c in all_real:
        if is_v:
            match_info = "Stokes V source detected"
        else:
            match_info = "New Transient"

        if cat_coords is not None:
            idx, d2d, _ = c['coord'].match_to_catalog_sky(cat_coords)
            if d2d < cfg['CATALOG_MATCH_RADIUS']:
                match_info = f"Known: {cat_names[idx]}"

        detections.append({
            'coord': c['coord'],
            'flux': c['flux'],
            'snr': c['snr'],
            'el': c['el'],
            'match_info': match_info,
            'stokes': 'V' if is_v else 'I',
            'tgt_image': tgt_path,
            'cutout_path': None,
        })

    detections.sort(key=lambda d: abs(d['snr']), reverse=True)

    print(f"\n--- FINAL CANDIDATES ({len(detections)}) ---")
    print(f"{'Type':<30} | {'RA (deg)':>9} {'Dec (deg)':>9} | {'El':>5} | {'Flux':>8} | {'SNR':>5}")
    print("-" * 80)

    for det in detections:
        c = det['coord']
        print(f"{det['match_info']:<30} | {c.ra.deg:>9.3f} {c.dec.deg:>9.3f} | "
              f"{det['el']:>5.1f} | {det['flux']:>6.3f} Jy | {det['snr']:>5.1f}")

    if output_dir is None:
        work_dir_guess = os.path.dirname(os.path.dirname(os.path.dirname(tgt_path)))
        if not work_dir_guess:
            work_dir_guess = '.'
        output_dir = os.path.join(work_dir_guess, "detections")

    os.makedirs(output_dir, exist_ok=True)
    cutout_dir = output_dir
    debug_dir = output_dir
    cutout_count = 0

    if is_v:
        data_i_cutout = ref_data_for_mask
        wcs_i_cutout = WCS(header_for_mask).celestial
        header_i_cutout = header_for_mask
    else:
        header_i_cutout = None

    for det in detections[:MAX_CUTOUTS]:
        try:
            if is_v:
                path = _make_cutout_v(
                    det, d_tgt, wcs_tgt, data_i_cutout, wcs_i_cutout,
                    h_tgt, header_i_cutout, cutout_dir, basename_tag
                )
            else:
                path = _make_cutout_i(
                    det, search_map, wcs_tgt, d_ref_sub, wcs_tgt,
                    h_tgt, h_ref_file, cutout_dir, basename_tag
                )
            det['cutout_path'] = path
            cutout_count += 1
        except Exception as e:
            print(f"  Cutout failed for candidate at ({det['coord'].ra.deg:.3f}, "
                  f"{det['coord'].dec.deg:.3f}): {e}")

    if cutout_count > 0:
        print(f"  Created {cutout_count} cutouts in {cutout_dir}")

    if len(detections) > MAX_CUTOUTS:
        print(f"  WARNING: {len(detections)} candidates exceed cutout limit of {MAX_CUTOUTS}.")

    suffix = 'V' if is_v else 'I'
    debug_subdir = os.path.join(debug_dir, "transients", "debug")
    os.makedirs(debug_subdir, exist_ok=True)

    fits.writeto(
        os.path.join(debug_subdir, f"debug_rms_{suffix}_{basename_tag}.fits"),
        rms_map, h_tgt, overwrite=True
    )
    fits.writeto(
        os.path.join(debug_subdir, f"debug_mask_{suffix}_{basename_tag}.fits"),
        final_mask.astype(int), h_tgt, overwrite=True
    )

    if not is_v:
        fits.writeto(
            os.path.join(debug_subdir, f"debug_diff_{suffix}_{basename_tag}.fits"),
            search_map, h_tgt, overwrite=True
        )

    return detections
