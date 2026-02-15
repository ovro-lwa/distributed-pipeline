#!/usr/bin/env python3
"""
extractor_pb_75.py — OVRO-LWA beam model & warp utilities.

Copied from ExoPipe into orca so pb_correction.py can import locally
without depending on an external PYTHONPATH entry.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, NearestNDInterpolator, RegularGridInterpolator
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.table import Table
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

# --- HARDCODED CONFIGURATION ---
OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg, height=1222*u.m)
BEAM_PATH = "/lustre/gh/calibration/pipeline/reference/beams/OVRO-LWA_MROsoil_updatedheight.h5"

MATCH_RADIUS = 5.0 * u.arcmin
DEFAULT_ALPHA = -0.7
FLUX_RATIO_MIN = 0.80
FLUX_RATIO_MAX = 1.20
VLSS_MAX_SIZE_DEG = 75.0 / 3600.0
NVSS_MAX_SIZE_DEG = 45.0 / 3600.0
LWA_SIZE_TOLERANCE = 1.20
WARP_ANCHOR_THRESH_PIX = 10.0

# --- COEFFICIENTS ---
CALIB_DATA = {
    '3C48':  {'coords': SkyCoord('01h37m41.3s', '+33d09m35s'), 'scale': 'SH12', 'coeffs': [64.768, -0.387, -0.420, 0.181]},
    '3C147': {'coords': SkyCoord('05h42m36.1s', '+49d51m07s'), 'scale': 'SH12', 'coeffs': [66.738, -0.022, -1.017, 0.549]},
    '3C196': {'coords': SkyCoord('08h13m36.0s', '+48d13m03s'), 'scale': 'SH12', 'coeffs': [83.084, -0.699, -0.110]},
    '3C286': {'coords': SkyCoord('13h31m08.3s', '+30d30m33s'), 'scale': 'SH12', 'coeffs': [27.477, -0.158, 0.032, -0.180]},
    '3C295': {'coords': SkyCoord('14h11m20.5s', '+52d12m10s'), 'scale': 'SH12', 'coeffs': [97.763, -0.582, -0.298, 0.583, -0.363]},
    '3C380': {'coords': SkyCoord('18h29m31.8s', '+48d44m46s'), 'scale': 'SH12', 'coeffs': [77.352, -0.767]},
    '3C123': {'coords': SkyCoord('04h37m04.4s', '+29d40m14s'), 'scale': 'PB17', 'coeffs': [1.8017, -0.7884, -0.1035, -0.0248, 0.0090]},
}


class BeamModel:
    def __init__(self, h5_path):
        self.path = h5_path
        self.interpolator = None
        self.loaded = False
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Beam file not found at: {self.path}")

    def load_data(self):
        with h5py.File(self.path, 'r') as hf:
            fq_orig = hf['freq_Hz'][:]
            th_orig = hf['theta_pts'][:]
            ph_orig = hf['phi_pts'][:]
            Exth = hf['X_pol_Efields/etheta'][:]
            Exph = hf['X_pol_Efields/ephi'][:]
            Eyth = hf['Y_pol_Efields/etheta'][:]
            Eyph = hf['Y_pol_Efields/ephi'][:]

        if np.max(th_orig) < 10.0: th_orig = np.degrees(th_orig)
        if np.max(ph_orig) < 10.0: ph_orig = np.degrees(ph_orig)

        StokesI = (np.abs(Exth)**2 + np.abs(Exph)**2 + np.abs(Eyth)**2 + np.abs(Eyph)**2)
        fq_idx = np.argsort(fq_orig)
        th_idx = np.argsort(th_orig)
        ph_idx = np.argsort(ph_orig)

        StokesI_s = StokesI[fq_idx,:,:][:,th_idx,:][:,:,ph_idx]
        zenith_idx = np.argmin(np.abs(th_orig[th_idx]))
        zenith_vals = StokesI_s[:, zenith_idx, 0]
        zenith_vals[zenith_vals == 0] = 1.0

        self.norm_beam = StokesI_s / zenith_vals[:, np.newaxis, np.newaxis]
        self.interpolator = RegularGridInterpolator(
            (fq_orig[fq_idx], th_orig[th_idx], ph_orig[ph_idx]),
            self.norm_beam, bounds_error=False, fill_value=0.0
        )
        self.loaded = True

    def get_response(self, ra, dec, obs_time, freq_hz):
        if not self.loaded: self.load_data()
        time_obj = Time(obs_time, location=OVRO_LOC)
        sc = SkyCoord(ra, dec, unit='deg')
        altaz = sc.transform_to(AltAz(obstime=time_obj, location=OVRO_LOC))

        az = altaz.az.deg
        el = altaz.alt.deg
        theta = 90.0 - el
        phi = az % 360.0

        if np.isscalar(ra):
            pts = [freq_hz, theta, phi]
        else:
            n_pts = len(ra)
            pts = np.column_stack((np.full(n_pts, freq_hz), theta, phi))

        resp = self.interpolator(pts)
        if np.isscalar(el):
            if el < 10.0: resp = np.nan
        else:
            resp[el < 10.0] = np.nan
        return resp


def parse_vlssr_text(path):
    coords, fluxes, maj_axes = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8: continue
        try:
            ra_str = f"{parts[0]}h{parts[1]}m{parts[2]}s"
            dec_str = f"{parts[3]}d{parts[4]}m{parts[5]}s"
            flux_val = float(parts[7].replace('<',''))

            maj_deg = 0.0
            if len(parts) >= 11:
                maj_deg = float(parts[9].replace('<','')) / 3600.0

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                coords.append(SkyCoord(ra_str, dec_str, frame='icrs'))

            fluxes.append(flux_val)
            maj_axes.append(maj_deg)
        except Exception: continue

    if not coords:
        return None

    return {
        'coords': SkyCoord(coords),
        'flux': np.array(fluxes),
        'maj': np.array(maj_axes),
        'ra': np.array([c.ra.deg for c in coords]),
        'dec': np.array([c.dec.deg for c in coords]),
        'type': 'VLSS'
    }


def load_catalog(path, name):
    is_fits = path.lower().endswith(('.fits', '.fit'))
    tab = None
    cat_type = 'VLSS' if 'VLSS' in name.upper() else 'NVSS'

    if is_fits:
        try:
            with fits.open(path) as hdul:
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                tab = Table(data)
        except Exception:
            pass
    elif not is_fits and "VLSS" in name.upper():
        return parse_vlssr_text(path)

    if tab is not None:
        cols = tab.colnames
        ra_col = next((c for c in cols if 'RA' in c.upper()), None)
        dec_col = next((c for c in cols if 'DEC' in c.upper()), None)
        flux_col = next((c for c in cols if any(x in c.upper() for x in ['S1400','S74','PEAK','INT']) and 'ERR' not in c.upper()), None)
        maj_col = next((c for c in cols if 'MAJ' in c.upper() and 'ERR' not in c.upper()), None)

        if all([ra_col, dec_col, flux_col]):
            mask = ~np.isnan(tab[ra_col]) & ~np.isnan(tab[dec_col]) & ~np.isnan(tab[flux_col])
            tab = tab[mask]

            maj_vals = np.zeros(len(tab))
            if maj_col:
                maj_vals = np.array(tab[maj_col])
                if len(maj_vals) > 0 and np.median(maj_vals) > 1.0: maj_vals /= 3600.0

            flux_vals = np.array(tab[flux_col])
            if np.nanmedian(flux_vals) > 50: flux_vals /= 1000.0

            return {
                'ra': np.array(tab[ra_col]),
                'dec': np.array(tab[dec_col]),
                'flux': flux_vals,
                'maj': maj_vals,
                'coords': SkyCoord(tab[ra_col], tab[dec_col], unit='deg'),
                'type': cat_type
            }
    return None


def calc_model_flux(name, freq_mhz):
    if name not in CALIB_DATA: return np.nan, "UNK"
    data = CALIB_DATA[name]
    coeffs = data['coeffs']

    if data['scale'] == 'SH12':
        nu_rat = freq_mhz / 150.0
        log_nu = np.log10(nu_rat)
        log_s = np.log10(coeffs[0])
        for i, a_i in enumerate(coeffs[1:], 1):
            log_s += a_i * (log_nu ** i)
        return 10**log_s, "SH12"

    elif data['scale'] == 'PB17':
        nu_ghz = freq_mhz / 1000.0
        log_nu = np.log10(nu_ghz)
        log_s = 0.0
        for i, a_i in enumerate(coeffs):
            log_s += a_i * (log_nu ** i)
        return 10**log_s, "PB17"

    return np.nan, "UNK"


def print_calibrator_table(img_sources, cat_data, obs_date, freq_mhz):
    print("\n" + "="*95)
    print(f"CALIBRATOR FLUX CHECK ({freq_mhz:.2f} MHz)")
    print("="*95)
    print(f"{'Source':<8} {'El':<5} {'Scale':<5} {'S_Model':<8} {'S_VLSSr':<8} {'S_LWA':<8} {'Diff(%)':<8}")
    print("-" * 95)

    time_obj = Time(obs_date, location=OVRO_LOC)
    lwa_coords = SkyCoord(img_sources['ra'].values*u.deg, img_sources['dec'].values*u.deg)

    for name, data in CALIB_DATA.items():
        coord = data['coords']
        altaz = coord.transform_to(AltAz(obstime=time_obj, location=OVRO_LOC))
        el = altaz.alt.deg
        if el < 0: continue

        s_model, scale_name = calc_model_flux(name, freq_mhz)

        s_vlss = "---"
        if cat_data is not None:
            idx_v, d2d_v, _ = coord.match_to_catalog_sky(cat_data['coords'])
            if d2d_v < (2.0 * u.arcmin):
                s_vlss = f"{cat_data['flux'][idx_v]:.2f}"

        s_lwa_str = "---"
        diff_str = "---"
        idx, d2d, _ = coord.match_to_catalog_sky(lwa_coords)
        if d2d < (10.0 * u.arcmin):
            s_lwa_val = img_sources.iloc[idx]['flux_peak_I_app']
            s_lwa_str = f"{s_lwa_val:.2f}"
            if not np.isnan(s_model):
                diff = 100.0 * (s_lwa_val - s_model) / s_model
                diff_str = f"{diff:<+6.1f}"

        print(f"{name:<8} {el:<5.1f} {scale_name:<5} {s_model:<8.2f} {s_vlss:<8} {s_lwa_str:<8} {diff_str}")
    print("="*95 + "\n")


def plot_distortion_map(mag, sx, sy, nx, title, outname):
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

    if np.ma.is_masked(sx_sub): sx_sub = sx_sub.filled(np.nan)
    if np.ma.is_masked(sy_sub): sy_sub = sy_sub.filled(np.nan)

    min_y = min(Y.shape[0], sx_sub.shape[0])
    min_x = min(X.shape[1], sx_sub.shape[1])

    plt.imshow(mag, origin='lower', cmap='gray', alpha=0.3)
    plt.quiver(X[:min_y, :min_x], Y[:min_y, :min_x],
               sx_sub[:min_y, :min_x], sy_sub[:min_y, :min_x],
               color='cyan', scale=None, scale_units='xy')

    plt.savefig(outname, dpi=150)
    plt.close()


def generate_warp_screens(img_sources, cat_data, wcs, shape, freq_mhz, cat_freq_mhz, lwa_bmaj_deg, clip_sigma, base_name=""):
    diag_dir = "Dewarp_Diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    cat_limit = VLSS_MAX_SIZE_DEG if cat_data['type'] == 'VLSS' else NVSS_MAX_SIZE_DEG

    if 'maj' in img_sources.columns:
        mask_point_lwa = img_sources['maj'] < (lwa_bmaj_deg * LWA_SIZE_TOLERANCE)
    else:
        mask_point_lwa = np.ones(len(img_sources), dtype=bool)

    lwa_unresolved = img_sources[mask_point_lwa].copy()
    lwa_unresolved.sort_values(by='flux_peak_I_app', ascending=False, inplace=True)
    if lwa_unresolved.empty: return None, None, None, None

    if 'maj' in cat_data:
        mask_cat_point = cat_data['maj'] < cat_limit
    else:
        mask_cat_point = np.ones(len(cat_data['ra']), dtype=bool)

    cat_coords_filt = cat_data['coords'][mask_cat_point]
    cat_flux_filt = cat_data['flux'][mask_cat_point]
    cat_ra_filt = cat_data['ra'][mask_cat_point]
    cat_dec_filt = cat_data['dec'][mask_cat_point]

    c_img = SkyCoord(lwa_unresolved['ra'].values*u.deg, lwa_unresolved['dec'].values*u.deg)
    idx, d2d, _ = c_img.match_to_catalog_sky(cat_coords_filt)

    predicted_flux = cat_flux_filt[idx] * ((freq_mhz/cat_freq_mhz)**DEFAULT_ALPHA)
    lwa_flux = lwa_unresolved['flux_peak_I_app'].values
    flux_ratio = lwa_flux / predicted_flux

    log_freq_ratio = abs(np.log10(freq_mhz / cat_freq_mhz))
    tolerance_factor = max(1.0, 1.0 + 3.0 * log_freq_ratio)
    eff_flux_min = FLUX_RATIO_MIN / tolerance_factor
    eff_flux_max = FLUX_RATIO_MAX * tolerance_factor

    mask_pos = d2d < MATCH_RADIUS
    mask_flux = (flux_ratio > eff_flux_min) & (flux_ratio < eff_flux_max)
    mask_warp = mask_pos & mask_flux

    n_warp = np.sum(mask_warp)

    if n_warp < 6: return None, None, None, None

    anchors_warp = {
        'ra_lwa': lwa_unresolved['ra'].values[mask_warp],
        'dec_lwa': lwa_unresolved['dec'].values[mask_warp],
        'ra_cat': cat_ra_filt[idx[mask_warp]],
        'dec_cat': cat_dec_filt[idx[mask_warp]]
    }

    anchors_diag = {
        'ra_lwa': lwa_unresolved['ra'].values[mask_pos],
        'dec_lwa': lwa_unresolved['dec'].values[mask_pos],
        'flux_lwa': lwa_flux[mask_pos],
        'flux_cat': predicted_flux[mask_pos]
    }

    mx, my = wcs.all_world2pix(anchors_warp['ra_lwa'], anchors_warp['dec_lwa'], 0)
    tx, ty = wcs.all_world2pix(anchors_warp['ra_cat'], anchors_warp['dec_cat'], 0)

    ny, nx = shape
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]

    screen_x = griddata((ty, tx), mx - tx, (grid_y, grid_x), method='linear', fill_value=0.0)
    screen_y = griddata((ty, tx), my - ty, (grid_y, grid_x), method='linear', fill_value=0.0)

    mask_nan = np.isnan(screen_x)
    if np.any(mask_nan):
        interp_nn_x = NearestNDInterpolator(list(zip(ty, tx)), mx - tx)
        interp_nn_y = NearestNDInterpolator(list(zip(ty, tx)), my - ty)
        screen_x[mask_nan] = interp_nn_x(grid_y[mask_nan], grid_x[mask_nan])
        screen_y[mask_nan] = interp_nn_y(grid_y[mask_nan], grid_x[mask_nan])

    try:
        scales = wcs.proj_plane_pixel_scales()
        scale_deg = np.mean([scales[0].value, scales[1].value])
        shift_mag = np.sqrt(screen_x**2 + screen_y**2) * scale_deg * 60.0
        plot_out = os.path.join(diag_dir, f"distortion_map_raw_{base_name}.png")
        plot_distortion_map(shift_mag, screen_x, screen_y, nx, "Raw", plot_out)
    except Exception:
        pass

    return screen_x, screen_y, anchors_diag, (mx, my, tx, ty)


def apply_warp(data, screen_x, screen_y):
    ny, nx = data.shape
    sy, sx = screen_x.shape
    if (ny != sy) or (nx != sx):
        return None
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]
    return map_coordinates(data, [grid_y + screen_y, grid_x + screen_x], order=1, mode='constant', cval=0.0)


def fit_primary_beam(anchors, obs_date, freq_mhz, base_name, beam_model):
    diag_dir = "Dewarp_Diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    ra_lwa = anchors['ra_lwa']
    dec_lwa = anchors['dec_lwa']
    flux_lwa = anchors['flux_lwa']
    flux_cat = anchors['flux_cat']

    c_anchors = SkyCoord(ra_lwa, dec_lwa, unit='deg')
    time_obj = Time(obs_date, location=OVRO_LOC)
    altaz = c_anchors.transform_to(AltAz(obstime=time_obj, location=OVRO_LOC))
    el_vals = altaz.alt.deg

    beam_theory = beam_model.get_response(ra_lwa, dec_lwa, obs_date, freq_mhz*1e6)
    recovery = flux_lwa / flux_cat

    plt.figure(figsize=(8, 8))
    plt.scatter(flux_lwa, flux_cat, c=el_vals, cmap='plasma', alpha=0.8, edgecolors='k', s=20)
    plt.plot([min(flux_lwa), max(flux_lwa)], [min(flux_lwa), max(flux_lwa)], 'r--', label='1:1')
    plt.colorbar(label='Elevation (deg)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('LWA Flux (Jy)'); plt.ylabel('Catalog Flux (Jy)')
    plt.title("Flux Comparison (Spatial Matches)")
    plt.savefig(os.path.join(diag_dir, f"flux_comparison_{base_name}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(el_vals, recovery, alpha=0.5, label='Meas/Cat Ratio', s=15)
    plt.scatter(el_vals, beam_theory, alpha=0.3, color='red', label='Model Beam', s=5)
    plt.axhline(1.0, color='k', linestyle='--')
    plt.xlabel("Elevation (deg)"); plt.ylabel("Ratio")
    plt.ylim(0, 3.0)
    plt.legend()
    plt.title("Beam Diagnostic (Spatial Matches)")
    plt.savefig(os.path.join(diag_dir, f"beam_diagnostic_{base_name}.png"), dpi=150)
    plt.close()

    out_df = pd.DataFrame({
        'Source_RA': ra_lwa, 'Source_Dec': dec_lwa,
        'El': el_vals, 'Meas_Flux_Corr': flux_lwa, 'Cat_Flux': flux_cat,
        'Ratio': recovery, 'Model_Beam_Val': beam_theory
    })
    out_df.to_csv(os.path.join(diag_dir, f"{base_name}_beam_samples.csv"), index=False)
