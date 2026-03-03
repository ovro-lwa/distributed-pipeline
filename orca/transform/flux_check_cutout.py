"""OVRO-LWA Flux Scale Validation.

Ported from ExoPipe/flux_check_cutout.py into the orca package for
use with the Celery pipeline.

Validates the absolute flux scale of calibrated images against
Scaife & Heald (2012) models using CASA imfit on cutouts around
known calibrators.
"""
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D

warnings.simplefilter('ignore', category=FITSFixedWarning)

# --- OBSERVATORY & CALIBRATOR DATA ---
from orca.resources.subband_config import OVRO_LOC, CALIB_DATA as SH_CATALOG

# --- CASA IMPORTS ---
CASA_TASKS_AVAILABLE = False
try:
    from casatasks import imfit, imstat
    CASA_TASKS_AVAILABLE = True
except ImportError:
    pass

CUTOUT_SIZE = 2.0 * u.deg
SEARCH_RADIUS = 0.25 * u.deg
MIN_ELEVATION = 20.0


def get_model_flux(name, freq_mhz):
    """Predict flux density of a calibrator at given frequency.

    Uses Scaife & Heald (2012) polynomial coefficients stored in
    ``CALIB_DATA``.  The reference frequency is 150 MHz.
    """
    A = SH_CATALOG[name]['coeffs']
    x = np.log10(freq_mhz / 150.0)
    log_s = np.log10(A[0])
    for i in range(1, len(A)):
        log_s += A[i] * (x ** i)
    return 10 ** log_s


def process_cutout(img_path, source_name, source_data, temp_dir):
    """Create a cutout around *source_name* and fit with ``imfit``."""
    try:
        subband = img_path.split('/')[-4]
        with fits.open(img_path) as hdul:
            data = hdul[0].data.squeeze()
            header = hdul[0].header
            wcs = WCS(header).celestial

            try:
                el = source_data['coords'].transform_to(
                    AltAz(obstime=Time(header['DATE-OBS'], format='isot', scale='utc'),
                           location=OVRO_LOC)).alt.deg
                if el < MIN_ELEVATION:
                    return None
            except Exception:
                el = np.nan

            cutout = Cutout2D(data, source_data['coords'],
                              (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs, mode='trim')

            cut_header = cutout.wcs.to_header()
            cut_header['BUNIT'] = 'Jy/beam'
            for kw in ['BMAJ', 'BMIN', 'BPA']:
                if kw in header:
                    cut_header[kw] = header[kw]

            temp_path = os.path.join(temp_dir, f"{source_name}_{subband}.fits")
            fits.writeto(temp_path, cutout.data, cut_header, overwrite=True)

            cx, cy = cutout.wcs.world_to_pixel(source_data['coords'])
            cdelt = np.abs(cutout.wcs.wcs.cdelt[0])
            r_pix = SEARCH_RADIUS.to(u.deg).value / cdelt
            ny, nx = cutout.data.shape
            x1, y1 = max(0, int(cx - r_pix)), max(0, int(cy - r_pix))
            x2, y2 = min(nx - 1, int(cx + r_pix)), min(ny - 1, int(cy + r_pix))
            box = f"{x1},{y1},{x2},{y2}"

            imfit_flux, imfit_err = np.nan, np.nan
            if CASA_TASKS_AVAILABLE:
                try:
                    fit = imfit(imagename=temp_path, box=box)
                    if fit and fit.get('converged') and 'results' in fit:
                        imfit_flux = fit['results']['component0']['flux']['value'][0]
                        imfit_err = fit['results']['component0']['flux']['error'][0]
                except Exception:
                    pass

            return {'imfit_flux': imfit_flux, 'imfit_err': imfit_err, 'elevation': el}
    except Exception:
        return None


def run_flux_check(run_dir, logger=None):
    """Validate absolute flux scale against SH12 models.

    Parameters
    ----------
    run_dir : str
        Root directory for the subband run (contains ``<freq>MHz/`` folders).
    logger : logging.Logger, optional
        If supplied, messages go through ``logger.info`` etc.
    """
    log = logger.info if logger else print
    log_warn = logger.warning if logger else print
    log_err = logger.error if logger else print

    log("--- Flux Scale Check ---")
    if not CASA_TASKS_AVAILABLE:
        log_err("casatasks not available — skipping flux check.")
        return

    qa_dir = os.path.join(run_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    temp_dir = os.path.join(qa_dir, "temp_flux_cutouts")
    os.makedirs(temp_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(
        run_dir, "I", "deep",
        "*I-Deep-Taper-Robust-0.75*pbcorr_dewarped*.fits")))

    if not images:
        # Fallback: try non-dewarped pbcorr
        images = sorted(glob.glob(os.path.join(
            run_dir, "I", "deep",
            "*I-Deep-Taper-Robust-0.75*pbcorr*.fits")))
        images = [f for f in images if "dewarped" not in f]

    if not images:
        # Fallback: raw images (no PB correction available)
        images = sorted(glob.glob(os.path.join(
            run_dir, "I", "deep",
            "*I-Deep-Taper-Robust-0.75*image*.fits")))
        images = [f for f in images if "pbcorr" not in f and "dewarped" not in f]

    if not images:
        log("No matching images found for flux check.")
        return

    log(f"  Processing {len(images)} bands")

    results = []
    for img in images:
        try:
            # Extract freq from the subband dir name (e.g. .../55MHz/I/deep/...)
            # run_dir is the subband dir itself, so its basename has the freq
            freq = float(os.path.basename(run_dir).replace('MHz', ''))
        except Exception:
            continue

        for name, data in SH_CATALOG.items():
            res = process_cutout(img, name, data, temp_dir)
            if res:
                try:
                    res['model_flux'] = get_model_flux(name, freq)
                except Exception as e:
                    log_warn(f"Model flux failed for {name} @ {freq} MHz: {e}")
                    continue
                res['source'] = name
                res['freq'] = freq
                results.append(res)

    if results:
        csv_path = os.path.join(qa_dir, "flux_check_hybrid.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        log(f"  Saved CSV: {csv_path}")

        try:
            df = pd.DataFrame(results)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            ax1 = axes[0]
            for src in df['source'].unique():
                sub = df[df['source'] == src].sort_values('freq')
                valid = sub['imfit_flux'].notna() & (sub['model_flux'] > 0)
                if valid.any():
                    ratio = sub.loc[valid, 'imfit_flux'] / sub.loc[valid, 'model_flux']
                    ax1.plot(sub.loc[valid, 'freq'], ratio, 'o-', label=src, ms=5)
            ax1.axhline(1.0, color='grey', ls='--', alpha=0.7)
            ax1.set_xlabel('Frequency (MHz)')
            ax1.set_ylabel('Measured / Model Flux')
            ax1.set_title('Flux Scale Check')
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, 3)
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            valid = df['imfit_flux'].notna() & (df['model_flux'] > 0)
            if valid.any():
                ax2.scatter(df.loc[valid, 'model_flux'], df.loc[valid, 'imfit_flux'],
                            c=df.loc[valid, 'freq'], cmap='viridis', s=30,
                            edgecolors='k', lw=0.5)
                max_flux = max(df.loc[valid, 'model_flux'].max(),
                               df.loc[valid, 'imfit_flux'].max())
                ax2.plot([0, max_flux * 1.1], [0, max_flux * 1.1], 'k--', alpha=0.5)
                ax2.set_xlabel('Model Flux (Jy)')
                ax2.set_ylabel('Measured Flux (Jy)')
                ax2.set_title('Measured vs Model')
                plt.colorbar(ax2.collections[0], ax=ax2, label='Freq (MHz)')
                ax2.grid(True, alpha=0.3)

            png_path = os.path.join(qa_dir, "flux_check_diagnostic.png")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            log(f"  Saved diagnostic plot: {png_path}")
        except Exception as e:
            log_warn(f"Diagnostic plot failed: {e}")
            plt.close('all')
    else:
        log("  No flux check results produced.")
