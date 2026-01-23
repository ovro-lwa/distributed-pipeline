"""Photometry transforms for FITS image analysis.

Provides functions for source detection, flux measurement, noise estimation,
and image quality assessment from radio FITS images.
"""
from typing import Optional, Tuple, List
import logging
import warnings

from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord, ICRS
import matplotlib.pyplot as plt
from astropy.wcs.utils import skycoord_to_pixel

from orca.celery import app

from orca.utils import fitsutils
from orca.utils.coordutils import TAU_BOO
import numpy as np

PATCH_SIDE_SIZE = 12

logger = logging.getLogger(__name__)

def std_and_max_around_coord(fits_file, coord, radius=5):
    """
    Calculate the stdev around a given coordinate in a FITS file.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    coord : astropy.Coordinates
        The coordinate around which to calculate the stdev.
    radius : int, optional
        Radius in pixels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, header = fitsutils.read_image_fits(fits_file)
    w = wcs.WCS(header)
    return fitsutils.std_and_max_around_src(data.T, radius, coord, w)

@app.task
def average_with_rms_threshold(fits_list: List[str], out_fn: str, source_coord: Optional[SkyCoord],
                                radius_px: int, threshold_multiple: float) -> Optional[str]:
    """Average FITS images while rejecting those with high RMS near a source.

    Calculates RMS in a box around the source coordinate and rejects images
    where RMS exceeds the median RMS times threshold_multiple.

    Args:
        fits_list: List of FITS file paths to average.
        out_fn: Output file path.
        source_coord: Coordinate for RMS measurement. If None, no filtering.
        radius_px: Box half-size in pixels for RMS calculation.
        threshold_multiple: Reject images with RMS > median * threshold_multiple.

    Returns:
        Output file path on success.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        contents = [fitsutils.read_image_fits(fn) for fn in fits_list]
        rms_arr = np.zeros(len(contents))
        rms_threshold = np.inf
        if source_coord is not None:
            for i, (data, header) in enumerate(contents):
                w = wcs.WCS(header)
                x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
                if np.isnan(x) or np.isnan(y):
                    raise ValueError(f'Coordinate {source_coord} is not in the image.')
                x = int(x)
                y = int(y)
                im_box = data.T[x - radius_px :x + radius_px, y - radius_px : y + radius_px]
                rms_arr[i] = np.std(im_box)
            rms_threshold = np.median(rms_arr[rms_arr > 0]) * threshold_multiple

        n = np.sum(rms_arr <= rms_threshold)
        out_data = np.zeros_like(contents[0][0])
        logger.info(f'{len(contents)-n} images were skipped.')
        actual_fns = []
        for i, (data, header) in enumerate(contents):
            if rms_arr[i] > rms_threshold:
                continue
            out_data += data / n
            actual_fns.append(fits_list[i])
        out_header = fits.Header()
        out_header['SRCFILES'] = ' '.join(fn.strip('/lustre/celery/') for fn in actual_fns)
        for k in contents[0][1]:
            if ('COMMENT' in k) or ('HISTORY' in k):
                continue
            out_header[k] = contents[0][1][k]
        fitsutils.write_image_fits(out_fn, out_header, out_data, overwrite=True)
    return out_fn

def estimate_image_noise(arr: np.ndarray) -> float:
    """
    Estimate the noise in an image using the median absolute deviation (MAD).

    Parameters
    ----------
    arr : np.ndarray
        The image data.

    Returns
    -------
    float
        The estimated noise.
    """
    return 1.4826 * np.median(np.abs(arr - np.median(arr)))

@app.task
def search_src(fn: str, src: SkyCoord, stats_box_size: int, peak_search_box_size: int) -> Tuple[float, float]:
    """Search for a source and measure its peak flux and local RMS.

    Args:
        fn: Path to the FITS image.
        src: Sky coordinate of the source to measure.
        stats_box_size: Box size in pixels for noise estimation.
        peak_search_box_size: Box size in pixels for peak search.

    Returns:
        Tuple of (peak_flux, rms) values.
    """
    data, header = fitsutils.read_image_fits(fn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w = wcs.WCS(header)
        noise_cutout = fitsutils.get_cutout(data, src, w, stats_box_size // 2)
        rms = estimate_image_noise(noise_cutout)
        peak_cutout = fitsutils.get_cutout(data, src, w, peak_search_box_size // 2)
        peak_cutout_flattened = peak_cutout.flatten()
        peak = peak_cutout_flattened[np.argmax(np.abs(peak_cutout_flattened))]
    return peak, rms


def noise(im: np.ndarray, stats_box_size: int, src: SkyCoord, w: wcs.WCS) -> float:
    """Estimate noise level around a source position.

    Args:
        im: Image data array.
        stats_box_size: Box size in pixels for noise estimation.
        src: Sky coordinate of the source.
        w: WCS object for coordinate transformation.

    Returns:
        Estimated noise level using MAD estimator.
    """
    noise_cutout = fitsutils.get_cutout(im, src, w, stats_box_size // 2)
    return estimate_image_noise(noise_cutout)


def make_fig(v_fns: List[str], src: SkyCoord, stats_box_size: int, out_dir: str):
    """Generate diagnostic figures showing Stokes I and V images.

    Creates side-by-side plots of Stokes I and V cutouts around a source
    with calibrator positions marked.

    Args:
        v_fns: List of Stokes V FITS file paths.
        src: Source coordinate for cutout center.
        stats_box_size: Box size for noise estimation.
        out_dir: Output directory for figures.
    """
    hw = 140 # half width of cutout
    
    s1 = SkyCoord('13h49m39.28s', '+21deg07m28.2s', frame=ICRS)
    s2 = SkyCoord('13h57m4.71s',  '+19deg19m7.7s', frame=ICRS)
    s3 = SkyCoord('13h54m40.61s', '+16deg14m44.9s', frame=ICRS)
    for i, vfn in enumerate(v_fns):
        im2, header2 = fitsutils.read_image_fits(vfn)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = wcs.WCS(header2)
        cutout2 = fitsutils.get_cutout(im2, src, w, hw)
        im1, header1 = fitsutils.read_image_fits(vfn.replace('V', 'I'))
        cutout1 = fitsutils.get_cutout(im1, src, w, hw)
        rms = noise(im2, stats_box_size, src, w)
        fig, axes = plt.subplots(1,2, figsize=(6,3))
        for ax in axes:
            ax.set_axis_off()
        fig.subplots_adjust(wspace=0, hspace=0)

        xy_s1 = np.array(skycoord_to_pixel(s1, w))
        xy_s2 = np.array(skycoord_to_pixel(s2, w))
        xy_s3 = np.array(skycoord_to_pixel(s3, w))
        xy_center = np.array(skycoord_to_pixel(src, w))
        
        axes[0].imshow(cutout1, vmin=-2, vmax=8, origin='lower')
        center = np.array([hw-1,hw-1])
        axes[0].scatter(*(xy_s1 - xy_center + center), marker='+', color='k', s=100)
        axes[0].scatter(*(xy_s2 - xy_center + center), marker='+', color='k', s=100)
        axes[0].scatter(*(xy_s3 - xy_center + center), marker='+', color='k', s=100)
        
        axes[0].scatter(*center, marker='o', color='k', facecolor='none', s=200)
        axes[1].scatter(*center, marker='o', color='k', facecolor='none', s=200)
        
        axes[1].imshow(cutout2, vmin=-2, vmax=2, origin='lower')
        parts = vfn.split('/')
        fig.suptitle(f'{parts[4]} {parts[7]} rms={rms:.3f}Jy')
        plt.savefig(f'{out_dir}/{i:05d}.jpg')
        plt.close(fig)
