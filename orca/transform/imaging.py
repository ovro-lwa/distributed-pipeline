"""Transforms that make images
"""
from typing import Tuple, Optional, List, Union
import subprocess
import os
import logging
from os import path
from datetime import datetime
from tempfile import TemporaryDirectory

from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from astropy import wcs
from astropy.coordinates import SkyCoord
import numpy as np

from orca.utils import fitsutils, coordutils
from orca.wrapper import wsclean

log = logging.getLogger(__name__)

CLEAN_THRESHOLD_JY = 5
CLEAN_THRESHOLD_JY_CRAB = 20

CLEAN_MGAIN = 0.8
SUN_CHANNELS_OUT = 2

IMSIZE = 4096
IM_SCALE_DEGREE = 0.03125


def make_movie_from_fits(fits_tuple: Tuple[str], output_dir: str, scale: float,
                         output_filename: Optional[str] = None) -> str:
    with TemporaryDirectory() as tmpdir:
        dpi = 200
        for i, fn in enumerate(fits_tuple):
            fig = plt.figure(figsize=(1024./dpi, 1024./dpi), dpi=dpi)
            im, _ = fitsutils.read_image_fits(fn)
            ax = fig.add_subplot(111)
            ax.imshow(im, origin='lowerleft', norm=mpl_colors.Normalize(vmin=-scale, vmax=scale), cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.text(10, 10, f'{path.basename(fn)}', color='white')
            plt.savefig(path.join(tmpdir, f'frame{i:03d}.jpg'), bbox_inches='tight', pad_inches=0)
            plt.close()

        if output_filename is None:
            output_filename = f'{path.splitext(path.basename(fits_tuple[0]))[0]}.mp4'

        output_path = path.join(output_dir, output_filename)
        try:
            subprocess.check_output(['ffmpeg', '-f', 'image2', '-pattern_type', 'glob', '-i', f'{tmpdir}/frame*.jpg',
                                     '-codec:v', 'libx264', output_path])
        except subprocess.CalledProcessError as e:
            print(e.output)
    return output_path


def make_residual_image_with_source_removed(ms_list: List[str], timestamp: datetime, output_dir: str,
                                            output_prefix: str, tmp_dir: str,
                                            inner_tukey: Optional[int] = None, n_thread: int = 10) -> str:
    """Make images with bright source(s) removed.
    Makes a dirty image. Remove the Sun and/or the Crab when they are up.

    Args:
        ms_list: List of measurement sets to make an image out of.
        timestamp: Timestamp used to calculate what sources are up.
        output_dir: Output directory for the images.
        output_prefix: Image prefix (as required by wsclean).
        tmp_dir: Temporary directory to hold wsclean re-ordered files.
        inner_tukey: Inner Tukey Parameter.
        n_thread: Number of threads for wsclean to use.

    Returns: Path to the output image (residing in output_dir).

    """
    log.info(f'ms_list is {ms_list}')
    dirty_image = make_dirty_image(ms_list, output_dir, output_prefix, inner_tukey=inner_tukey)

    extra_args = ['-size', str(IMSIZE), str(IMSIZE), '-scale', str(IM_SCALE_DEGREE),
                  '-weight', 'briggs', '0',
                  '-no-update-model-required',
                  '-j', str(n_thread), '-niter', '4000', '-tempdir', tmp_dir]
    taper_args = ['-taper-inner-tukey', str(inner_tukey)] if inner_tukey else []
    assert isinstance(dirty_image, str)
    im, header = fitsutils.read_image_fits(dirty_image)
    im_T = im.T
    fits_mask = f'{output_dir}/{output_prefix}-mask.fits'

    fits_mask_center_list = []
    fits_mask_width_list = []
    channelsout = 0

    if coordutils.is_visible(coordutils.TAU_A, timestamp):
        fits_mask_center_list.append(get_peak_around_source(im_T, coordutils.TAU_A, wcs.WCS(header)))
        fits_mask_width_list.append(3)

    sun_icrs = coordutils.sun_icrs(timestamp)
    if coordutils.is_visible(sun_icrs, timestamp):
        fits_mask_center_list.append(get_peak_around_source(im_T, sun_icrs, wcs.WCS(header)))
        fits_mask_width_list.append(81)
        channelsout = SUN_CHANNELS_OUT

    if fits_mask_width_list:

        fitsutils.write_fits_mask_with_box_xy_coordindates(fits_mask, imsize=im.T.shape[0],
                                                           center_list=fits_mask_center_list,
                                                           width_list=fits_mask_width_list)
        if channelsout:
            wsclean.wsclean(ms_list, output_dir, output_prefix, extra_arg_list=extra_args +
                            ['-channelsout', str(channelsout), '-fitsmask', fits_mask,
                             '-threshold', str(CLEAN_THRESHOLD_JY),
                             '-mgain', str(CLEAN_MGAIN)] +
                            taper_args)
            os.renames(f'{output_dir}/{output_prefix}-MFS-residual.fits', f'{output_dir}/{output_prefix}-image.fits')
        else:
            wsclean.wsclean(ms_list, output_dir, output_prefix,
                            extra_arg_list=extra_args +
                                           ['-fitsmask', fits_mask, '-threshold',
                                            str(CLEAN_THRESHOLD_JY_CRAB),
                                            '-mgain', str(CLEAN_MGAIN)] +
                                           taper_args)
            os.renames(f'{output_dir}/{output_prefix}-residual.fits', f'{output_dir}/{output_prefix}-image.fits')
    else:
        log.info(f'No sources to remove for {ms_list}')
    return f'{output_dir}/{output_prefix}-image.fits'


def get_peak_around_source(im_T: np.ndarray, source_coord: SkyCoord, w: wcs.WCS) -> Tuple[int, int]:
    x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
    x_start = int(x) - 100
    y_start = int(y) - 100
    im_box = im_T[x_start:x_start + 200, y_start:y_start + 200]
    peakx, peaky = np.unravel_index(np.argmax(im_box),
                                    im_box.shape)
    peakx += x_start
    peaky += y_start
    return peakx, peaky


def make_dirty_image(ms_list: List[str], output_dir: str, output_prefix: str, make_psf: bool = False,
                     inner_tukey: Optional[int] = None, n_thread: int = 10) -> Union[str, Tuple[str, str]]:
    """Make dirty image out of list of measurement sets.

    Args:
        ms_list:
        output_dir:
        output_prefix:
        make_psf:
        inner_tukey:
        n_thread:

    Returns: if make_psf, (image path, psf path), else just the image path.

    """
    taper_args = ['-taper-inner-tukey', str(inner_tukey)] if inner_tukey else []

    extra_args = ['-size', str(IMSIZE), str(IMSIZE), '-scale', str(IM_SCALE_DEGREE),
                  '-niter', '0', '-weight', 'briggs', '0',
                  '-no-update-model-required', '-no-reorder',
                  '-j', str(n_thread)] + taper_args
    wsclean.wsclean(ms_list, output_dir, output_prefix, extra_arg_list=extra_args)
    if make_psf:
        extra_args = ['-size', str(2 * IMSIZE), str(2 * IMSIZE), '-scale', str(IM_SCALE_DEGREE),
                      '-niter', '0', '-weight', 'briggs', '0',
                      '-no-update-model-required', '-no-reorder', '-make-psf-only',
                      '-j', str(n_thread)] + taper_args
        wsclean.wsclean(ms_list, output_dir, output_prefix, extra_arg_list=extra_args)
        return f'{output_dir}/{output_prefix}-image.fits', f'{output_dir}/{output_prefix}-psf.fits'
    else:
        return f'{output_dir}/{output_prefix}-image.fits'
