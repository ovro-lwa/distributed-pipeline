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

NARROW_ONLY = True

log = logging.getLogger(__name__)

CLEAN_THRESHOLD_SUN_JY = 50 if NARROW_ONLY else 5
CLEAN_THRESHOLD_JY = 50 if NARROW_ONLY else 20

CLEAN_MGAIN = 0.8
SUN_CHANNELS_OUT = 2

IMSIZE = 4096
IM_SCALE_DEGREE = 0.03125


def make_movie_from_fits(fits_tuple: Tuple[str], output_dir: str, scale: float,
                         output_filename: Optional[str] = None) -> str:
    # Check this out https://github.com/will-henney/fits2image
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
                     briggs: float = 0, inner_tukey: Optional[int] = None, n_thread: int = 10,
                     more_args: Optional[List[str]] = None) -> Union[str, Tuple[str, str]]:
    """Make dirty image out of list of measurement sets.

    Args:
        ms_list:
        output_dir:
        output_prefix:
        make_psf:
        briggs:
        inner_tukey:
        n_thread:
        more_args:

    Returns: if make_psf, (image path, psf path), else just the image path.

    """
    taper_args = ['-taper-inner-tukey', str(inner_tukey)] if inner_tukey else []

    extra_args = ['-size', str(IMSIZE), str(IMSIZE), '-scale', str(IM_SCALE_DEGREE),
                  '-niter', '0', '-weight', 'briggs', str(briggs),
                  '-no-update-model-required', '-no-reorder',
                  '-j', str(n_thread)] + taper_args

    if more_args:
        extra_args += more_args
    wsclean.wsclean(ms_list, output_dir, output_prefix, extra_arg_list=extra_args)
    if make_psf:
        extra_args = ['-size', str(2 * IMSIZE), str(2 * IMSIZE), '-scale', str(IM_SCALE_DEGREE),
                      '-niter', '0', '-weight', 'briggs', str(briggs),
                      '-no-update-model-required', '-no-reorder', '-make-psf-only',
                      '-j', str(n_thread)] + taper_args
        wsclean.wsclean(ms_list, output_dir, output_prefix, extra_arg_list=extra_args)
        return f'{output_dir}/{output_prefix}-image.fits', f'{output_dir}/{output_prefix}-psf.fits'
    else:
        return f'{output_dir}/{output_prefix}-image.fits'
