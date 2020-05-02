from typing import Tuple, Optional, List, Union
import subprocess
import os
from os import path
from tempfile import TemporaryDirectory

from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from astropy import wcs
from astropy.coordinates import SkyCoord, get_sun
import numpy as np

from orca.utils import fitsutils
from orca.wrapper import wsclean

CRAB = 'CRAB'
SUN = 'SUN'


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


def make_residual_image_with_source_removed(ms_list: List[str], output_dir: str, output_prefix: str,
                                            source_to_remove: str, tmp_dir: str,
                                            inner_tukey: Optional[str] = None) -> str:
    dirty_image = make_dirty_image(ms_list, output_dir, output_prefix, inner_tukey=inner_tukey)
    extra_args = ['-size', '4096', '4096', '-scale', '0.03125',
                  '-weight', 'briggs', '0',
                  '-no-update-model-required',
                  '-j', '10', '-niter', '4000', '-tempdir', tmp_dir]
    taper_args = ['-taper-inner-tukey', inner_tukey] if inner_tukey else []
    im, header = fitsutils.read_image_fits(dirty_image)
    fits_mask = f'{output_dir}/{output_prefix}-mask.fits'
    # Find peak within a box containing the source
    peakx, peaky = get_peak_around_source(im, get_sun(), wcs.WCS(header))
    if source_to_remove is SUN:
        fitsutils.write_fits_mask_with_box(fits_mask, imsize=im.shape[0],
                                           center=(peakx, peaky),
                                           width=45)
        wsclean.wsclean(ms_list, output_dir, f'{output_dir}/{output_prefix}', extra_arg_list=extra_args +
                        ['-channelsout', '2' '-fitsmask', fits_mask, '-threshold', '5', '-mgain', '0.8'] +
                        taper_args)
        os.renames(f'{output_dir}/{output_prefix}-MFS-residual.fits', f'{output_dir}/{output_prefix}-residual.fits')
        # clean up the 0000 and the 0001 stuff.
    elif source_to_remove is CRAB:
        fitsutils.write_fits_mask_with_box(fits_mask, imsize=4096,
                                           center=(peakx, peaky),
                                           width=3)
        wsclean.wsclean(ms_list, output_dir, f'{output_dir}/{output_prefix}', extra_arg_list=extra_args +
                        ['-fitsmask', fits_mask, '-threshold', '5', '-mgain', '0.8'] +
                        taper_args)
    else:
        raise Exception(f'Unknown source to subtract {source_to_remove}.')
    # cp?
    return f'{output_dir}/{output_prefix}-residual.fits'


def get_peak_around_source(im: np.ndarray, source_coord: SkyCoord, w: wcs.WCS) -> Tuple[int, int]:
    x, y = wcs.utils.skycoord_to_pixel(source_coord, w)
    x_start = int(x) - 100
    y_start = int(y) - 100
    im_box = im[x_start:x_start+200, y_start:y_start+200]
    peakx, peaky = np.unravel_index(np.argmax(im_box),
                                    im_box.shape)
    peakx += x_start
    peaky += y_start
    return peakx, peaky


def make_dirty_image(ms_list: List[str], output_dir: str, output_prefix: str, make_psf: bool = False,
                     inner_tukey: Optional[str] = None) ->\
        Union[str, Tuple[str, str]]:
    taper_args = ['-taper-inner-tukey', inner_tukey] if inner_tukey else []

    extra_args = ['-size', '4096', '4096', '-scale', '0.03125',
                  '-niter', '0', '-weight', 'briggs', '0',
                  '-no-update-model-required', '-no-reorder',
                  '-j', '10'] + taper_args
    wsclean.wsclean(ms_list, output_dir, f'{output_dir}/{output_prefix}', extra_arg_list=extra_args)
    if make_psf:
        extra_args = ['-size', '8192', '8192', '-scale', '0.03125',
                      '-niter', '0', '-weight', 'briggs', '0',
                      '-no-update-model-required', '-no-reorder', '-make-psf-only',
                      '-j', '10'] + taper_args
        wsclean.wsclean(ms_list, output_dir, f'{output_dir}/{output_prefix}', extra_arg_list=extra_args)
        return f'{output_dir}/{output_prefix}-image.fits', f'{output_dir}/{output_prefix}-psf.fits'
    else:
        return f'{output_dir}/{output_prefix}-image.fits'
