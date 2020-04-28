from typing import Tuple, Optional, List
import subprocess
from os import path
from tempfile import TemporaryDirectory

from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt

from orca.utils import fitsutils
from orca.wrapper import wsclean
from orca.wrapper.wsclean import WSCLEAN_1_11_EXEC, NEW_ENV

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


def make_residual_image_with_source_removed(ms_list: List[str], date_times_string: str, out_dir: str,
                                            source_to_remove: str) -> str:
    residual_fits = ''
    # Clean up
    return residual_fits


def make_dirty_image(ms_list: List[str], date_times_string: str, out_dir: str, make_psf: bool = False) ->\
        Tuple[str, Optional[str]]:
    extra_args = ['-size', '4096', '4096', '-scale', '0.03125',
                  '-niter', '0', '-weight', 'briggs', '0',
                  '-no-update-model-required', '-no-reorder',
                  '-j', '10', '-name', f'{out_dir}/{date_times_string}']
    wsclean.wsclean(ms_list, out_dir, f'{out_dir}/{date_times_string}', extra_args)
    if make_psf:
        extra_args = ['-size', '8192', '8192', '-scale', '0.03125',
                      '-niter', '0', '-weight', 'briggs', '0',
                      '-no-update-model-required', '-no-reorder', '-make-psf-only',
                      '-j', '10', '-name', f'{out_dir}/{date_times_string}']
        return f'{out_dir}/{date_times_string}-image.fits', f'{out_dir}/{date_times_string}-psf.fits'
    else:
        return f'{out_dir}/{date_times_string}-image.fits'
