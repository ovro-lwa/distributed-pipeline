"""Transforms that make images
"""
from typing import Tuple, Optional, List, Union
import uuid
import subprocess
import shutil
import os
import logging
from os import path
from datetime import datetime
from tempfile import TemporaryDirectory
from glob import glob

from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from astropy import wcs
from astropy.coordinates import SkyCoord
import numpy as np

from orca.utils import fitsutils, coordutils, copyutils
from orca.wrapper import wsclean
from orca.transform.flagging import flag_with_aoflagger, flag_ants, flag_on_autocorr
from orca.metadata.stageiii import StageIIIPathsManager
from orca.transform.integrate import integrate
from orca.transform.calibration import applycal_data_col_nocopy
from orca.celery import app

logger = logging.getLogger(__name__)

CLEAN_MGAIN = 0.8
SUN_CHANNELS_OUT = 2

IMSIZE = 4096
IM_SCALE_DEGREE = 0.03125


from kombu.utils.json import register_type

register_type(SkyCoord, 'SkyCoord',
              SkyCoord.to_string,
              lambda s: SkyCoord(s, unit='degree'))

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


def integrated_image(msl: List[str]):
    pass


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


@app.task(autoretry_for=(Exception,), max_retries=1)
def stokes_IV_imaging(spw_list:List[str], start_time: datetime, end_time: datetime,
                        source_dir: str, work_dir: str, scratch_dir: str,
                        phase_center: Optional[SkyCoord] = None, taper_inner_tukey: int = 30,
                        make_snapshots: bool = False,
                        keep_sratch_dir: bool = False):
    s = start_time
    e = end_time
    tmpdir = f'{scratch_dir}/tmp-{str(uuid.uuid4())}'
    os.mkdir(tmpdir)
    datetime_list = []
    try:
        integrated_msl = []
        n_timesteps = None
        if phase_center is None:
            phase_center = coordutils.zenith_coord_at_ovro(start_time + (end_time - start_time) / 2)
        for spw in spw_list:
            logger.info('Applycal SPW %s', spw)
            pm = StageIIIPathsManager(source_dir, work_dir, spw, s, e)
            if not pm.ms_list:
                logger.warning('No measurement sets found for SPW %s', spw)
                continue
            if not path.exists(pm.get_bcal_path(s.date())):
                logger.warning('No bandpass solutions found for SPW %s', spw)
                continue
            msl = []
            with ThreadPoolExecutor(20) as pool:
                futures = [ pool.submit(shutil.copytree, ms, f'{tmpdir}/{path.basename(ms)}'
                                        , copy_function=copyutils.copy) for _, ms in pm.ms_list ]
                """
                for _, ms in pm.ms_list:
                    # applycal
                    msl.append(applycal_data_col(ms, pm.get_bcal_path(s.date()),
                                    f'{tmpdir}/{path.basename(ms)}'))
                """
                for r in as_completed(futures):
                    m = r.result()
                    applycal_data_col_nocopy(m, pm.get_bcal_path(s.date()))

                    flag_on_autocorr(m, s.date())
                    msl.append(m)

            msl.sort()
            if n_timesteps is None:
                n_timesteps = len(msl)
                datetime_list = [dt for dt, _ in pm.ms_list]

            logger.info('Integrating SPW %s', spw)
            integrated = integrate(msl, f'{tmpdir}/{spw}.ms', phase_center=phase_center)
            for ms in msl:
                shutil.rmtree(ms)
        # flag
            logger.info('Flagging SPW %s', spw)
            # flag_ants(integrated, [70,79,80,117,137,193,150,178,201,208,224,261,215,236,246,294,298,301,307,289,33,3,41,42,44,92,12,14,17,21,154,29,28,127,126])
            flag_with_aoflagger(integrated)
            integrated_msl.append(integrated)

        logger.info('Done with all spws. Start imaging.')
        if integrated_msl:
            # image
            arg_list=['-weight', 'briggs', '0', '-niter', '0', '-size', '4096', '4096', '-scale', '0.03125', '-pol', 'IV',
                            '-no-update-model-required', '-taper-inner-tukey', str(taper_inner_tukey)]
            if make_snapshots:
                arg_list += ['-intervals-out', str(n_timesteps)]

            print(integrated_msl)
            print(arg_list)
            wsclean.wsclean(integrated_msl, tmpdir, 'OUT',
                    extra_arg_list=arg_list,
                    num_threads=20, mem_gb=100)

            if not make_snapshots:
                out_path = pm.data_product_path(s, f'V.image.fits')
                os.makedirs(path.dirname(out_path), exist_ok=True)
                shutil.copy(f'{tmpdir}/OUT-V-image.fits', out_path)

                out_path = pm.data_product_path(s, f'I.image.fits')
                os.makedirs(path.dirname(out_path), exist_ok=True)
                shutil.copy(f'{tmpdir}/OUT-I-image.fits', out_path)
            else:
                out_images = sorted(glob(f'{tmpdir}/OUT-t*-I-image.fits'))
                for dt, fitsname in zip(datetime_list, out_images):
                    out_path = pm.data_product_path(dt, f'snap.I.image.fits')
                    os.makedirs(path.dirname(out_path), exist_ok=True)
                    shutil.copy(fitsname, out_path)

            logger.info('Done imaging.')
    finally:
        if not keep_sratch_dir:
            shutil.rmtree(tmpdir)


def coadd_fits(fits_list: List[str], output_path: str) -> Optional[str]:
    n = len(fits_list)
    if n < 2:
        logger.warning('Cannot coadd less than 2 images.')
        return None
    avg, header = fitsutils.read_image_fits(fits_list[0])
    avg /= n
    for f in fits_list[1:]:
        dat, _ = fitsutils.read_image_fits(f)
        avg += dat / n
    fitsutils.write_image_fits(output_path, header, avg, overwrite=True)
    return output_path
