import subprocess
import os
from typing import List

WSCLEAN_1_11_EXEC = '/opt/astro/wsclean-1.11-gcc4.8.5_cxx11/bin/wsclean'

"""
 -size 4096 4096 -scale 0.03125 -fitbeam -tempdir /dev/shm/yuping/ -niter 0 -weight briggs 0 
 -weighting-rank-filter 3 -weighting-rank-filter-size 128 -no-update-model-required -j 10 -name fullband ??_*ms
"""


def make_image(ms_list: List[str], date_times_string: str, out_dir: str, make_psf: bool = False):
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/:/opt/astro/casacore-1.7.0/lib',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')
    p = subprocess.Popen([WSCLEAN_1_11_EXEC, '-size', '4096', '4096', '-scale', '0.03125',
                          '-niter', '0', '-weight', 'briggs', '0',
                          '-no-update-model-required', '-no-reorder',
                          '-j', '10', '-name', f'{out_dir}/{date_times_string}'] + ms_list, env=new_env)
    p.communicate()
    if make_psf:
        p = subprocess.Popen([WSCLEAN_1_11_EXEC, '-size', '8192', '8192', '-scale', '0.03125',
                              '-niter', '0', '-weight', 'briggs', '0',
                              '-no-update-model-required', '-no-reorder', '-make-psf-only',
                              '-j', '10', '-name', f'{out_dir}/{date_times_string}'] + ms_list, env=new_env)
        p.communicate()
        return f'{out_dir}/{date_times_string}-image.fits', f'{out_dir}/{date_times_string}-psf.fits'
    else:
        return f'{out_dir}/{date_times_string}-image.fits'

