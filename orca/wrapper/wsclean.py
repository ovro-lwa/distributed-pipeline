"""wsclean wrapper
"""
import subprocess
import os
from typing import List
import logging
from orca.configmanager import execs

log = logging.getLogger(__name__)

NEW_ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1')

def wsclean(ms_list: List[str], out_dir: str, filename_prefix: str, extra_arg_list: List[str],
            num_threads: int=1, mem_gb: int=50) -> None:
    """Run wsclean with arguments and put output fits files in out_dir.
    wsclean will writes the following files (depending on the options):

    Dirty image: {out_dir}/{filename_prefix}-dirty.fits

    Cleaned image (dirty if -niter 0): {out_dir}/{filename_prefix}-image.fits

    PSF (if -make-psf or cleaning enabled): {out_dir}/{filename_prefix}-psf.fits

    Residual image (if cleaning enabled): {out_dir}/{filename_prefix}-residual.fits

    Args:
        ms_list: List of input measurement sets.
        out_dir: Output directory.
        filename_prefix: Prefix for output fits files.
        extra_arg_list: List of arguments to supply to wsclean in addition to the ones about file names.

    Returns: None. But you can reconstruct the fits file names with the wsclean conventions.

    """
    args_list = [execs.wsclean] + ['-j', str(num_threads), '-abs-mem', str(mem_gb)] + extra_arg_list + \
        ['-name', f'{out_dir}/{filename_prefix}'] + ms_list
    proc = subprocess.Popen(args_list, env=NEW_ENV)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            if stdoutdata:
                log.error(stdoutdata.decode())
            if stderrdata:
                log.error(f'Error in wsclean: {stderrdata.decode()}')
            raise Exception('Error in wsclean.')
    finally:
        proc.terminate()
