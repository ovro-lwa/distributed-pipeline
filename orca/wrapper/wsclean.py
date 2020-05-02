import subprocess
import os
from typing import List, Tuple
import logging

log = logging.getLogger(__name__)

WSCLEAN_1_11_EXEC = '/opt/astro/wsclean-1.11-gcc4.8.5_cxx11/bin/wsclean'

NEW_ENV = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/:/opt/astro/casacore-1.7.0/lib',
               AIPSPATH='/opt/astro/casa-data dummy dummy')


def wsclean(ms_list: List[str], out_dir: str, filename_prefix: str, extra_arg_list: List[str]) -> None:
    extra_arg_list = [WSCLEAN_1_11_EXEC] + extra_arg_list + ['-name', f'{out_dir}/{filename_prefix}'] + ms_list
    proc = subprocess.Popen(extra_arg_list, env=NEW_ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode is not 0:
            if stderrdata:
                log.error(f'Error in wsclean: {stderrdata.decode()}')
            if stdoutdata:
                log.info(f'stdout is {stdoutdata.decode()}')
            raise Exception('Error in wsclean.')
    finally:
        proc.terminate()
