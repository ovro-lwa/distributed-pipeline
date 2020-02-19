import subprocess
import logging
import os

log = logging.getLogger(__name__)

TTCAL_EXEC = '/opt/astro/mwe/bin/ttcal-0.3.0'


def peel_with_ttcal(ms: str, sources: str):
    """
    Use TTCal to peel sources. Assumes that the input measurement set only has DATA column and writes to the
    CORRECTED_DATA column.
    :param ms:
    :param sources:
    :return:
    """
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')
    proc = subprocess.Popen([TTCAL_EXEC, 'peel', ms, sources, '--beam', 'sine', '--maxiter', '50',
                             '--tolerance', '1e-4', '--minuvw', '10'], env=new_env,
                            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    if proc.returncode is not 0:
        logging.error(f'Error in TTCal: {stderrdata}')
        logging.info(f'stdout is {stdoutdata}')
        raise Exception('Error in TTCal.')
    return ms
