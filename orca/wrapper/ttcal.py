import subprocess
import logging
import os

log = logging.getLogger(__name__)

TTCAL_EXEC = '/opt/astro/mwe/bin/ttcal-0.3.0'


def peel_with_ttcal(ms: str, sources: str):
    """
    Use TTCal to peel sources.
    :param ms: Path to the measurement set.
    :param sources: Path to the sources.json file.
    :return:
    """
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')
    proc = subprocess.Popen([TTCAL_EXEC, 'peel', ms, sources, '--beam', 'sine', '--maxiter', '50',
                             '--tolerance', '1e-4', '--minuvw', '10'], env=new_env,
                            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode is not 0:
            logging.error(f'Error in TTCal: {stderrdata.decode()}')
            logging.info(f'stdout is {stdoutdata.decode()}')
            raise Exception('Error in TTCal.')
    finally:
        proc.terminate()
    return ms
