import subprocess
import  logging
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
    try:
        subprocess.check_call([TTCAL_EXEC, 'peel', ms, sources, '--beam', 'sine', '--maxiter', '50',
                               '--tolerance', '1e-4', '--minuvw', '10'], stderr=subprocess.STDOUT)
        return ms
    except subprocess.CalledProcessError as e:
        log.error(e)
