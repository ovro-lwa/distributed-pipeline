import logging
import subprocess

TTCAL_EXEC = '/opt/astro/mwe/bin/ttcal-0.3.0'


def peel_with_ttcal(ms: str, sources: str):
    """
    Use TTCal to peel sources. Assumes that the input measurement set only has DATA column and writes to the
    CORRECTED_DATA column.
    :param ms:
    :param sources:
    :return:
    """
    subprocess.check_output([TTCAL_EXEC, 'peel', ms, sources, '--beam', 'sine', '--maxiter', '50',
                             '--tolerance', '1e-4', '--minuvw', '10'])
