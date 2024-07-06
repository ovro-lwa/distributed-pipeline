from typing import List, Optional

import subprocess
import logging
from matplotlib.pyplot import flag
import numpy as np
import datetime

from casacore import tables

from orca.configmanager import execs, telescope as tele
from orca.utils.maths import core_outrigger_slices
from orca.utils import flagutils

log = logging.getLogger(__name__)

FLAG_COUNT_FACTOR = 10

def flag_with_aoflagger(ms: str, strategy: str='/opt/share/aoflagger/strategies/nenufar-lite.lua') -> str:
    # TODO use the API
    arg_list = [execs.aoflagger, '-strategy', strategy, ms]
    proc = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            if stderrdata:
                log.error(f'Error in aoflagger: {stderrdata.decode()}')
            raise RuntimeError(f'Error in aoflagger for {ms}.')
    finally:
        proc.terminate()
    return ms


def flag_ants(ms: str, ants: List[int]) -> str:
    """
    Input: msfile, list of antennas to flag
    Flags the antennas in the list.
    """
    if len(ants) > 0 :
        tables.taql('UPDATE %s SET FLAG = True WHERE ANTENNA1 IN %s OR ANTENNA2 IN %s' % (ms, tuple(ants), tuple(ants)))
    return ms


def flag_ant_chan_from_autocorr(ms: str, threshold: float=5.0) -> str:
    """
    Flags antennas and channels based on autocorrelation data. Only works on single spw data.
    """
    return ms


def flag_on_autocorr(ms, date: Optional[datetime.date] = None, thresh: float=7.0, column='DATA') -> str:
    if date:
        a_priori_bad_ants = flagutils.get_bad_ants(date)
        flag_ants(ms, a_priori_bad_ants)
    with tables.table(ms, readonly=False) as t:
        bad_ants = identify_bad_ants(t, thresh, column)
        return flag_ants(ms, bad_ants)


def identify_bad_ants(t: tables.table, thresh: float=7, column='DATA') -> List[int]:
    tautos = t.query('ANTENNA1=ANTENNA2')
    autos_corrected : np.ndarray = np.abs(tautos.getcol(column))[...,(0,3)]
    autos_flags : np.ndarray = tautos.getcol('FLAG')[...,(0,3)]
    autos_antnums : np.ndarray = tautos.getcol('ANTENNA1')

    n_ants = tele.n_ant
    assert autos_corrected.shape[0] == n_ants, 'Doing multiple timestamp or multiple spws is too complicated for now.'

    """
    One day I may get back to this...

    assert autos_corrected.shape == (n_ants*n_ints, n_chans, 2)
    assert autos_flags.shape == (n_ants*n_ints, n_chans, 2)
    bad_row_mask = _mask_bad_rows(autos_corrected, autos_flags, n_chans)
    bandpasses = autos_corrected if n_ints == 1 else _calculate_avg_bandpass(autos_corrected, bad_row_mask, n_ants, n_ints, n_chans)
    """
    bandpasses = autos_corrected
    is_core, is_exp = core_outrigger_slices(autos_antnums, tele.outriggers)

    core_template = np.median(bandpasses[is_core], axis=0)
    exp_template = np.median(bandpasses[is_exp], axis=0)
    diff = np.zeros_like(bandpasses)
    diff[is_core] = np.abs(bandpasses[is_core] - core_template) / core_template
    diff[is_exp] = np.abs(bandpasses[is_exp] - exp_template) / exp_template
    diff = diff * (1 - autos_flags)
    sum_diff = np.mean(diff, axis=1)
    mad_core = np.median(sum_diff[is_core], axis=0)
    mad_exp = np.median(sum_diff[is_exp], axis=0)
    scores = sum_diff / mad_core
    scores[is_exp] = sum_diff[is_exp] / mad_exp
    return autos_antnums[np.any(scores > thresh, axis=1)]


def _mask_bad_rows(autos_corrected, autos_flags, n_chans):
    bad_row_mask = np.zeros(autos_corrected.shape[0], dtype=bool)
    # mask antenna/time data points with high flagged fractions
    for pol in range(2):
        flagged_chan_count = np.sum(autos_flags[..., pol], axis=1)
        median_flag_count = np.median(flagged_chan_count)
        # only do this if only a small number of channels are flagged
        if median_flag_count < n_chans // FLAG_COUNT_FACTOR:
            bad_row_mask[flagged_chan_count > FLAG_COUNT_FACTOR * median_flag_count] = True
    return bad_row_mask


def _calculate_avg_bandpass(autos_corrected, bad_row_mask, n_ants, n_ints, n_chans) -> np.ndarray:
    bp = np.zeros((n_ants, n_chans, 2))
    autos_corrected[bad_row_mask] = np.nan
    return bp

