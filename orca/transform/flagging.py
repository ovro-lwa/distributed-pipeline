"""Flagging transforms for measurement sets.

Provides high-level flagging operations that combine detection and application,
including AOFlagger integration and autocorrelation-based antenna flagging.

Functions
---------
flag_with_aoflagger
    Run AOFlagger RFI detection on a measurement set.
flag_ants
    Flag specified antennas in a measurement set.
flag_ant_chan_from_autocorr
    Flag antennas and channels based on autocorrelation anomalies.
flag_on_autocorr
    Identify and flag bad antennas from autocorrelation statistics.
identify_bad_ants
    Identify bad antennas without applying flags.
save_flag_metadata
    Save flag statistics in a compact binary format.
"""
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

import os
from casatools import table

log = logging.getLogger(__name__)

FLAG_COUNT_FACTOR = 10


def flag_with_aoflagger(ms: str, strategy: str='/opt/share/aoflagger/strategies/nenufar-lite.lua',
                        in_memory : bool =False, n_threads : int = 5) -> str:
    """Run AOFlagger RFI detection on a measurement set.

    Args:
        ms: Path to the measurement set.
        strategy: Path to the AOFlagger Lua strategy file.
        in_memory: If True, load data into memory for processing.
        n_threads: Number of threads for AOFlagger.

    Returns:
        Path to the flagged measurement set.

    Raises:
        RuntimeError: If AOFlagger returns a non-zero exit code.
    """
    # TODO use the API
    arg_list = [execs.aoflagger, '-strategy', strategy, '-j', str(n_threads)]
    if not in_memory:
        arg_list.append('-direct-read')
    arg_list.append(ms)
    proc = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            if stderrdata:
                log.error(f'Error in aoflagger: {stderrdata.decode()}')
            if stdoutdata:
                log.error(stdoutdata.decode())
            raise RuntimeError(f'Error in aoflagger for {ms}.')
    finally:
        proc.terminate()
    return ms


def flag_ants(ms: str, ants: List[int]) -> str:
    """Flag all visibilities involving specified antennas.

    Args:
        ms: Path to the measurement set.
        ants: List of antenna indices (0-indexed) to flag.

    Returns:
        Path to the modified measurement set.
    """
    if len(ants) > 0 :
        ants_str = '(' + ', '.join([str(ant) for ant in ants]) + ')'
        tables.taql('UPDATE %s SET FLAG = True WHERE ANTENNA1 IN %s OR ANTENNA2 IN %s' % (ms, ants_str, ants_str))
    return ms


def flag_ant_chan_from_autocorr(ms: str, threshold: float=5.0) -> str:
    """Flag antennas and channels based on autocorrelation anomalies.

    Args:
        ms: Path to the measurement set.
        threshold: Sigma threshold for outlier detection.

    Returns:
        Path to the flagged measurement set.

    Note:
        Currently only works on single spectral window data.
    """
    return ms


def flag_on_autocorr(ms, date: Optional[datetime.date] = None, thresh: float=7.0, column='DATA') -> str:
    """Identify and flag bad antennas from autocorrelation statistics.

    Optionally loads a priori bad antenna list for the given date before
    performing autocorrelation-based detection.

    Args:
        ms: Path to the measurement set.
        date: Observation date for loading a priori bad antennas.
        thresh: Sigma threshold for flagging (default 7.0).
        column: Data column to analyze ('DATA' or 'CORRECTED_DATA').

    Returns:
        Path to the flagged measurement set.
    """
    if date:
        a_priori_bad_ants = flagutils.get_bad_ants(date)
        flag_ants(ms, a_priori_bad_ants)
    with tables.table(ms, readonly=False, ack=False) as t:
        bad_ants = identify_bad_ants(t, thresh, column)
    return flag_ants(ms, bad_ants)


def identify_bad_ants(t: tables.table, thresh: float=7, column='DATA') -> List[int]:
    """Identify bad antennas from autocorrelation statistics.

    Compares each antenna's autocorrelation bandpass to a median template
    for core and outrigger antennas separately. Antennas with normalized
    deviations exceeding the threshold are flagged.

    Args:
        t: Open casacore table object for the measurement set.
        thresh: Sigma threshold for flagging (default 7.0).
        column: Data column to analyze ('DATA' or 'CORRECTED_DATA').

    Returns:
        List of antenna indices identified as bad.

    Raises:
        AssertionError: If data contains multiple timestamps or spectral windows.
    """
    tautos = t.query('ANTENNA1=ANTENNA2')
    autos_corrected : np.ndarray = np.abs(tautos.getcol(column))[...,(0,3)]
    autos_flags : np.ndarray = tautos.getcol('FLAG')[...,(0,3)]
    autos_antnums : np.ndarray = tautos.getcol('ANTENNA1')

    n_ants = tele.n_ant
    assert autos_corrected.shape[0] == n_ants, 'Doing multiple timestamp or multiple spws is too complicated for now.'

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
    """Mask rows with high flagged channel fractions.

    Internal helper function for autocorrelation analysis.
    """
    bad_row_mask = np.zeros(autos_corrected.shape[0], dtype=bool)
    for pol in range(2):
        flagged_chan_count = np.sum(autos_flags[..., pol], axis=1)
        median_flag_count = np.median(flagged_chan_count)
        if median_flag_count < n_chans // FLAG_COUNT_FACTOR:
            bad_row_mask[flagged_chan_count > FLAG_COUNT_FACTOR * median_flag_count] = True
    return bad_row_mask


def _calculate_avg_bandpass(autos_corrected, bad_row_mask, n_ants, n_ints, n_chans) -> np.ndarray:
    """Calculate average bandpass from autocorrelation data.

    Internal helper function for multi-integration autocorrelation analysis.
    """
    bp = np.zeros((n_ants, n_chans, 2))
    autos_corrected[bad_row_mask] = np.nan
    return bp

def save_flag_metadata(ms: str, output_dir: str = '/lustre/pipeline/slow-averaged/') -> str:
    """Save FLAG column in a compact bit-packed binary format.

    Creates a binary file containing the packed flag data, which can be
    unpacked later using numpy.unpackbits().

    Args:
        ms: Path to the measurement set.
        output_dir: Directory for output file.

    Returns:
        Path to the measurement set (unchanged).
    """
    base_name = os.path.splitext(os.path.basename(ms))[0]
    output_file = os.path.join(output_dir, f"{base_name}_flagmeta.bin")

    with tables.table(ms, ack=False, readonly=True) as tb:
        flags = tb.getcol('FLAG')

    bit_packed = np.packbits(flags.flatten().astype(np.uint8))
    bit_packed.tofile(output_file)

    log.info(f"Flag metadata saved in packed binary format to '{output_file}'.")

    return ms
