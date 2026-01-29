"""Low-level flag manipulation operations for measurement sets.

Provides functions for directly manipulating FLAG columns in CASA
measurement sets, including flagging antennas, baselines, merging
flags between observations, and I/O with numpy arrays.

Originally adapted from Marin Anderson's code (3/8/2019).

Functions
---------
flag_ants
    Flag all visibilities involving specified antennas.
merge_flags
    OR-merge flags between two measurement sets.
merge_group_flags
    OR-merge flags across a group of measurement sets.
write_to_flag_column
    Apply flags from a numpy array to a measurement set.
save_to_flag_npy
    Save the FLAG column to a numpy file.
flag_bls
    Apply baseline flags from a text file.
"""
"""
Copy from Marin Anderson 3/8/2019
"""

import casacore.tables as pt
import numpy as np

from typing import Tuple, List

DATA = 'DATA'
CORRECTED_DATA = 'CORRECTED_DATA'


def flag_ants(ms: str, ants: List[int]) -> str:
    """Flag all visibilities involving specified antennas.

    Args:
        ms: Path to the measurement set.
        ants: List of antenna indices (0-indexed) to flag.

    Returns:
        Path to the modified measurement set.
    """
    if len(ants) > 0 :
        pt.taql('UPDATE %s SET FLAG = True WHERE ANTENNA1 IN %s OR ANTENNA2 IN %s' % (ms, tuple(ants), tuple(ants)))
    return ms
    

def merge_flags(ms1: str, ms2: str) -> Tuple[str, str]:
    """OR-merge flags between two measurement sets.

    Updates both measurement sets so their FLAG columns contain the
    logical OR of the original flags from both files.

    Args:
        ms1: Path to the first measurement set.
        ms2: Path to the second measurement set.

    Returns:
        Tuple of (ms1, ms2) paths.
    """
    with pt.table(ms1, readonly=False, ack=False) as t_prev, pt.table(ms2, readonly=False, ack=False) as t:
            flagcol1 = t_prev.getcol('FLAG')
            flagcol2 = t.getcol('FLAG')
            flagcol = flagcol1 | flagcol2
            t.putcol('FLAG', flagcol)
            t_prev.putcol('FLAG', flagcol)
    return ms1, ms2


def merge_group_flags(ms_list: List[str]) -> List[str]:
    """OR-merge flags across a group of measurement sets.

    Updates all measurement sets so they share the same merged FLAG column,
    containing the logical OR of flags from all input files.

    Args:
        ms_list: List of measurement set paths.

    Returns:
        The input list of paths (all now updated).
    """
    with pt.table(ms_list[0], readonly=True, ack=False) as table:
        merged_flags = table.getcol('FLAG')
    for ms in ms_list[1:]:
        with pt.table(ms, readonly=True, ack=False) as tt:
            merged_flags = merged_flags | tt.getcol('FLAG')
    for ms in ms_list:
        with pt.table(ms, readonly=False, ack=False) as tt:
            tt.putcol('FLAG', merged_flags)
    return ms_list


def write_to_flag_column(ms: str, flag_npy: str) -> str:
    """Apply flags from a numpy array to a measurement set.

    The numpy array is OR-merged with existing flags.

    Args:
        ms: Path to the measurement set.
        flag_npy: Path to the numpy file containing boolean flag array.

    Returns:
        Path to the modified measurement set.

    Raises:
        AssertionError: If the flag array shape doesn't match the MS.
    """
    with pt.table(ms, readonly=False, ack=False) as t:
        flagcol = np.load(flag_npy)
        assert flagcol.shape == t.getcol('FLAG').shape, 'Flag file and measurement set have different shapes'
        t.putcol('FLAG', flagcol | t.getcol('FLAG'))
    return ms


def save_to_flag_npy(ms: str, flag_npy: str) -> str:
    """Save the FLAG column from a measurement set to a numpy file.

    Args:
        ms: Path to the measurement set.
        flag_npy: Output path for the numpy file.

    Returns:
        Path to the saved numpy file.
    """
    with pt.table(ms, ack=False) as t:
        flagcol = t.getcol('FLAG')
    np.save(flag_npy, flagcol)
    return flag_npy


def flag_bls(msfile: str, blfile: str) -> str:
    """Apply baseline flags from a text file to a measurement set.

    Args:
        msfile: Path to the measurement set.
        blfile: Path to the baseline flag file. Format is one baseline
            per line as 'ant1&ant2' (0-indexed).

    Returns:
        Path to the modified measurement set.

    Raises:
        ValueError: If the number of visibilities is unexpected.
    """
    with pt.table(msfile, readonly=False, ack=False) as t:
        flagcol = t.getcol('FLAG')  # flagcol.shape = (N*(N-1)/2 + N)*Nspw*Nints,Nchans,Ncorrs
        Nants = t.getcol('ANTENNA1')[-1] + 1
        Nbls = int((Nants*(Nants-1)/2.) + Nants)
        if not (flagcol.shape[0] >= Nbls):
            raise ValueError(f'Unexpected number of visibilities in flagcol {flagcol.shape}')
        Nspw = int(flagcol.shape[0]/Nbls)
        Nchans = flagcol.shape[1]
        Ncorrs = flagcol.shape[2]
        # make the correlation matrix
        flagmat = np.zeros((Nants,Nants,Nspw,Nchans,Ncorrs),dtype=bool)
        tiuinds = np.triu_indices(Nants)
        # put the FLAG column into the correlation matrix
        flagmat[tiuinds] = flagcol.reshape(Nspw,Nbls,Nchans,Ncorrs).transpose(1,0,2,3)
        # read in baseline flags
        ant1,ant2 = np.genfromtxt(blfile,delimiter='&',unpack=True,dtype=int)
        # flag the correlation matrix
        flagmat[(ant1,ant2)] = 1
        # reshape correlation matrix into FLAG column
        newflagcol = flagmat[tiuinds].transpose(1,0,2,3).reshape(Nbls*Nspw,Nchans,Ncorrs)
        #
        t.putcol('FLAG',newflagcol)
    return msfile
