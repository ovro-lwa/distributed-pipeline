"""
Copy from Marin Anderson 3/8/2019
"""

import casacore.tables as pt
import numpy as np

DATA = 'DATA'
CORRECTED_DATA = 'CORRECTED_DATA'


def merge_flags(ms1: str, ms2: str):
    with pt.table(ms1, readonly=False) as t_prev:
        with pt.table(ms2, readonly=False) as t:
            flagcol1 = t_prev.getcol('FLAG')
            flagcol2 = t.getcol('FLAG')
            flagcol = flagcol1 | flagcol2
            t.putcol('FLAG', flagcol)
            t_prev.putcol('FLAG', flagcol)
    return ms1, ms2


def write_to_flag_column(ms: str, flag_npy: str):
    with pt.table(ms, readonly=False) as t:
        flagcol = np.load(flag_npy)
        assert flagcol.shape == t.getcol('FLAG').shape, 'Flag file and measurement set have different shapes'
        t.putcol('FLAG', flagcol | t.getcol('FLAG'))
    return ms


def flag_bls(msfile: str, blfile: str):
    """
    Input: msfile, .bl file
    Applies baseline flags to FLAG column.
    """
    with pt.table(msfile, readonly=False) as t:
        flagcol = t.getcol('FLAG')  # flagcol.shape = (N*(N-1)/2 + N)*Nspw*Nints,Nchans,Ncorrs
        Nants   = t.getcol('ANTENNA1')[-1] + 1
        Nbls    = int((Nants*(Nants-1)/2.) + Nants)
        if not (flagcol.shape[0] >= Nbls):
            raise ValueError(f'Unexpected number of visibilities in flagcol {flagcol.shape}')
        Nspw    = int(flagcol.shape[0]/Nbls)
        Nchans  = flagcol.shape[1]
        Ncorrs  = flagcol.shape[2]
        # make the correlation matrix
        flagmat = np.zeros((Nants,Nants,Nspw,Nchans,Ncorrs))
        tiuinds = np.triu_indices(Nants)
        # put the FLAG column into the correlation matrix
        flagmat[tiuinds] = flagcol.reshape(Nbls,Nspw,Nchans,Ncorrs)
        # read in baseline flags
        ant1,ant2 = np.genfromtxt(blfile,delimiter='&',unpack=True,dtype=int)
        # flag the correlation matrix
        flagmat[(ant1,ant2)] = 1
        # reshape correlation matrix into FLAG column
        newflagcol = flagmat[tiuinds].reshape(Nbls*Nspw,Nchans,Ncorrs)
        #
        t.putcol('FLAG',newflagcol)
    return msfile