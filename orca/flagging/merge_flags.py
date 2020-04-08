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


def write_to_flag_column(ms: str, flag_npy: str):
    with pt.table(ms, readonly=False) as t:
        flagcol = np.load(flag_npy)
        assert flagcol.shape == t.getcol('FLAG').shape, 'Flag file and measurement set have different shapes'
        t.putcol('FLAG', flagcol | t.getcol('FLAG'))
