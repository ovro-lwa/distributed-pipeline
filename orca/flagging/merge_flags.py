"""
Copy from Marin Anderson 3/8/2019
"""

import casacore.tables as pt
import numpy as np

def merge_flags(ms1: str, ms2: str):
    t_prev = pt.table(ms1, readonly=False)
    t = pt.table(ms2, readonly=False)

    flagcol1 = t.getcol('FLAG')
    flagcol2 = t.getcol('FLAG')
    flagcol = flagcol1 | flagcol2
    t.putcol('FLAG', flagcol)
    t_prev.putcol('FLAG', flagcol)
    t_prev.close()
    t.close()


def write_to_flag_column(ms: str, flag_npy: str, create_corrected_data_column: bool = False):
    with pt.table(ms, readonly=False) as t:
        flagcol = np.load(flag_npy)
        assert flagcol.shape == t.getcol('FLAG').shape, 'Flag file and measurement set have different shapes'
        t.putcol('FLAG', flagcol)
        if create_corrected_data_column:
            # Copied from https://github.com/casacore/python-casacore/blob/master/casacore/tables/msutil.py#L48
            dminfo = t.getdminfo("DATA")
            cdesc = t.getcoldesc('DATA')
            dminfo['NAME'] = 'correcteddata'
            cdesc['comment'] = 'The corrected data column'
            t.addcols(pt.maketabdesc(pt.makecoldesc('CORRECTED_DATA', cdesc)), dminfo)
            # For the sake of making TTCal work.
            t.putcol('CORRECTED_DATA', 'DATA')
