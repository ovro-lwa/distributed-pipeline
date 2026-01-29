"""Baseline flagging based on cross-hand visibility statistics.

Identifies bad baselines by analyzing cross-hand (XY, YX) correlation
amplitudes. Baselines with excessive cross-hand power are flagged.

Functions
---------
flag_bad_bls
    Identify and write bad baselines to a text file.
"""
#!/usr/bin/env python

from __future__ import division
from typing import Optional
import numpy as np
import casacore.tables as pt
import os,argparse,sys
import numpy.ma as ma


def flag_bad_bls(msfile: str, usedatacol: bool = False) -> Optional[str]:
    """Flag bad baselines using the visibilities.

    Identifies baselines with anomalously high cross-hand (XY, YX) correlation
    amplitudes and writes them to a text file. Does NOT apply flags directly
    to the measurement set.

    Args:
        msfile: Path to the measurement set.
        usedatacol: If True, use DATA column; otherwise use CORRECTED_DATA.

    Returns:
        Path to the text file listing baselines to flag, or None if no
        bad baselines were found. Format is one baseline per line as 'ant1&ant2'.
    """
    with pt.table(msfile, readonly=True) as t:
        tcross = t.query('ANTENNA1!=ANTENNA2')
        if usedatacol:
            datacol = tcross.getcol('DATA')  # data.shape = (32640, 109, 4)
        else:
            datacol = tcross.getcol('CORRECTED_DATA')
        flagcol = tcross.getcol('FLAG')
        ant1col = tcross.getcol('ANTENNA1')
        ant2col = tcross.getcol('ANTENNA2')
        datacolamp = np.abs(datacol)
        datacolamp_mask = ma.masked_array(datacolamp, mask=flagcol, fill_value=np.nan)

        cutoffval = np.ma.mean(datacolamp_mask[:,:,1:3]) + 3*np.ma.std(datacolamp_mask[:,:,1:3])
        tmp = np.where((datacolamp_mask.filled()[:,:,1] > cutoffval) | (datacolamp_mask.filled()[:,:,2] > cutoffval))
        tmparr = np.zeros((datacolamp.shape[0],datacolamp.shape[1]))
        tmparr[tmp] = 1
        tmpflag = np.sum(tmparr,axis=1)/109
        flaglist = np.where(tmpflag > 0.25)
        
        if flaglist[0].size > 0:
            # turn flaglist into text file of baseline flags
            textfile = os.path.splitext(os.path.abspath(msfile))[0]+'.bl'
            with open(textfile, 'w') as f:
                for flagind in flaglist[0]:
                    f.write('%d&%d\n' % (ant1col[flagind],ant2col[flagind]))
            return textfile
        else:
            return None


def main():
    parser = argparse.ArgumentParser(description="Identify bad baselines and write out list of baselines to flag.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("--usedatacol", action="store_true", default=False, help="Grab DATA column, else CORRECTED_DATA.")
    args = parser.parse_args()
    flag_bad_bls(args.msfile, usedatacol=args.usedatacol)


if __name__ == '__main__':
    main()
    