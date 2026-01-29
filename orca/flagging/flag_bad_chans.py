"""Channel flagging based on visibility amplitude statistics.

Identifies bad frequency channels by analyzing visibility amplitude
and SNR distributions. Uses median filtering to detect outliers.

Originally adapted from Marin Anderson's code (3/8/2019).

Functions
---------
flag_bad_chans
    Identify and optionally apply channel flags.
"""
#!/usr/bin/env python

"""
Copy from Marin Anderson 3/8/2019
"""
from __future__ import division
from typing import Optional
import numpy as np
import casacore.tables as pt
import os,argparse
import numpy.ma as ma
import logging
from scipy.ndimage import filters

from orca.configmanager import telescope as tele

logger = logging.getLogger(__name__)

def flag_bad_chans(msfile: str, band: str, usedatacol=False, generate_plot=False, apply_flag=False, crosshand=False,
                   uvcut_m: Optional[float] = None):
    """Flag bad channels.
    Finds remaining bad channels and flags those in the measurement set. Also writes out text file that lists
    flags that were applied.

    Args:
        msfile: measurement set to flag.
        band: Subband number, must be convertable to integer.
        usedatacol: If True, uses DATA column, else use CORRECTED_DATA.
        generate_plot: generate a plot or not.
        apply_flag: Whether to apply the flags.
        crosshand: If true, it will use the XY and YX correlations when determining flags.
            Otherwise, it will ignore the flags that are in flaglist[:,1] and flaglist[:,2].
        uvcut_m:  uvcut in meters before doing thresholding to suppress short baseline flux

    """
    with pt.table(msfile, readonly=False) as t:
        tcross  = t.query('ANTENNA1!=ANTENNA2')
        if usedatacol:
            datacol = tcross.getcol('DATA')
        else:
            datacol = tcross.getcol('CORRECTED_DATA')
        flagcol = tcross.getcol('FLAG')
        
        if uvcut_m:
            uvw          = tcross.getcol('UVW')
            uvdist       = np.sqrt( uvw[:,0]**2. + uvw[:,1]**2. )
            indsbyuvdist = np.where(uvdist > uvcut_m)
            datacol      = datacol[indsbyuvdist]
            flagcol      = flagcol[indsbyuvdist]

        datacolamp      = np.abs(datacol)
        datacolamp_mask = ma.masked_array(datacolamp, mask=flagcol, fill_value=np.nan)

        maxamps              = np.ma.max(datacolamp_mask, axis=0)
        meanamps             = np.ma.mean(datacolamp_mask, axis=0)
        maxamps_medfilt      = filters.median_filter(maxamps, size=(25,1)) #10,1))
        maxamps_norm         = maxamps / maxamps_medfilt
        maxamps_norm_stdfilt = filters.generic_filter(maxamps_norm, np.std, size=(25,1))
        threshold_vec        = np.array([10,6,6,10])
        maxamps_lower        = 1 - threshold_vec*np.ma.min(maxamps_norm_stdfilt, axis=0)
        maxamps_upper        = 1 + threshold_vec*np.ma.min(maxamps_norm_stdfilt, axis=0)
        meanamps_stdfilt     = filters.generic_filter(meanamps, np.std, size=(25,1))

        flaglist = np.where( (maxamps_norm < maxamps_lower) | (maxamps_norm > maxamps_upper) | 
                             (meanamps > np.ma.median(meanamps, axis=0)+100*np.ma.min(meanamps_stdfilt, axis=0)) )
        if not crosshand:
            flaglist = np.unique(flaglist[0][np.where( (flaglist[1] == 0) | (flaglist[1] == 3) )])
        else:
            flaglist = np.unique(flaglist[0])
        #################################################
        #this is for testing purposes only
        #generate plot of visibilities for quick check of how well flagging performed
        if generate_plot:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,10))
            chans = np.arange(0, tele.n_chan)
            for chan in chans:
                if chan not in flaglist:
                    chanpts = np.zeros(len(datacolamp_mask[:,chan,0]))+chan
                    plt.plot(datacolamp_mask[:,chan,0],chanpts, '.', color='Blue', markersize=0.5)
                    plt.plot(datacolamp_mask[:,chan,3],chanpts, '.', color='Green', markersize=0.5)
            plt.ylim([0, tele.n_chan - 1])
            plt.ylabel('channel')
            plt.xlabel('Amp')
            plt.gca().invert_yaxis()
            plotfile = os.path.splitext(os.path.abspath(msfile))[0]+'.png'
            plt.savefig(plotfile)

        ################################################

        logger.info('Flaglist size is %i' % flaglist.size)
        if flaglist.size > 0:
            # turn flaglist into text file of channel flags
            textfile = os.path.splitext(os.path.abspath(msfile))[0]+'.chans'
            chans    = np.arange(0,tele.n_chan)
            chanlist = chans[flaglist]
            with open(textfile, 'w') as f:
                for chan in chanlist:
                    f.write('%02d:%03d\n' % (int(band),chan))

            # write flags into FLAG column
            if apply_flag:
                flagcol_altered = t.getcol('FLAG')
                flagcol_altered[:,flaglist,:] = 1
                t.putcol('FLAG', flagcol_altered)
    return msfile


def main():
    parser = argparse.ArgumentParser(description="Flag bad channels and write out list of channels that were \
                                                  flagged into text file of same name as ms. MUST BE RUN ON \
                                                  SINGLE SUBBAND MS.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("band", help="Subband number.")
    parser.add_argument("--usedatacol", action="store_true", default=False, help="Grab DATA column, not CORRECTED_DATA.")
    parser.add_argument('--plot', action='store_true', default=False, help='Generate plot of amp vs channel.')
    parser.add_argument('--apply-flag', action='store_true', default=False, help='Apply flags to measurement set.')
    parser.add_argument('--crosshand', action='store_true', default=False, help='Use the cross-hand visibilities also.')
    parser.add_argument('--uvcut_m', action='store', type=float, default=None, help='Only use visibilities greater than {uvcut_m} in meters when determining channel flags. Default is None.')
    args = parser.parse_args()
    flag_bad_chans(args.msfile, args.band, usedatacol=args.usedatacol,
                   generate_plot=args.plot, apply_flag=args.apply_flag,
                   crosshand=args.crosshand, uvcut_m=args.uvcut_m)


if __name__ == '__main__':
    main()
