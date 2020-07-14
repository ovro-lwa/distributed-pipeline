#!/usr/bin/env python

"""
Copy from Marin Anderson 3/8/2019
"""
from __future__ import division
import numpy as np
import casacore.tables as pt
import os,argparse
import numpy.ma as ma
import logging
from scipy.ndimage import filters


def flag_bad_chans(msfile, band, usedatacol=False, generate_plot=False, apply_flag=False, crosshand=False, uvcut_m: float = None):
    """
    Input: msfile
    Finds remaining bad channels and flags those in the measurement set. Also writes out text file that lists
    flags that were applied.
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

        maxamps              = np.max(datacolamp_mask, axis=0)
        meanamps             = np.mean(datacolamp_mask, axis=0)
        maxamps_medfilt      = filters.median_filter(maxamps, size=10)
        maxamps_norm         = maxamps / maxamps_medfilt
        maxamps_norm_stdfilt = filters.generic_filter(maxamps_norm, np.std, size=25)
        maxamps_lower        = 1 - 10*np.min(maxamps_norm_stdfilt, axis=0)
        maxamps_upper        = 1 + 10*np.min(maxamps_norm_stdfilt, axis=0)
        meanamps_stdfilt     = filters.generic_filter(meanamps, np.std, size=25)

        flaglist = np.where( (maxamps_norm < maxamps_lower) | (maxamps_norm > maxamps_upper) | 
                             (meanamps > np.median(meanamps, axis=0)+100*np.min(meanamps_stdfilt, axis=0)) )
        if not crosshand:
            flaglist = np.unique(flaglist[0][np.where( (flaglist[1] == 0) | (flaglist[1] == 3) )])
        else:
            flaglist = np.unique(flaglist[0])
        #import pylab
        #pylab.ion()
        #pylab.plot(skewxx_norm, '.', color='Blue')
        #pylab.plot(skewyy_norm, '.', color='Green')
        #pylab.hlines(skewvalxx, 0, 108, color='blue')
        #pylab.hlines(skewval2xx, 0, 108, color='blue')
        #pylab.hlines(skewvalyy, 0, 108, color='green')
        #pylab.hlines(skewval2yy, 0, 108, color='green')
        #pylab.plot(flaglist[0],skewxx_norm[flaglist], '.', color='Red')
        #pylab.plot(flaglist[0], skewyy_norm[flaglist], '.', color='Red')
        #pylab.grid('on')
        #pylab.figure()
        #pylab.plot(meanxx, '.', color='Blue')
        #pylab.plot(meanyy, '.', color='Green')
        #pylab.hlines(np.median(meanxx)+100*np.min(meanxx_stdfilt), 0, 108, color='Blue')
        #pylab.hlines(np.median(meanyy)+100*np.min(meanyy_stdfilt), 0, 108, color='Green')
        #pylab.grid('on')
        #import pdb
        #pdb.set_trace()

        #################################################
        #this is for testing purposes only
        #generate plot of visibilities for quick check of how well flagging performed
        if generate_plot:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,10))
            chans = np.arange(0,109)
            for chan in chans:
                if chan not in flaglist:
                    chanpts = np.zeros(len(datacolxxamp_mask[:,chan]))+chan
                    plt.plot(datacolxxamp_mask[:,chan],chanpts, '.', color='Blue', markersize=0.5)
                    plt.plot(datacolyyamp_mask[:,chan],chanpts, '.', color='Green', markersize=0.5)
            plt.ylim([0,108])
            plt.ylabel('channel')
            plt.xlabel('Amp')
            plt.gca().invert_yaxis()
            plotfile = os.path.splitext(os.path.abspath(msfile))[0]+'.png'
            plt.savefig(plotfile)

        ################################################

        logging.info('Flaglist size is %i' % flaglist.size)
        if flaglist.size > 0:
            # turn flaglist into text file of channel flags
            textfile = os.path.splitext(os.path.abspath(msfile))[0]+'.chans'
            chans    = np.arange(0,109)
            chanlist = chans[flaglist]
            with open(textfile, 'w') as f:
                for chan in chanlist:
                    f.write('%02d:%03d\n' % (np.int(band),chan))

            # write flags into FLAG column
            if apply_flag:
                logging.info('Applying the changes to the measurement set.')
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
    parser.add_argument('--uvcut_m', action='store', default=None, help='Only use visibilities greater than {uvcut_m} in meters when determining channel flags. Default is None.')
    args = parser.parse_args()
    flag_bad_chans(args.msfile, args.band, usedatacol=args.usedatacol,
                   generate_plot=args.plot, apply_flag=args.apply_flag,
                   crosshand=args.crosshand, uvcut_m=args.uvcut_m)


if __name__ == '__main__':
    main()
