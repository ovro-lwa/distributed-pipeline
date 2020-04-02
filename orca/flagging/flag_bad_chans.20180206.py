#!/usr/bin/env python

from __future__ import division
import numpy as np
import pyrap.tables as pt
import os,argparse
import numpy.ma as ma
from scipy.stats import skew
from scipy.ndimage import filters

def flag_bad_chans(msfile, band, usedatacol=False):
    """
    Input: msfile
    Finds remaining bad channels and flags those in the measurement set. Also writes out text file that lists
    flags that were applied.
    """
    t       = pt.table(msfile, readonly=False)
    tcross  = t.query('ANTENNA1!=ANTENNA2')
    if usedatacol:
        datacol = tcross.getcol('DATA')
    else:
        datacol = tcross.getcol('CORRECTED_DATA')
    flagcol = tcross.getcol('FLAG')
    
    datacolxx = datacol[:,:,0]
    datacolyy = datacol[:,:,3]
    datacolxy = datacol[:,:,1]
    datacolyx = datacol[:,:,2]

    datacolxxamp = np.sqrt( np.real(datacolxx)**2. + np.imag(datacolxx)**2. )
    datacolyyamp = np.sqrt( np.real(datacolyy)**2. + np.imag(datacolyy)**2. )
    datacolxyamp = np.sqrt( np.real(datacolxy)**2. + np.imag(datacolxy)**2. )
    datacolyxamp = np.sqrt( np.real(datacolyx)**2. + np.imag(datacolyx)**2. )

    #flagarr = flagcol[:,:,0] | flagcol[:,:,3]   # probably unnecessary since flags are never pol-specific,
                                                 # but doing this just in cases
    flagarr = flagcol[:,:,0] | flagcol[:,:,1] | flagcol[:,:,2] | flagcol[:,:,3]

    datacolxxamp_mask = ma.masked_array(datacolxxamp, mask=flagarr, fill_value=np.nan)
    datacolyyamp_mask = ma.masked_array(datacolyyamp, mask=flagarr, fill_value=np.nan)
    datacolxyamp_mask = ma.masked_array(datacolxyamp, mask=flagarr, fill_value=np.nan)
    datacolyxamp_mask = ma.masked_array(datacolyxamp, mask=flagarr, fill_value=np.nan)

    maxxx = np.max(datacolxxamp_mask,axis=0)
    maxyy = np.max(datacolyyamp_mask,axis=0)
    maxxy = np.max(datacolxyamp_mask,axis=0)
    maxyx = np.max(datacolyxamp_mask,axis=0)
    meanxx = np.mean(datacolxxamp_mask,axis=0)
    meanyy = np.mean(datacolyyamp_mask,axis=0)
    meanxy = np.mean(datacolxyamp_mask,axis=0)
    meanyx = np.mean(datacolyxamp_mask,axis=0)

    maxxx_medfilt = filters.median_filter(maxxx,size=10)
    maxyy_medfilt = filters.median_filter(maxyy,size=10)
    maxxy_medfilt = filters.median_filter(maxxy,size=10)
    maxyx_medfilt = filters.median_filter(maxyx,size=10)

    maxxx_norm = maxxx/maxxx_medfilt
    maxyy_norm = maxyy/maxyy_medfilt
    maxxy_norm = maxxy/maxxy_medfilt
    maxyx_norm = maxyx/maxyx_medfilt
    
    maxxx_norm_stdfilt = filters.generic_filter(maxxx_norm, np.std, size=25)
    maxyy_norm_stdfilt = filters.generic_filter(maxyy_norm, np.std, size=25)
    maxxy_norm_stdfilt = filters.generic_filter(maxxy_norm, np.std, size=25)
    maxyx_norm_stdfilt = filters.generic_filter(maxyx_norm, np.std, size=25)
    maxvalxx  = 1 - 10*np.min(maxxx_norm_stdfilt)
    maxval2xx = 1 + 10*np.min(maxxx_norm_stdfilt)
    maxvalyy  = 1 - 10*np.min(maxyy_norm_stdfilt)
    maxval2yy = 1 + 10*np.min(maxyy_norm_stdfilt)
    maxvalxy  = 1 - 6*np.min(maxxy_norm_stdfilt)
    maxval2xy = 1 + 6*np.min(maxxy_norm_stdfilt)
    maxvalyx  = 1 - 6*np.min(maxyx_norm_stdfilt)
    maxval2yx = 1 + 6*np.min(maxyx_norm_stdfilt)
    meanxx_stdfilt = filters.generic_filter(meanxx, np.std, size=25)
    meanyy_stdfilt = filters.generic_filter(meanyy, np.std, size=25)
    meanxy_stdfilt = filters.generic_filter(meanxy, np.std, size=25)
    meanyx_stdfilt = filters.generic_filter(meanyx, np.std, size=25)

    # bad channels tend to have maxness values close to zero or slightly negative, compared to
    # good channels, which have significantly positive maxs, or right-maxed distributions.
    #flaglist = np.where( (maxxx < 1) | (maxyy < 1)  )
    flaglist = np.where( (maxxx_norm < maxvalxx) | (maxyy_norm < maxvalyy) |   \
                         (maxxx_norm > maxval2xx) | (maxyy_norm > maxval2yy) | \
                         (meanxx > np.median(meanxx)+100*np.min(meanxx_stdfilt))  | \
                         (meanyy > np.median(meanyy)+100*np.min(meanyy_stdfilt)) | \
                         (maxxy_norm < maxvalxy) | (maxyx_norm < maxvalyx) | \
                         (maxxy_norm > maxval2xy) | (maxyx_norm > maxval2yx) | \
                         (meanxy > np.median(meanxy)+100*np.min(meanxy_stdfilt)) | \
                         (meanyx > np.median(meanyx)+100*np.min(meanyx_stdfilt)) ) 
    #import pylab
    #pylab.ion()
    #pylab.plot(maxxx_norm, '.', color='Blue')
    #pylab.plot(maxyy_norm, '.', color='Green')
    #pylab.plot(maxxy_norm, '.', color='Orange')
    #pylab.plot(maxyx_norm, '.', color='Magenta')
    #pylab.hlines(maxvalxx, 0, 108, color='blue')
    #pylab.hlines(maxval2xx, 0, 108, color='blue')
    #pylab.hlines(maxvalyy, 0, 108, color='green')
    #pylab.hlines(maxval2yy, 0, 108, color='green')
    #pylab.hlines(maxvalxy, 0, 108, color='orange')
    #pylab.hlines(maxval2xy, 0, 108, color='orange')
    #pylab.hlines(maxvalyx, 0, 108, color='magenta')
    #pylab.hlines(maxval2yx, 0, 108, color='magenta')
    #pylab.plot(flaglist[0], maxxx_norm[flaglist], '.', color='Red')
    #pylab.plot(flaglist[0], maxyy_norm[flaglist], '.', color='Red')
    #pylab.plot(flaglist[0], maxxy_norm[flaglist], '.', color='Red')
    #pylab.plot(flaglist[0], maxyx_norm[flaglist], ',', color='Red')
    #pylab.grid('on')
    #pylab.figure()
    #pylab.plot(meanxx, '.', color='Blue')
    #pylab.plot(meanyy, '.', color='Green')
    #pylab.plot(meanxy, '.', color='Orange')
    #pylab.plot(meanyx, '.', color='Magenta')
    #pylab.hlines(np.median(meanxx)+100*np.min(meanxx_stdfilt), 0, 108, color='Blue')
    #pylab.hlines(np.median(meanyy)+100*np.min(meanyy_stdfilt), 0, 108, color='Green')
    #pylab.hlines(np.median(meanxy)+100*np.min(meanxy_stdfilt), 0, 108, color='Orange')
    #pylab.hlines(np.median(meanyx)+100*np.min(meanyx_stdfilt), 0, 108, color='Magenta')
    #pylab.grid('on')
    #import pdb
    #pdb.set_trace()

    #################################################
    #this is for testing purposes only
    #generate plot of visibilities for quick check of how well flagging performed
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,10))
    chans = np.arange(0,109)
    for chan in chans:
        if chan not in flaglist[0]:
            chanpts = np.zeros(len(datacolxxamp_mask[:,chan]))+chan
            plt.plot(datacolxxamp_mask[:,chan],chanpts, '.', color='Blue', markersize=0.5)
            plt.plot(datacolyyamp_mask[:,chan],chanpts, '.', color='Green', markersize=0.5)
            plt.plot(datacolxyamp_mask[:,chan],chanpts, '.', color='Orange', markersize=0.5)
            plt.plot(datacolyxamp_mask[:,chan],chanpts, '.', color='Magenta', markersize=0.5)
    plt.ylim([0,108])
    plt.ylabel('channel')
    plt.xlabel('Amp')
    plt.gca().invert_yaxis()
    plotfile = os.path.splitext(os.path.abspath(msfile))[0]+'.png'
    plt.savefig(plotfile)

    ################################################

    if flaglist[0].size > 0:
        # turn flaglist into text file of channel flags
        textfile = os.path.splitext(os.path.abspath(msfile))[0]+'.chans'
        chans    = np.arange(0,109)
        chanlist = chans[flaglist]
        with open(textfile, 'w') as f:
            for chan in chanlist:
                f.write('%02d:%03d\n' % (np.int(band),chan))

        # write flags into FLAG column
        flagcol_altered = t.getcol('FLAG')
        flagcol_altered[:,flaglist,:] = 1
        t.putcol('FLAG', flagcol_altered)
        #os.system('apply_sb_flags_single_band_ms2.py %s %s %02d' % (textfile,msfile,np.int(band)) )
    t.close()

def main():
    parser = argparse.ArgumentParser(description="Flag bad channels and write out list of channels that were \
                                                  flagged into text file of same name as ms. MUST BE RUN ON \
                                                  SINGLE SUBBAND MS. Modified on 2020-03-26 to include cross-hand vis.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("band", help="Subband number.")
    parser.add_argument("--usedatacol", action="store_true", default=False, help="Grab DATA column, not CORRECTED_DATA.")
    args = parser.parse_args()
    flag_bad_chans(args.msfile,args.band,usedatacol=args.usedatacol)

if __name__ == '__main__':
    main()
