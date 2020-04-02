#!/usr/bin/env python

from __future__ import division
import numpy as np
import pyrap.tables as pt
import os,argparse
import numpy.ma as ma
from scipy.stats import skew
from scipy.ndimage import filters

def flag_bad_ants(msfile):
    """
    Input: msfile
    Returns list of antennas to be flagged based on autocorrelations.
    """
    t       = pt.table(msfile, readonly=True)
    tautos  = t.query('ANTENNA1=ANTENNA2')
    
    # iterate over antenna, 1-->256
    datacolxx = np.zeros((256,2398))
    datacolyy = np.copy(datacolxx)
    for antind,tauto in enumerate(tautos.iter("ANTENNA1")):
        print antind
        for bandind,tband in enumerate(tauto):
            datacolxx[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,0]
            datacolyy[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,3]

    datacolxxamp = np.sqrt( np.real(datacolxx)**2. + np.imag(datacolxx)**2. )
    datacolyyamp = np.sqrt( np.real(datacolyy)**2. + np.imag(datacolyy)**2. )

    datacolxxampdb = 10*np.log10(datacolxxamp/1.e2)
    datacolyyampdb = 10*np.log10(datacolyyamp/1.e2)

    # median value for every antenna
    medamp_perantx = np.median(datacolxxampdb,axis=1)
    medamp_peranty = np.median(datacolyyampdb,axis=1)

    # get flags based on deviation from median amp
    xthresh_pos = np.median(medamp_perantx) + np.std(medamp_perantx)
    xthresh_neg = np.median(medamp_perantx) - 2*np.std(medamp_perantx)
    ythresh_pos = np.median(medamp_peranty) + np.std(medamp_peranty)
    ythresh_neg = np.median(medamp_peranty) - 2*np.std(medamp_peranty)
    flags = np.where( (medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) |\
                      (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) )

    # use unflagged antennas to generate median spectrum
    flagmask = np.zeros((256,2398))
    flagmask[flags[0],:] = 1
    datacolxxampdb_mask = ma.masked_array(datacolxxampdb, mask=flagmask, fill_value=np.nan)
    datacolyyampdb_mask = ma.masked_array(datacolyyampdb, mask=flagmask, fill_value=np.nan)

    medamp_allantsx = np.median(datacolxxampdb_mask,axis=0)
    medamp_allantsy = np.median(datacolyyampdb_mask,axis=0)

    stdarrayx = np.array( [np.std(antarr/medamp_allantsx) for antarr in datacolxxampdb_mask] )
    stdarrayy = np.array( [np.std(antarr/medamp_allantsy) for antarr in datacolyyampdb_mask] )
    
    # this threshold was manually selected...should be changed to something better at some point
    flags2 = np.where( (stdarrayx > 0.02) | (stdarrayy > 0.02) )

    flagsall = np.sort(np.append(flags,flags2))
    flagsallstr = [str(flag) for flag in flagsall]
    flagsallstr2 = ",".join(flagsallstr)

    antflagfile = os.path.dirname(os.path.abspath(msfile)) + '/flag_bad_ants.ants'
    with open(antflagfile,'w') as f:
        f.write(flagsallstr2)
    
    t.close()

def main():
    parser = argparse.ArgumentParser(description="Returns list of antennas to flag, based on power levels for \
                                                  autocorrelations in a single msfile. DOES NOT ACTUALLY FLAG \
                                                  THOSE ANTENNAS, JUST RETURNS A LIST TO BE FLAGGED.")
    parser.add_argument("msfile", help="Measurement set. Must be fullband measurement set, created with \
                                        ~/imaging_scripts/gen_autos.py.")
    args = parser.parse_args()
    flag_bad_ants(args.msfile)

if __name__ == '__main__':
    main()
