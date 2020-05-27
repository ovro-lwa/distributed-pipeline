#!/usr/bin/env python

from __future__ import division
import numpy as np
import pyrap.tables as pt
import os,argparse

def flag_bls(msfile, blfile):
    """
    Input: msfile, .bl file
    Applies baseline flags to FLAG column.
    """
    t       = pt.table(msfile, readonly=False)
    flagcol = t.getcol('FLAG')  # flagcol.shape = (N*(N-1)/2 + N)*spw, 109, 4
    Nants   = t.getcol('ANTENNA1')[-1] + 1
    Nbls    = int((Nants*(Nants-1)/2.) + Nants)
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

    t.putcol('FLAG',newflagcol)
    t .close()

def main():
    parser = argparse.ArgumentParser(description="Baseline flagger.")
    parser.add_argument("msfile", help="Measurement set.")
    parser.add_argument("blfile", help="List of baseline flags. Expected format: one baseline per line, of the form 'ant1&ant2'. 0-indexed.")
    args = parser.parse_args()
    flag_bls(args.msfile,args.blfile)

if __name__ == '__main__':
    main()
