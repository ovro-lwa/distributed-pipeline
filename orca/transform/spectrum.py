from casacore import tables
from os import path
import numpy as np
import numpy.ma as ma
import pdb

def gen_spectrum(ms: str, sourcename: str, data_column: str = 'CORRECTED_DATA', timeavg: bool = False, outdir: str = None):
    """
    Generate spectrum (I,V,XX,XY,YX,YY) from the visibilities; assumes source of interest
    is already phase center.

    Args:
        ms: The measurement set.
        sourcename: The source for which spectrum is being generated. Used for naming output file.
        data_column: MS data column on which to operate. Default is CORRECTED_DATA.
        timeavg: Average in time. Default is False.

    Returns:
        Path to output .npz file containing spectrum.
    """
    # open ms, SPW table, datacol, flagcol, freqcol
    t       = tables.table(ms, readonly=True)
    tspw    = tables.table(ms+'/SPECTRAL_WINDOW')
    tcross  = t.query('ANTENNA1!=ANTENNA2')
    datacol = tcross.getcol(data_column) # datacol.shape = (N*(N-1)/2)*spw*int, Nchans, Ncorrs
    flagcol = tcross.getcol('FLAG')
    freqcol = tspw.getcol('CHAN_FREQ')
    timecol = tcross.getcol('TIME')
    #
    Nants   = t.getcol('ANTENNA1')[-1] + 1
    Nbls    = int(Nants*(Nants-1)/2.)
    Nchans  = datacol.shape[1]
    Ncorrs  = datacol.shape[2]
    Nspw    = freqcol.shape[0]
    Nints   = int(datacol.shape[0]/(Nbls*Nspw))
    #    
    t.close()
    tspw.close()
    #
    # reorder visibilities by Nints, Nbls, Nchans*Nspw, Ncorr and take the mean on the Nbls axis
    datacol_ma = ma.masked_array(datacol, mask=flagcol, fill_value=np.nan).reshape(Nspw, Nints, Nbls, Nchans, Ncorrs).transpose(1,2,0,3,4).reshape(Nints,Nbls,-1,Ncorrs).mean(axis=1)
    #
    specI      = 0.5 * (datacol_ma[:,:,0] + datacol_ma[:,:,3]).real
    specV      = 0.5 * (datacol_ma[:,:,1] - datacol_ma[:,:,2]).imag
    #
    frqarr  = freqcol.reshape(-1)
    timearr = np.unique(timecol)
    #
    if timeavg:
        datacol_ma.mean(axis=0)
        specI  = specI.mean(axis=0)
        specV  = specV.mean(axis=0)
    #
    if outdir:
        outfile = f'{outdir}/{path.splitext(path.basename(ms))[0]}_{sourcename}-spectrum'
    else:
        outfile = f'{path.splitext(path.abspath(ms))[0]}_{sourcename}-spectrum'
    datacol_ma.set_fill_value(np.nan)
    specI.set_fill_value(np.nan)
    specV.set_fill_value(np.nan)
    np.savez(outfile, specI=specI.filled(), specV=specV.filled(), frqarr=frqarr, timearr=timearr, speccorr=datacol_ma.filled())
    
    return outfile+'.npz'