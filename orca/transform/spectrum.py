from casacore.tables import table
from orca.transform import dftspectrum
from astropy.coordinates import SkyCoord
from os import path
import numpy as np
import numpy.ma as ma
import pdb

def gen_spectrum(ms: str, sourcename: str, data_column: str = 'CORRECTED_DATA', timeavg: bool = False, outdir: str = None, target_coordinates: str = None, apply_weights: str = None):
    """
    Generate spectrum (I,V,XX,XY,YX,YY) from the visibilities; if target_coordinates not assigned, assumes source of interest
    is already at phase center; if apply_weights not assigned, no weights applied.

    Args:
        ms: The measurement set.
        sourcename: The source for which spectrum is being generated. Used for naming output file.
        data_column: MS data column on which to operate. Default is CORRECTED_DATA.
        timeavg: Average in time. Default is False.
        outdir: Path to where output .npz file should be written. Default is path to input ms.
        apply_weights: Imaging weights npy file (from wsclean-2.5 -store-imaging-weights,
            IMAGING_WEIGHT_SPECTRUM column).

    Returns:
        Path to output .npz file containing spectrum.
    """
    # open ms, SPW table, datacol, flagcol, freqcol
    with table(f'{ms}/SPECTRAL_WINDOW') as tspw:
        freqcol = tspw.getcol('CHAN_FREQ')
    with table(ms) as t:
        tcross  = t.query('ANTENNA1!=ANTENNA2')
        flagcol = tcross.getcol('FLAG')
        timecol = tcross.getcol('TIME')
        if target_coordinates:
            with table(f'{ms}/FIELD') as tfield:
                ra, dec      = tfield.getcol('PHASE_DIR')[0][0]
                phase_center = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian')
            datacol = dftspectrum.phase_shift_vis(tcross, freqcol, phase_center, SkyCoord(target_coordinates), data_column)
        else:
            datacol = tcross.getcol(data_column) # datacol.shape = (N*(N-1)/2)*spw*int, Nchans, Ncorrs
        #
        Nants   = t.getcol('ANTENNA1')[-1] + 1
    Nbls    = int(Nants*(Nants-1)/2.)
    Nchans  = datacol.shape[1]
    Ncorrs  = datacol.shape[2]
    Nspw    = freqcol.shape[0]
    Nints   = int(datacol.shape[0]/(Nbls*Nspw))

    #
    # reorder visibilities by Nints, Nbls, Nchans*Nspw, Ncorr
    datacol_ma = ma.masked_array(datacol, mask=flagcol, fill_value=np.nan).reshape(Nspw, Nints, Nbls, Nchans, Ncorrs).transpose(1,2,0,3,4).reshape(Nints,Nbls,-1,Ncorrs)

    # apply weights
    if apply_weights:
        weights = np.load(apply_weights).reshape(Nspw, Nints, Nbls, Nchans, Ncorrs).transpose(1,2,0,3,4).reshape(Nints,Nbls,-1,Ncorrs)
        datacol_ma *= np.multiply(datacol, weights)
        Nbls_eff = np.sum((~datacol_ma * weights), axis=1)
    else:
        Nbls_eff = np.sum(~datacol_ma, axis=1)

    # Weighted mean along the bl axis.
    datacol_ma = datacol_ma.sum(axis=1) / Nbls_eff

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