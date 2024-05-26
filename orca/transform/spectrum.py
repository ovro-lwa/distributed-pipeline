from os import path
import os
import uuid
from typing import List, Iterable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from kombu.utils.json import register_type

from casacore.tables import table
from astropy.coordinates import SkyCoord
import numpy as np
import numpy.ma as ma
import numpy as np

from astropy.io import fits
import redis

from orca.celery import app, REDIS_URL
from orca.transform import dftspectrum, calibration
from orca.utils.datetimeutils import STAGE_III_INTEGRATION_TIME

logger = logging.getLogger(__name__)

N_CHAN = 192

REDIS_EXPIRE_S = 3600*5
REDIS_KEY_PREFIX = 'dynspec-'

# row numbers in cross corr
ROW_NUMS = [('LWA-128&LWA-160', 54282), 
            ('LWA-048&LWA-051', 33335),
            ('LWA-018&LWA-237', 30360),
            ('LWA-065&LWA-094', 41689)]

def gen_spectrum(ms: str, sourcename: str, data_column: str = 'CORRECTED_DATA', timeavg: bool = False, outdir: str = None, target_coordinates: str = None, apply_weights: str = None, apply_beam: bool = False):
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
    # apply weights
    if apply_weights:
        weights = np.load(apply_weights)
        datacol *= weights
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

@dataclass
class _SnapshotSpectrum:
    type: str
    subband_no : int
    scan_no: int

    key: str

    @classmethod
    def from_json(cls, data):
        return cls(data['type'], data['subband_no'], data['scan_no'],
                   data['key'])


register_type(_SnapshotSpectrum, '_SnapshotSpectrum',
              asdict,
              _SnapshotSpectrum.from_json)

_TRANSPORT_DTYPE = np.float32

@app.task
def dynspec_map(subband_no:int, scan_no:int, ms: str, bcal: str) -> List[_SnapshotSpectrum]:
    with table(ms, ack=False) as t:
        tcross = t.query('ANTENNA1!=ANTENNA2')
        dat = tcross.getcol('DATA')

    if '13MHz' in bcal:
        bcal_dat = np.ones((352, 192, 2), dtype=np.complex64) * 0.008
    else:
        with table(bcal, ack=False) as t:
            bcal_dat = t.getcol('CPARAM')
            bcal_flag = t.getcol('FLAG')
            bcal_dat[bcal_flag] = np.inf

    amp = np.abs(calibration.applycal_in_mem_cross(dat, bcal_dat))
    incoh_sum = np.mean(amp, axis=0).astype(_TRANSPORT_DTYPE)
    r = redis.Redis.from_url(REDIS_URL)
    key1 = REDIS_KEY_PREFIX + str(uuid.uuid4())
    r.set(key1, incoh_sum.tobytes(), ex=REDIS_EXPIRE_S)
    out = [_SnapshotSpectrum('incoherent-sum', subband_no, scan_no, key1)]
    
    amp = amp.astype(_TRANSPORT_DTYPE)
    for name, i in ROW_NUMS:
        entry_key = REDIS_KEY_PREFIX + str(uuid.uuid4())
        r.set(entry_key, amp[i, :, :].tobytes(), ex=REDIS_EXPIRE_S)
        out.append(
            _SnapshotSpectrum(name, subband_no, scan_no, entry_key)
        )

    return out

@app.task
def dynspec_reduce(spectra: Iterable[List[_SnapshotSpectrum]], start_ts: datetime, out_dir: str) -> None:
    n_scans = max(spectra, key=lambda x: x[0].scan_no)[0].scan_no + 1
    n_freqs = 192 * 16
    types = ['incoherent-sum'] + [name for name, _ in ROW_NUMS]
    # first axis is corr
    all_dat = {t: np.zeros((4, n_scans, n_freqs), dtype = np.float32) for t in types}

    r = redis.Redis.from_url(REDIS_URL)

    for all_spec_in_snapshot in spectra:
        for spec in all_spec_in_snapshot:
            if spec.type in all_dat:
                j = spec.scan_no
                k = spec.subband_no * N_CHAN
                spec_dat = np.frombuffer(r.get(spec.key), dtype=_TRANSPORT_DTYPE).reshape(N_CHAN, 4)
                r.delete(spec.key)
                all_dat[spec.type][0, j, k:k+N_CHAN] = spec_dat[:, 0]
                all_dat[spec.type][1, j, k:k+N_CHAN] = spec_dat[:, 1]
                all_dat[spec.type][2, j, k:k+N_CHAN] = spec_dat[:, 2]
                all_dat[spec.type][3, j, k:k+N_CHAN] = spec_dat[:, 3]

    for name, dat in all_dat.items():
        for i, corr in enumerate(['XX', 'XY', 'YX', 'YY']):
            hdu = fits.PrimaryHDU(dat[i].T)

            zero_ut = datetime(start_ts.year, start_ts.month, start_ts.day, 0, 0, 0)
            header = hdu.header
            header['CTYPE2'] = 'FREQ    '
            header['CUNIT2'] = 'MHz      '
            header['CRVAL2'] = 17.992 # lowest channel freq
            header['CDELT2'] = 0.023926 # channel width
            header['CRPIX2'] = 1

            header['CTYPE1'] = 'TIME    '
            header['CUNIT1'] = 'HOUR      '
            header['CRVAL1'] = (start_ts - zero_ut).total_seconds() / 3600
            header['CDELT1'] = STAGE_III_INTEGRATION_TIME.seconds / 3600
            header['CRPIX1'] = 1
            header['DATE-OBS'] = start_ts.date().isoformat()
            hdulist = fits.HDUList([hdu])
            out_dir_dir = f'{out_dir}/{name}'
            os.makedirs(out_dir_dir, exist_ok=True)
            hdulist.writeto(f'{out_dir_dir}/{start_ts.date().isoformat()}-{corr}.fits', overwrite=True)


def test_make_dynspec_fits():
    frequency = np.linspace(100, 200, 50)
    time = np.linspace(0, 10, 100)
    intensity = np.zeros((len(time), len(frequency)))
    intensity[0, 0] = 1
    intensity[49, 24] = 7

    # Create a Primary HDU
    hdu = fits.PrimaryHDU(intensity.T)

    # Create the header and add necessary information
    header = hdu.header
    header['CTYPE2'] = 'FREQ    '
    header['CUNIT2'] = 'MHz      '
    header['CRVAL2'] = frequency[0]
    header['CDELT2'] = frequency[1] - frequency[0]
    header['CRPIX2'] = 1  # Reference pixel for the first axis

    header['CTYPE1'] = 'TIME    '
    header['CUNIT1'] = 'UTC_HOUR      '
    header['CRVAL1'] = time[0]
    header['CDELT1'] = time[1] - time[0]
    header['CRPIX1'] = 1
    header['DATE-OBS'] = '2021-01-01'

    hdulist = fits.HDUList([hdu])
    hdulist.writeto('dynamic_spectrum.fits', overwrite=True)
