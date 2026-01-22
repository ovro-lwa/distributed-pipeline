"""
Improved dynamic-spectrum helpers that
* are 100 % read-only,
* respect the MS FLAG column,
* accept bcal=None.
"""
from __future__ import annotations
from pathlib import Path
from typing  import List, Iterable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import numpy as np
import numpy.ma  as ma
from casacore.tables import table
import redis
from astropy.io import fits
import os
from numba import njit

from orca.transform          import calibration 
from orca.utils.datetimeutils import STAGE_III_INTEGRATION_TIME
from orca.configmanager       import queue_config

from orca.celery import app

logger = logging.getLogger(__name__)

# 0. Constants reused from the orca/transform/spectrum.py taking into account averaging in freq by 4
N_CHAN_OUT       = 48
_CHAN_WIDTH_MHZ  = 0.023926 * 4
_TRANSPORT_DTYPE = np.float32
REDIS_URL        = queue_config.result_backend_uri
REDIS_EXPIRE_S   = 3600 * 10
REDIS_KEY_PREFIX = "spec-v2-"          # <= different key so no clash

ROW_NUMS = [                            # keep baseline list as in orca/transform/spectrum.py
    ('LWA-128&LWA-160', 54282),
    ('LWA-048&LWA-051', 33335),
    ('LWA-018&LWA-237', 30360),
    ('LWA-065&LWA-094', 41689),
    ('LWA-286&LWA-333', 28524),
]


#re-bin BCAL (192 → 48)
def _rebin_bcal(bcal: np.ndarray,
                n_chan_out: int = N_CHAN_OUT) -> np.ndarray:
    """
    Collapse a (n_ant, 192, 2) array down to (n_ant, 48, 2)
    by averaging every block of 4 channels, ignoring NaNs.

    Precondition: any flagged gains in `bcal` are already set to np.nan.
    """
    n_ant, n_chan_in, npol = bcal.shape
    factor = n_chan_in // n_chan_out   # should be 4

    # make sure it's complex64 so downstream dtype matches data
    bcal = bcal.astype(np.complex64)

    # group into (n_ant, 48, 4, 2)
    blocks = bcal.reshape(n_ant, n_chan_out, factor, npol)

    # nanmean over the 4-channel axis → (n_ant, 48, 2)
    rebinned = np.nanmean(blocks, axis=2)

    return rebinned.astype(np.complex64)


# 1. New calibration helper that also masks flags
'''
@njit
def _applycal_cross_with_flags(data, flags, bcal):
    """
    Calibrate |DATA| in-memory for cross-correlations.

    Parameters
    ----------
    data  : complex64[:, :, 4]
        Raw visibilities for ALL cross-corr rows (shape nRow × nChan × 4).
    flags : bool[:, :, 4]  (same shape)
        True  → flagged sample → output becomes NaN.
    bcal  : complex64[nAnt, nChan, 2]  OR  None
        Band-pass gains.  If None, a unity cube is assumed.
    """
    if bcal is None:
        return np.where(flags, np.nan, data)
    print("data, flags, bcal shapes", data.shape, flags.shape, bcal.shape)
    # invert gains once
    bcal_inv = 1.0 / bcal                   # (nAnt, nChan, 2)
    
    n_ant, n_chan = bcal.shape[0], bcal.shape[1]
    out = np.empty_like(data)

    i_row = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            for c in range(n_chan):
                if np.any(flags[i_row, c]):
                    out[i_row, c] = np.nan
                else:
                    g_i = np.diag(bcal_inv[i, c])        # 2×2
                    g_j = np.conjugate(np.diag(bcal_inv[j, c]))
                    out[i_row, c] = g_i @ data[i_row, c] @ g_j
            i_row += 1
    return out
'''



@njit
def _applycal_cross_with_flags(data, flags, bcal):
    """
    data  : complex64[nRow, nChan, 4]
    flags : bool     [nRow, nChan, 4]
    bcal  : complex64[nAnt, nChan, 2]  or None
    """

    n_row, n_chan, _ = data.shape

    # reshape into (nRow, nChan, 2, 2)
    data4 = data.reshape(n_row, n_chan, 2, 2)
    ans4  = np.empty_like(data4)

    # no bcal? just mask MS flags → NaNs
    if bcal is None:
        for i_row in range(n_row):
            for c in range(n_chan):
                if flags[i_row, c, 0] or flags[i_row, c, 1] \
                   or flags[i_row, c, 2] or flags[i_row, c, 3]:
                    ans4[i_row, c, 0, 0] = np.nan
                    ans4[i_row, c, 0, 1] = np.nan
                    ans4[i_row, c, 1, 0] = np.nan
                    ans4[i_row, c, 1, 1] = np.nan
                else:
                    # copy through
                    ans4[i_row, c, 0, 0] = data4[i_row, c, 0, 0]
                    ans4[i_row, c, 0, 1] = data4[i_row, c, 0, 1]
                    ans4[i_row, c, 1, 0] = data4[i_row, c, 1, 0]
                    ans4[i_row, c, 1, 1] = data4[i_row, c, 1, 1]
        return ans4.reshape(n_row, n_chan, 4)

    # invert bcal (1/bcal) as complex64
    bcal_inv = (1.0 / bcal).astype(np.complex64)
    n_ant     = bcal_inv.shape[0]

    i_row = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            for c in range(n_chan):
                # MS flags?
                if flags[i_row, c, 0] or flags[i_row, c, 1] \
                   or flags[i_row, c, 2] or flags[i_row, c, 3]:
                    ans4[i_row, c, 0, 0] = np.nan
                    ans4[i_row, c, 0, 1] = np.nan
                    ans4[i_row, c, 1, 0] = np.nan
                    ans4[i_row, c, 1, 1] = np.nan
                else:
                    # build diagonal gain matrices
                    g_i = np.diag(bcal_inv[i, c])        # 2×2
                    g_j = np.conjugate(np.diag(bcal_inv[j, c]))
                    # apply: g_i @ V @ g_j
                    tmp = g_i @ data4[i_row, c]
                    ans4[i_row, c] = tmp @ g_j
            i_row += 1

    # flatten back to (nRow, nChan, 4)
    return ans4.reshape(n_row, n_chan, 4)



# 2. Dataclass matches the old one so reducer logic can be reused
@dataclass
class _SnapshotSpectrumV2:
    type:        str
    subband_no:  int
    scan_no:     int
    key:         str
    
    def to_json(self):
        return {
            "type": self.type,
            "subband_no": self.subband_no,
            "scan_no": self.scan_no,
            "key": self.key,
            }

    @classmethod
    def from_json(cls, d):          # needed if ever send through Celery
        return cls(d["type"], d["subband_no"], d["scan_no"], d["key"])

# 3. New dynspec_map_v2  (read-only, flag-aware, bcal optional)
#@app.task(bind=True,autoretry_for=(Exception,),retry_kwargs={"max_retries": 3, "countdown": 10},) 
#def dynspec_map_v2(subband_no: int,
#                   scan_no:   int,
#                   ms:  str,
#                   bcal: str | None = None,
#                   use_ms_flags: bool = True,
#                  ) -> List[_SnapshotSpectrumV2]:
@app.task(
    name="orca.transform.spectrum_v2.dynspec_map_v2",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 10},
)
def dynspec_map_v2(self, subband_no:int, scan_no:int, ms:str, *, bcal:str=None, use_ms_flags:bool=True):
    """
    Produce incoherent-sum + selected-baseline spectra from a **single** MS.

    * Uses FLAG column to mask samples (→ NaN).
    * If `bcal` is None ⇒ unity gains.
    """

    # ----- open MS once --------------------------------------------------
    with table(ms, ack=False) as t:
        tcross   = t.query("ANTENNA1!=ANTENNA2")
        data     = tcross.getcol("DATA")                      # complex64
        flags    = tcross.getcol("FLAG")                      # bool
        ant1     = tcross.getcol("ANTENNA1")
        ant2     = tcross.getcol("ANTENNA2")

    # ----- derive nAnt from the row indices ------------------------------
    n_ant = int(max(ant1.max(), ant2.max()) + 1)

    # ----- load / fabricate band-pass gains ------------------------------
    if bcal is None:
        bcal_dat = None                      # unity will be assumed
    else:
        with table(bcal, ack=False) as tb:
            bcal_raw = tb.getcol("CPARAM")  
            bcal_raw_flag = tb.getcol("FLAG")
        bcal_raw[bcal_raw_flag] = np.nan

        bcal_dat = _rebin_bcal(bcal_raw)
    

    if use_ms_flags:
        flags_in = flags                    # MS FLAG column
    else:
        flags_in = np.zeros_like(flags)

    # ----- apply calibration + masking -----------------------------------
    calibrated = _applycal_cross_with_flags(
        data.astype(np.complex64),
        flags_in,
        bcal_dat.astype(np.complex64) if bcal_dat is not None else None
    )                                         # → shape (nRow, nChan, 4)

    amp = np.abs(calibrated)                  # magnitude, keeps NaNs

    # ----- produce spectra -----------------------------------------------
    print(f"  -> {Path(ms).name}: calibrated.shape = {calibrated.shape}")
    print(f"     FLAGS = {np.sum(np.isnan(calibrated))} NaNs")
    print(f"     amp.shape = {amp.shape}")

    incoh_sum = np.nanmean(amp, axis=0).astype(_TRANSPORT_DTYPE)  # (nChan,4)

    r   = redis.Redis.from_url(REDIS_URL)
    out: list[_SnapshotSpectrumV2] = []

    # store incoherent cube
    key_sum = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-sum"
    r.set(key_sum, incoh_sum.tobytes(), ex=REDIS_EXPIRE_S)
    out.append(_SnapshotSpectrumV2("incoherent-sum", subband_no, scan_no, key_sum))

    # store selected baselines
    for name, row_idx in ROW_NUMS:
        if row_idx >= amp.shape[0]:
            logger.warning("%s missing row %d in %s", name, row_idx, ms)
            continue
        key = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-{row_idx}"
        r.set(key, amp[row_idx].astype(_TRANSPORT_DTYPE).tobytes(), ex=REDIS_EXPIRE_S)
        out.append(_SnapshotSpectrumV2(name, subband_no, scan_no, key))

    #return out
    return [x.to_json() for x in out]


@app.task
def dynspec_reduce_v2(spectra: Iterable[List[_SnapshotSpectrumV2]],
                      start_ts: datetime,
                      out_dir: str) -> None:
    """
    Assemble per-scan lists from `dynspec_map_v2` into 768-freq FITS cubes.
    Safe against missing Redis keys / malformed blobs.
    """
    #n_scans = max(spectra, key=lambda x: x[0].scan_no)[0].scan_no + 1
    if isinstance(spectra[0][0], dict):
        spectra = [[_SnapshotSpectrumV2.from_json(s) for s in snap] for snap in spectra]

    n_scans = max(spec.scan_no for snap in spectra for spec in snap) + 1
    scan_nos = sorted({spec.scan_no for snap in spectra for spec in snap})
    print(f"[reduce_v2] n_scans={n_scans}, unique scan_nos={scan_nos}")

    n_freqs = N_CHAN_OUT * 16                         # 48 × 16 = 768
    types   = ['incoherent-sum'] + [name for name, _ in ROW_NUMS]

    all_dat = {t: np.zeros((4, n_scans, n_freqs), dtype=np.float32) for t in types}
    r       = redis.Redis.from_url(REDIS_URL)

    for snapshot in spectra:
        for spec in snapshot:
            if spec.type not in all_dat:
                continue
            j = spec.scan_no
            k = spec.subband_no * N_CHAN_OUT

            buf = r.get(spec.key)
            r.delete(spec.key)

            if buf is None:
                logger.warning("Missing Redis key: %s", spec.key)
                continue
            arr = np.frombuffer(buf, dtype=_TRANSPORT_DTYPE)
            if arr.size != N_CHAN_OUT * 4:
                logger.warning("Corrupted Redis entry: %s (%d values)", spec.key, arr.size)
                continue

            arr = arr.reshape(N_CHAN_OUT, 4)
            all_dat[spec.type][:, j, k:k+N_CHAN_OUT] = arr.T

    # ---- write FITS ----------------------------------------------------
    for name, dat in all_dat.items():
        for i, corr in enumerate(['XX', 'XY', 'YX', 'YY']):
            hdu = fits.PrimaryHDU(dat[i].T)

            # WCS header
            zero_ut = datetime(start_ts.year, start_ts.month, start_ts.day)
            hdr = hdu.header
            hdr['CTYPE2'] = 'FREQ'
            hdr['CUNIT2'] = 'MHz'
            hdr['CRVAL2'] = 13.398                   # same first-chan freq
            hdr['CDELT2'] = _CHAN_WIDTH_MHZ          # 0.095704 MHz
            hdr['CRPIX2'] = 1

            hdr['CTYPE1'] = 'TIME'
            hdr['CUNIT1'] = 'HOUR'
            hdr['CRVAL1'] = (start_ts - zero_ut).total_seconds() / 3600
            hdr['CDELT1'] = STAGE_III_INTEGRATION_TIME.total_seconds() / 3600
            hdr['CRPIX1'] = 1
            hdr['DATE-OBS'] = start_ts.date().isoformat()

            hdulist = fits.HDUList([hdu])
            out_sub = f"{out_dir}/{name}"
            os.makedirs(out_sub, exist_ok=True)
            hdulist.writeto(f"{out_sub}/{start_ts.date().isoformat()}-{corr}.fits",
                            overwrite=True)


