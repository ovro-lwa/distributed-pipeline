# orca/transform/spectrum_v3.py
"""
Dynamic-spectrum helpers (v3)
* 100% read-only MeasurementSet access
* Optionally respects MS FLAG column
* bcal optional; if present, used at native resolution (NO rebin)
* NO spectral averaging: 192 chans/subband → 16 SPWs → 3072 total chans
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Iterable
from dataclasses import dataclass
from datetime import datetime
import logging
import os

import numpy as np
from numba import njit
from casacore.tables import table
from astropy.io import fits
import redis

from orca.celery import app
from orca.utils.datetimeutils import STAGE_III_INTEGRATION_TIME
from orca.configmanager import queue_config

logger = logging.getLogger(__name__)

# --- Constants: NON-averaged mode ----------------------------------------
N_CHAN_OUT       = 192                 # full SPW width
_CHAN_WIDTH_MHZ  = 0.023926            # native per-channel width
_TRANSPORT_DTYPE = np.float32

REDIS_URL        = queue_config.result_backend_uri
REDIS_EXPIRE_S   = 3600 * 10
REDIS_KEY_PREFIX = "spec-v3-"          # distinct keyspace

# Selected cross-correlation rows (unchanged)
ROW_NUMS = [
    ('LWA-128&LWA-160', 54282),
    ('LWA-048&LWA-051', 33335),
    ('LWA-018&LWA-237', 30360),
    ('LWA-065&LWA-094', 41689),
    ('LWA-286&LWA-333', 28524),
]

# --- Calibration + flagging at native channelization ---------------------
@njit
def _applycal_cross_with_flags_native(data, flags, bcal):
    """
    data  : complex64[nRow, nChan, 4]
    flags : bool     [nRow, nChan, 4]
    bcal  : complex64[nAnt, nChan, 2]  or None

    Applies MS flags. If bcal is provided, applies diagonal 2x2 gains per ant.
    """
    n_row, n_chan, _ = data.shape
    data4 = data.reshape(n_row, n_chan, 2, 2)
    out4  = np.empty_like(data4)

    if bcal is None:
        for i in range(n_row):
            for c in range(n_chan):
                if flags[i, c, 0] or flags[i, c, 1] or flags[i, c, 2] or flags[i, c, 3]:
                    out4[i, c, 0, 0] = np.nan
                    out4[i, c, 0, 1] = np.nan
                    out4[i, c, 1, 0] = np.nan
                    out4[i, c, 1, 1] = np.nan
                else:
                    out4[i, c, 0, 0] = data4[i, c, 0, 0]
                    out4[i, c, 0, 1] = data4[i, c, 0, 1]
                    out4[i, c, 1, 0] = data4[i, c, 1, 0]
                    out4[i, c, 1, 1] = data4[i, c, 1, 1]
        return out4.reshape(n_row, n_chan, 4)

    bcal_inv = (1.0 / bcal).astype(np.complex64)
    n_ant    = bcal_inv.shape[0]

    i_row = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            for c in range(n_chan):
                if flags[i_row, c, 0] or flags[i_row, c, 1] or flags[i_row, c, 2] or flags[i_row, c, 3]:
                    out4[i_row, c, 0, 0] = np.nan
                    out4[i_row, c, 0, 1] = np.nan
                    out4[i_row, c, 1, 0] = np.nan
                    out4[i_row, c, 1, 1] = np.nan
                else:
                    g_i = np.diag(bcal_inv[i, c])         # 2x2
                    g_j = np.conjugate(np.diag(bcal_inv[j, c]))
                    tmp = g_i @ data4[i_row, c]
                    out4[i_row, c] = tmp @ g_j
            i_row += 1

    return out4.reshape(n_row, n_chan, 4)

# --- Payload dataclass (same shape as v2 to make downstream code simple) --
@dataclass
class _SnapshotSpectrumV3:
    type:        str
    subband_no:  int
    scan_no:     int
    key:         str

    def to_json(self):
        return {"type": self.type, "subband_no": self.subband_no, "scan_no": self.scan_no, "key": self.key}

    @classmethod
    def from_json(cls, d):
        return cls(d["type"], d["subband_no"], d["scan_no"], d["key"])

# --- Tasks ----------------------------------------------------------------
@app.task(name="orca.transform.spectrum_v3.dynspec_map_v3")
def dynspec_map_v3(subband_no: int,
                   scan_no:   int,
                   ms:        str,
                   bcal:      str | None = None,
                   use_ms_flags: bool = True) -> List[_SnapshotSpectrumV3]:
    """
    Map step on a SINGLE MS:
      • Reads DATA & FLAG (cross-correlations only)
      • Optionally applies bcal at native resolution (192 channels) — NO rebin
      • Stores incoherent-sum and selected-baseline spectra in Redis
    Returns JSONable _SnapshotSpectrumV3 list (for the reducer).
    """
    with table(ms, ack=False) as t:
        tcross = t.query("ANTENNA1!=ANTENNA2")
        data   = tcross.getcol("DATA")        # complex64, (nRow, nChan, 4)
        flags  = tcross.getcol("FLAG")        # bool,     (nRow, nChan, 4)
        ant1   = tcross.getcol("ANTENNA1")
        ant2   = tcross.getcol("ANTENNA2")

    n_ant = int(max(ant1.max(), ant2.max()) + 1)

    # bcal handling (NO rebin)
    if bcal is None:
        bcal_dat = None
    else:
        with table(bcal, ack=False) as tb:
            bcal_raw      = tb.getcol("CPARAM")  # (nAnt, nChan, 2)
            bcal_raw_flag = tb.getcol("FLAG")
        bcal_raw = bcal_raw.astype(np.complex64)
        # respect flagged gains → NaN (they’ll propagate)
        bcal_raw[bcal_raw_flag] = np.nan
        bcal_dat = bcal_raw

    flags_in = flags if use_ms_flags else np.zeros_like(flags, dtype=np.bool_)

    calibrated = _applycal_cross_with_flags_native(
        data.astype(np.complex64), flags_in, bcal_dat
    )
    amp = np.abs(calibrated).astype(_TRANSPORT_DTYPE)   # keep NaNs from flags

    # incoherent sum across baselines
    incoh_sum = np.nanmean(amp, axis=0).astype(_TRANSPORT_DTYPE)  # (nChan, 4)

    r   = redis.Redis.from_url(REDIS_URL)
    out : list[_SnapshotSpectrumV3] = []

    key_sum = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-sum"
    r.set(key_sum, incoh_sum.tobytes(), ex=REDIS_EXPIRE_S)
    out.append(_SnapshotSpectrumV3("incoherent-sum", subband_no, scan_no, key_sum))

    for name, row_idx in ROW_NUMS:
        if row_idx >= amp.shape[0]:
            logger.warning("%s missing row %d in %s", name, row_idx, ms)
            continue
        key = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-{row_idx}"
        r.set(key, amp[row_idx].tobytes(), ex=REDIS_EXPIRE_S)
        out.append(_SnapshotSpectrumV3(name, subband_no, scan_no, key))

    return [x.to_json() for x in out]

@app.task(name="orca.transform.spectrum_v3.dynspec_reduce_v3")
def dynspec_reduce_v3(spectra: Iterable[List[_SnapshotSpectrumV3]],
                      start_ts: datetime,
                      out_dir: str) -> None:
    """
    Reduce step:
      • Gathers Redis blobs from dynspec_map_v3 outputs
      • Builds 4×(time×freq) cubes per type, freq = 192×16 = 3072
      • Writes FITS into {out_dir}/{type}/{DATE}-{corr}.fits
    """
    # payload may arrive as list of dicts (Celery JSON); normalize
    if isinstance(spectra[0][0], dict):
        spectra = [[_SnapshotSpectrumV3.from_json(s) for s in snap] for snap in spectra]

    n_scans  = max(spec.scan_no for snap in spectra for spec in snap) + 1
    n_freqs  = N_CHAN_OUT * 16     # 192 × 16 = 3072
    types    = ['incoherent-sum'] + [name for name, _ in ROW_NUMS]

    cubes = {t: np.zeros((4, n_scans, n_freqs), dtype=_TRANSPORT_DTYPE) for t in types}
    r     = redis.Redis.from_url(REDIS_URL)

    for snapshot in spectra:
        for spec in snapshot:
            if spec.type not in cubes:
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
            arr = arr.reshape(N_CHAN_OUT, 4)     # (chan, corr)
            cubes[spec.type][:, j, k:k+N_CHAN_OUT] = arr.T  # (corr, time, freq)

    # --- Write FITS (XX, XY, YX, YY) ---
    for name, dat in cubes.items():
        for i, corr in enumerate(['XX', 'XY', 'YX', 'YY']):
            hdu = fits.PrimaryHDU(dat[i].T)

            # WCS header
            zero_ut = datetime(start_ts.year, start_ts.month, start_ts.day)
            hdr = hdu.header
            hdr['CTYPE2'] = 'FREQ'
            hdr['CUNIT2'] = 'MHz'
            hdr['CRVAL2'] = 13.398           # first chan MHz (unchanged baseline)
            hdr['CDELT2'] = _CHAN_WIDTH_MHZ  # 0.023926 MHz native
            hdr['CRPIX2'] = 1

            hdr['CTYPE1'] = 'TIME'
            hdr['CUNIT1'] = 'HOUR'
            hdr['CRVAL1'] = (start_ts - zero_ut).total_seconds() / 3600
            hdr['CDELT1'] = STAGE_III_INTEGRATION_TIME.total_seconds() / 3600
            hdr['CRPIX1'] = 1
            hdr['DATE-OBS'] = start_ts.date().isoformat()

            out_sub = f"{out_dir}/{name}"
            os.makedirs(out_sub, exist_ok=True)
            fits.HDUList([hdu]).writeto(
                f"{out_sub}/{start_ts.date().isoformat()}-{corr}.fits",
                overwrite=True
            )

