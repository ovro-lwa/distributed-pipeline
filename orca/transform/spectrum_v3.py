# orca/transform/spectrum_v3.py
"""
Dynamic-spectrum helpers (v3)
* 100% read-only MeasurementSet access
* Optionally respects MS FLAG column
* bcal optional; if present, used at native resolution (NO rebin)
* NO spectral averaging: 192 chans/subband → 16 SPWs → 3072 total chans
* Accepts ``.ms`` directories or slow-buffer ``.ms.tar`` archives
  (extracted to local NVMe for the duration of the task)
* Verifies that each hardcoded ROW_NUMS row really is the named baseline
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Iterable
from dataclasses import dataclass
from datetime import datetime
import logging
import os
import shutil
import tarfile
import tempfile

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
_CHAN_WIDTH_MHZ  = 0.02392578125       # native per-channel width (196 MHz / 8192)
_FREQ0_MHZ       = 13.3984375          # centre of channel 0 of the 13MHz SPW
_TRANSPORT_DTYPE = np.float32

REDIS_URL        = queue_config.result_backend_uri
REDIS_EXPIRE_S   = 3600 * 10
REDIS_KEY_PREFIX = "spec-v3-"          # distinct keyspace

TAR_WORK_DIR     = "/fast/pipeline/dynspec_tmp"   # node-local NVMe scratch for .ms.tar

# Selected cross-correlation rows (index into the ANTENNA1!=ANTENNA2 query).
# Correlator order is not antenna-name order, so dynspec_map_v3 checks each row
# against the ANTENNA table and falls back to a name lookup if it has moved.
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

def _baseline_names(name: str) -> List[str]:
    """'LWA-128&LWA-160' -> ['LWA128', 'LWA160'] (MS ANTENNA NAME spelling)."""
    return sorted(a.replace('-', '') for a in name.split('&'))


def _resolve_rows(ms: str, ant1: np.ndarray, ant2: np.ndarray) -> dict:
    """Map each ROW_NUMS name to the cross-corr row that holds that baseline.

    Uses the hardcoded row when the ANTENNA table confirms it; otherwise
    searches by antenna name and logs a warning.  Names that are not present
    in the MS are left out.
    """
    with table(f"{ms}/ANTENNA", ack=False) as ta:
        ant_names = list(ta.getcol("NAME"))
    rows = {}
    for name, row_idx in ROW_NUMS:
        want = _baseline_names(name)
        if row_idx < len(ant1) and sorted([ant_names[ant1[row_idx]], ant_names[ant2[row_idx]]]) == want:
            rows[name] = row_idx
            continue
        if not all(w in ant_names for w in want):
            logger.warning("%s: antennas not in %s; skipping", name, ms)
            continue
        i, j = ant_names.index(want[0]), ant_names.index(want[1])
        hit = np.where(((ant1 == i) & (ant2 == j)) | ((ant1 == j) & (ant2 == i)))[0]
        if len(hit) != 1:
            logger.warning("%s: found %d rows in %s; skipping", name, len(hit), ms)
            continue
        logger.warning("%s: hardcoded row %d is wrong in %s, using row %d", name, row_idx, ms, hit[0])
        rows[name] = int(hit[0])
    return rows


def _extract_ms_tar(tar_path: str, work_dir: str) -> tuple:
    """Extract a ``.ms.tar`` into a fresh temp dir; return (tmp_dir, ms_path)."""
    os.makedirs(work_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="dynspec-", dir=work_dir)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=tmp_dir)
    ms_path = os.path.join(tmp_dir, os.path.basename(tar_path)[:-4])
    if not os.path.isdir(ms_path):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError(f"{ms_path} not found after extracting {tar_path}")
    return tmp_dir, ms_path


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

    ``ms`` may be a ``.ms.tar``; it is extracted under TAR_WORK_DIR and
    removed again when the task finishes.
    """
    tmp_dir = None
    ms_path = ms
    try:
        if ms.endswith(".ms.tar"):
            tmp_dir, ms_path = _extract_ms_tar(ms, TAR_WORK_DIR)
        with table(ms_path, ack=False) as t:
            tcross = t.query("ANTENNA1!=ANTENNA2")
            data   = tcross.getcol("DATA")        # complex64, (nRow, nChan, 4)
            flags  = tcross.getcol("FLAG")        # bool,     (nRow, nChan, 4)
            ant1   = tcross.getcol("ANTENNA1")
            ant2   = tcross.getcol("ANTENNA2")
            tcross.close()
        rows = _resolve_rows(ms_path, ant1, ant2)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
        data.astype(np.complex64, copy=False), flags_in, bcal_dat
    )
    amp = np.abs(calibrated).astype(_TRANSPORT_DTYPE)   # keep NaNs from flags

    # incoherent sum across baselines
    incoh_sum = np.nanmean(amp, axis=0).astype(_TRANSPORT_DTYPE)  # (nChan, 4)

    r   = redis.Redis.from_url(REDIS_URL)
    out : list[_SnapshotSpectrumV3] = []

    key_sum = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-sum"
    r.set(key_sum, incoh_sum.tobytes(), ex=REDIS_EXPIRE_S)
    out.append(_SnapshotSpectrumV3("incoherent-sum", subband_no, scan_no, key_sum))

    for name, _ in ROW_NUMS:
        if name not in rows:
            continue
        row_idx = rows[name]
        key = f"{REDIS_KEY_PREFIX}{Path(ms).stem}-{row_idx}"
        r.set(key, amp[row_idx].tobytes(), ex=REDIS_EXPIRE_S)
        out.append(_SnapshotSpectrumV3(name, subband_no, scan_no, key))

    return [x.to_json() for x in out]

@app.task(name="orca.transform.spectrum_v3.dynspec_reduce_v3")
def dynspec_reduce_v3(spectra: Iterable[List[_SnapshotSpectrumV3]],
                      start_ts: datetime,
                      out_dir: str,
                      meta: dict | None = None) -> None:
    """
    Reduce step:
      • Gathers Redis blobs from dynspec_map_v3 outputs
      • Builds 4×(time×freq) cubes per type, freq = 192×16 = 3072
      • Writes FITS into {out_dir}/{type}/{DATE}-{corr}.fits
    ``start_ts`` is the centre of scan 0.  ``meta`` holds extra string/number
    FITS cards (key → value or (value, comment)) written to every file.
    """
    if isinstance(start_ts, str):
        start_ts = datetime.fromisoformat(start_ts)
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
            hdr['CRVAL2'] = _FREQ0_MHZ       # first chan MHz
            hdr['CDELT2'] = _CHAN_WIDTH_MHZ  # 0.023926 MHz native
            hdr['CRPIX2'] = 1

            hdr['CTYPE1'] = 'TIME'
            hdr['CUNIT1'] = 'HOUR'
            hdr['CRVAL1'] = (start_ts - zero_ut).total_seconds() / 3600
            hdr['CDELT1'] = STAGE_III_INTEGRATION_TIME.total_seconds() / 3600
            hdr['CRPIX1'] = 1
            hdr['DATE-OBS'] = start_ts.date().isoformat()
            hdr['TIMESYS'] = 'UTC'
            hdr['TSTART'] = (start_ts.isoformat(), 'UTC centre of first integration')
            hdr['TELESCOP'] = 'OVRO-LWA'
            hdr['BASELINE'] = (name, 'baseline or incoherent-sum')
            hdr['STOKES'] = (corr, 'correlation product')
            hdr['BUNIT'] = ('arbitrary', '|visibility|, uncalibrated unless BCAL set')
            hdr['NSCANS'] = n_scans
            hdr['NCHAN'] = n_freqs
            for key, val in (meta or {}).items():
                hdr[key] = tuple(val) if isinstance(val, (list, tuple)) else val

            out_sub = f"{out_dir}/{name}"
            os.makedirs(out_sub, exist_ok=True)
            fits.HDUList([hdu]).writeto(
                f"{out_sub}/{start_ts.date().isoformat()}-{corr}.fits",
                overwrite=True
            )

