"""Bad-antenna flags and bandpass + XY-phase calibration in one casacore pass (prep: numpy).

Reproduces calibrate.flag_cal_average's CASA steps without CASA's applycal, which
spends most of its time writing a full-resolution CORRECTED_DATA column:

- flagdata(antenna=<csv>): cross-correlations of the listed antenna indices
  (CASA's antenna selection leaves their autocorrelations unflagged);
- flagdata(spw="*:85.0~100.0MHz") when the MS reaches above 85 MHz;
- applycal(gaintable=[B, Xf], calwt=False) with CASA's default 'linear' frequency
  interpolation: amplitude and phase interpolated linearly in frequency, flagged
  solution channels filled from the unflagged ones, data flagged only where an
  antenna-polarisation has no unflagged solution.  V_ij[pq] / (J_ip conj(J_jq)),
  J = B * Xf.

The calibrated data replace DATA in place (the MS is a scratch copy); mstransform
then averages DATA exactly as before.  Matches CASA to float32 precision (rms 7e-7,
max 6e-5 of the mean visibility on a 55 MHz snapshot), flags identical.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_CORR = [(0, 0), (0, 1), (1, 0), (1, 1)]
_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}
_lock = threading.Lock()


def _interp_ampphase(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    amp = np.interp(x, xp, np.abs(fp))
    ph = np.interp(x, xp, np.unwrap(np.angle(fp)))
    return amp * np.exp(1j * ph)


def _solutions(caltable: str, spw: int, freq: np.ndarray, nant: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(nant, nchan, 2) gains at the MS channels and (nant, 2) all-flagged mask."""
    from casacore.tables import table
    with table(caltable, ack=False) as t:
        q = t.query(f"SPECTRAL_WINDOW_ID=={spw}")
        g, f, ant = q.getcol("CPARAM"), q.getcol("FLAG"), q.getcol("ANTENNA1")
    with table(caltable + "/SPECTRAL_WINDOW", ack=False) as s:
        fr = s.getcell("CHAN_FREQ", spw)
    gains = np.ones((nant, len(freq), 2), complex)
    dead = np.zeros((nant, 2), bool)
    for row, a in enumerate(ant):
        for p in range(2):
            ok = ~f[row, :, p]
            if not ok.any():
                dead[a, p] = True
                continue
            gains[a, :, p] = _interp_ampphase(freq, fr[ok], g[row, ok, p])
    return gains, dead


def jones(ms: str, bp_table: str, xy_table: str, nant: int, freq: np.ndarray
          ) -> Tuple[np.ndarray, np.ndarray]:
    """B * Xf per antenna, channel and feed; cached per (MS band, tables)."""
    from ..subbands import spwmap
    key = (f"{freq[0]:.0f}-{freq[-1]:.0f}-{len(freq)}", bp_table, xy_table)
    with _lock:
        if key in _cache:
            return _cache[key]
    gb, db = _solutions(bp_table, spwmap(ms, bp_table)[0], freq, nant)
    gx, dx = _solutions(xy_table, spwmap(ms, xy_table)[0], freq, nant)
    out = (gb * gx, db | dx)
    with _lock:
        _cache[key] = out
    return out


def band_info(ms: str) -> Tuple[np.ndarray, int]:
    """CHAN_FREQ of the single SPW and the number of antennas."""
    from casacore.tables import table
    with table(ms + "/SPECTRAL_WINDOW", ack=False) as s:
        if s.nrows() != 1:
            raise ValueError(f"{ms}: expected one SPW, found {s.nrows()}")
        freq = s.getcell("CHAN_FREQ", 0)
    with table(ms + "/ANTENNA", ack=False) as a:
        nant = a.nrows()
    return freq, nant


def calibrate_arrays(data: np.ndarray, flag: np.ndarray, flag_row: np.ndarray,
                     a1: np.ndarray, a2: np.ndarray, freq: np.ndarray, J: np.ndarray,
                     dead: np.ndarray, badants_csv: str) -> None:
    """Bad-antenna/85-100 MHz flags and B*Xf calibration of DATA, all in place.

    Shared by flag_calibrate (v2, MS in place) and v3.fastprep (arrays only).
    """
    if badants_csv:
        bad = [int(x) for x in badants_csv.split(",") if x.strip()]
        rows = (np.isin(a1, bad) | np.isin(a2, bad)) & (a1 != a2)
        flag[rows] = True
        flag_row |= rows
    if freq.max() > 85e6:
        flag[:, (freq >= 85e6) & (freq <= 100e6), :] = True
    cached = _gain_products(J, a1, a2) if os.environ.get("GSI_PREP_GAIN_CACHE", "0") == "1" else None
    for k, (p, q) in enumerate(_CORR):
        g = cached[k] if cached is not None else J[a1, :, p] * np.conj(J[a2, :, q])
        data[:, :, k] = (data[:, :, k] / g).astype(data.dtype)
        flag[dead[a1, p] | dead[a2, q], :, k] = True


_gp_cache: Dict[tuple, list] = {}


def _gain_products(J: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> list:
    """GSI_PREP_GAIN_CACHE=1 (v3.prep_gain_cache): the per-row gain products J[a1, :, p] * conj(J[a2, :, q]) of
    the four correlations, kept per process while J and the row order stay the same (every snapshot of a
    band-hour): the same arrays as computed per snapshot, ~0.8 GB per process at 192 channels."""
    import hashlib
    key = (id(J), J.shape, a1.dtype.str, a2.dtype.str, hashlib.sha1(np.ascontiguousarray(a1).tobytes()).hexdigest(),
           hashlib.sha1(np.ascontiguousarray(a2).tobytes()).hexdigest())
    with _lock:
        hit = _gp_cache.get(key)
        if hit is not None and hit[0] is J:
            return hit[1]
    out = [J[a1, :, p] * np.conj(J[a2, :, q]) for p, q in _CORR]
    with _lock:
        _gp_cache.clear()
        _gp_cache[key] = (J, out)
    return out


def sigma_from_weight(w: np.ndarray) -> np.ndarray:
    """SIGMA = 1/sqrt(WEIGHT) (0 where WEIGHT <= 0), in WEIGHT's dtype.

    flagdata sets FLAG_ROW on the rows it flags (applycal does not); mstransform of DATA
    derives the output weights from SIGMA (raw MSes carry SIGMA=9999, WEIGHT=1), of
    CORRECTED_DATA from WEIGHT -- make SIGMA match WEIGHT so averaging DATA gives the
    old WEIGHT/SIGMA.
    """
    return np.where(w > 0, 1.0 / np.sqrt(np.where(w > 0, w, 1.0)), 0.0).astype(w.dtype)


def flag_calibrate(ms: Path, bp_table: str, xy_table: str, badants_csv: str) -> None:
    """Flag bad antennas (and 85-100 MHz) and calibrate DATA in place."""
    from casacore.tables import table
    ms = str(ms)
    freq, nant = band_info(ms)
    J, dead = jones(ms, bp_table, xy_table, nant, freq)
    with table(ms, readonly=False, ack=False) as t:
        data = t.getcol("DATA")
        flag = t.getcol("FLAG")
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        flag_row = t.getcol("FLAG_ROW")
        calibrate_arrays(data, flag, flag_row, a1, a2, freq, J, dead, badants_csv)
        t.putcol("DATA", data)
        t.putcol("FLAG", flag)
        t.putcol("FLAG_ROW", flag_row)
        t.putcol("SIGMA", sigma_from_weight(t.getcol("WEIGHT")))


def average(ms: Path, out_ms: Path, chanbin: int) -> Path:
    """mstransform channel average of DATA (CASA subprocess)."""
    from .casarun import run_py
    run_py(f"from casatasks import mstransform\n"
           f"mstransform(vis='{ms}', outputvis='{out_ms}', datacolumn='data', "
           f"chanaverage=True, chanbin={chanbin})")
    return out_ms
