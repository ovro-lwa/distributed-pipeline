"""In-memory numpy prep (gsi v3 step 2, docs/v3_design.md section 7).

One casacore read of the raw snapshot MS, then in numpy:

1. ``fastcal.calibrate_arrays``: bad-antenna and 85-100 MHz flags, B * Xf calibration
   (the v2 ``prep: numpy`` math, without its putcols);
2. ``average_like_mstransform``: CASA mstransform's channel average of DATA, FLAG,
   WEIGHT and SIGMA, bit for bit;
3. ``uvw.fix_uvw_arrays``: the v2 J2000 UVW fix;

giving a ``SnapArrays``.  ``write_ms_like`` turns one back into the CASA MS that v2's
mstransform + fix_uvw would have produced, for the stages that still read MSes.

What mstransform (CASA 6.5.5, ``datacolumn='data', chanaverage=True``, no
WEIGHT_SPECTRUM in the input) does, established on 23 MHz snapshots with random
partial flags and random WEIGHT / SIGMA (``tests/test_v3_fastprep.py`` pins it):

- DATA: per output channel and correlation, the *unweighted* mean of the unflagged
  input channels: float32 sum in channel order, then real and imaginary parts each
  divided by the float32 count.  A bin with every input flagged gets the mean of all
  its inputs (same arithmetic), and is flagged.  WEIGHT and SIGMA do not weight DATA.
- FLAG: set where every input channel of the bin is flagged.  FLAG_ROW: copied from
  the input; it does not flag the output data (only FLAG does).
- Channel weights are 1/SIGMA^2 (float32; the input WEIGHT column is ignored).  The
  output weight of a bin is their float32 sum over its unflagged channels (over all of
  them if all are flagged).  WEIGHT is the float32 mean of the bin weights over the
  output channels that are not flagged (over all if every one is), SIGMA the same mean
  of 1/sqrt(bin weight).  No WEIGHT_SPECTRUM / SIGMA_SPECTRUM column is written.
- CHAN_FREQ is the mean, CHAN_WIDTH / EFFECTIVE_BW / RESOLUTION the sum of each bin.
- Every other main-table column and every subtable value is copied unchanged, but the
  table and subtable *descriptions* are mstransform's own (tiled storage managers,
  different column comments, PROCESSOR without SETTINGS, ...).  ``write_ms_like`` takes
  the structure from a template written by mstransform and the values from the raw MS.

Inputs with SIGMA <= 0 or non-finite SIGMA are rejected (mstransform's handling there
is not reproduced); the v2 prep never produces them from the raw WEIGHT = 1.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..stages import fastcal, quality, uvw
from .snap import SnapArrays

# main-table columns a SnapArrays holds; the other ones are copied from the raw MS
SNAP_COLUMNS = {"ANTENNA1": "antenna1", "ANTENNA2": "antenna2", "UVW": "uvw", "DATA": "data",
                "FLAG": "flag", "WEIGHT": "weight", "SIGMA": "sigma", "FLAG_ROW": "flag_row",
                "TIME": "time", "TIME_CENTROID": "time_centroid", "EXPOSURE": "exposure",
                "INTERVAL": "interval"}
RAW_COLUMNS = ("ARRAY_ID", "DATA_DESC_ID", "FEED1", "FEED2", "FIELD_ID", "OBSERVATION_ID",
               "PROCESSOR_ID", "SCAN_NUMBER", "STATE_ID")
UNSET_COLUMNS = ("FLAG_CATEGORY",)               # mstransform leaves every cell undefined
# subtables whose values mstransform derives from the averaging (taken from the template)
TEMPLATE_SUBTABLES = ("SPECTRAL_WINDOW", "DATA_DESCRIPTION", "POLARIZATION")


# -- averaging ------------------------------------------------------------------------

def _divide_parts(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """complex64 / float32 as two IEEE float32 divisions (numpy's complex division
    multiplies by the reciprocal and rounds differently)."""
    parts = num.view(np.float32).reshape(num.shape + (2,))
    return (parts / den[..., None]).reshape(num.shape[:-1] + (2 * num.shape[-1],)
                                           ).view(np.complex64)


def average_like_mstransform(data: np.ndarray, flag: np.ndarray, sigma: np.ndarray,
                             chanbin: int
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """mstransform channel average: (data, flag, weight, sigma) of the averaged rows.

    ``data`` (nrow, nchan, ncorr) complex64, ``flag`` same shape bool, ``sigma``
    (nrow, ncorr) float32 (the input WEIGHT column does not enter, see module doc).
    ``nchan`` must be a multiple of ``chanbin``.
    """
    n, c, p = data.shape
    if data.dtype != np.complex64:
        raise TypeError(f"DATA must be complex64, got {data.dtype}")
    if flag.shape != data.shape or sigma.shape != (n, p):
        raise ValueError(f"shape mismatch: DATA {data.shape} FLAG {flag.shape} SIGMA {sigma.shape}")
    if chanbin < 1 or c % chanbin:
        raise ValueError(f"{c} channels are not a multiple of chanbin {chanbin}")
    sigma = np.asarray(sigma, dtype=np.float32)
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("SIGMA <= 0 or not finite: mstransform's weights not reproduced")
    m = c // chanbin
    d4 = data.reshape(n, m, chanbin, p)
    ok = ~flag.reshape(n, m, chanbin, p)

    # Sum of all inputs, in CASA's channel order.  A bin with every input unflagged has
    # the same sum, and one with every input flagged uses it too (mean of all), so only
    # partially flagged bins (rare: flags come per row or per channel range) need the
    # sum over their unflagged channels.
    acc = np.zeros((n, m, p), np.complex64)
    cnt = np.zeros((n, m, p), np.int16)
    for k in range(chanbin):
        acc += d4[:, :, k]
        cnt += ok[:, :, k]
    part = np.nonzero((cnt > 0) & (cnt < chanbin))
    if part[0].size:
        i, j, q = part
        sub = np.zeros(i.size, np.complex64)
        zero = np.complex64(0)
        for k in range(chanbin):
            sub += np.where(ok[i, j, k, q], d4[i, j, k, q], zero)
        acc[part] = sub
    allf = cnt == 0                                             # output FLAG
    neff = np.where(allf, np.int16(chanbin), cnt)               # channels in each mean
    out = _divide_parts(acc, neff.astype(np.float32))
    del acc

    # Bin weight: the float32 sum of neff channel weights 1/SIGMA^2; tabulate the partial
    # sums S[k] = w + w + ... (k terms) per row and correlation, then look them up.
    wch = np.float32(1) / np.square(sigma)                      # (n, p)
    S = np.zeros((n, chanbin + 1, p), np.float32)
    for k in range(1, chanbin + 1):
        S[:, k] = S[:, k - 1] + wch
    S[:, 0] = 1                                                 # never used (neff >= 1)
    idx = neff.astype(np.intp)
    ws = np.take_along_axis(S, idx, axis=1)                     # (n, m, p)
    sig_ch = np.take_along_axis(np.float32(1) / np.sqrt(S), idx, axis=1)
    use = ~allf
    use |= allf.all(axis=1, keepdims=True)                      # every bin flagged: use all
    w_sum = np.zeros((n, p), np.float32)
    s_sum = np.zeros((n, p), np.float32)
    zero32 = np.float32(0)
    for j in range(m):                                          # output channel order
        uj = use[:, j]
        w_sum += np.where(uj, ws[:, j], zero32)
        s_sum += np.where(uj, sig_ch[:, j], zero32)
    nuse = use.sum(axis=1).astype(np.float32)
    return out, allf, w_sum / nuse, s_sum / nuse


def average_channels(chan_freq: np.ndarray, chan_width: np.ndarray, chanbin: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """mstransform's output CHAN_FREQ (bin mean) and CHAN_WIDTH (bin sum)."""
    f = np.asarray(chan_freq, np.float64).reshape(-1, chanbin)
    w = np.asarray(chan_width, np.float64).reshape(-1, chanbin)
    width = w[:, 0].copy()
    for k in range(1, chanbin):
        width += w[:, k]
    return f.mean(axis=1), width


# -- prep -------------------------------------------------------------------------------

def _clock() -> Tuple[float, float]:
    return time.perf_counter(), time.process_time()


def _lap(timings: Dict[str, Any], name: str, t: Tuple[float, float]) -> Tuple[float, float]:
    now = _clock()
    timings[name] = {"wall": now[0] - t[0], "cpu": now[1] - t[1]}
    return now


def read_raw(raw_ms: str) -> Dict[str, Any]:
    """The main-table columns and subtable values prep needs, in one pass."""
    from casacore.tables import table
    raw_ms = str(raw_ms)
    out: Dict[str, Any] = {}
    freq, nant = fastcal.band_info(raw_ms)
    with table(raw_ms + "/SPECTRAL_WINDOW", ack=False) as s:
        out["chan_width_in"] = s.getcell("CHAN_WIDTH", 0)
    out["chan_freq_in"], out["nant"] = freq, nant
    with table(raw_ms + "/ANTENNA", ack=False) as a:
        out["ant_pos"] = a.getcol("POSITION")
    with table(raw_ms + "/FIELD", ack=False) as f:
        out["field_name"] = str(f.getcell("NAME", 0)) if f.nrows() else ""
        out["phase_dir"] = f.getcol("PHASE_DIR")[0, 0].copy() if f.nrows() else None
        out["phase_dir_ref"] = f.getcolkeywords("PHASE_DIR").get("MEASINFO", {}).get("Ref", "J2000")
        out["n_field"] = f.nrows()
    with table(raw_ms + "/OBSERVATION", ack=False) as o:
        out["telescope"] = str(o.getcell("TELESCOPE_NAME", 0)) if o.nrows() else ""
        out["observer"] = str(o.getcell("OBSERVER", 0)) if o.nrows() else ""
    with table(raw_ms, ack=False) as t:
        for c in ("DATA", "FLAG", "FLAG_ROW", "WEIGHT", "ANTENNA1", "ANTENNA2", "UVW", "TIME",
                  "TIME_CENTROID", "EXPOSURE", "INTERVAL"):
            out[c] = t.getcol(c)
    return out


def prep_arrays(raw_ms: str, bp: str, xy: str, badants: str, chanbin: int,
                fix_uvw_mode: str = "centroid") -> SnapArrays:
    """Raw snapshot MS -> calibrated, averaged, UVW-fixed SnapArrays (raw MS untouched).

    Equals v2 ``fastcal.flag_calibrate`` + ``fastcal.average`` (mstransform) +
    ``Worker._fix_uvw``: a failed UVW fix keeps the stored UVW (``meta['fix_uvw_error']``)
    as v2 does.  ``meta['timings']`` holds wall and CPU seconds per stage.
    """
    timings: Dict[str, Any] = {}
    t = t_all = _clock()
    r = read_raw(raw_ms)
    t = _lap(timings, "read", t)

    data, flag, flag_row = r.pop("DATA"), r.pop("FLAG"), r.pop("FLAG_ROW")
    a1, a2 = r.pop("ANTENNA1"), r.pop("ANTENNA2")
    freq = r["chan_freq_in"]
    J, dead = fastcal.jones(str(raw_ms), bp, xy, r["nant"], freq)
    fastcal.calibrate_arrays(data, flag, flag_row, a1, a2, freq, J, dead, badants)
    sigma_in = fastcal.sigma_from_weight(r.pop("WEIGHT"))
    t = _lap(timings, "cal", t)

    avg, aflag, weight, sigma = average_like_mstransform(data, flag, sigma_in, chanbin)
    del data, flag
    freqs, widths = average_channels(freq, r["chan_width_in"], chanbin)
    t = _lap(timings, "average", t)

    stored = r.pop("UVW")
    new_uvw, fix_stats, fix_err = stored, None, None
    if fix_uvw_mode != "off":
        try:
            if r["n_field"] != 1:
                raise ValueError(f"{raw_ms}: expected one FIELD row, got {r['n_field']}")
            if r["phase_dir_ref"] != "J2000":
                raise ValueError(f"{raw_ms}: PHASE_DIR frame {r['phase_dir_ref']}, expected J2000")
            if fix_uvw_mode not in uvw.EPOCHS:
                raise ValueError(f"uvw epoch must be one of {uvw.EPOCHS}, got {fix_uvw_mode!r}")
            tcol = r["TIME_CENTROID"] if fix_uvw_mode == "centroid" else r["TIME"]
            ra, dec = (float(x) for x in r["phase_dir"])
            new_uvw, phase, fix_stats = uvw.fix_uvw_arrays(
                stored, a1, a2, tcol, r["ant_pos"], ra, dec, freqs, label=str(raw_ms))
            if phase is not None:
                avg = (avg * phase).astype(avg.dtype)
        except Exception as e:          # v2: image with the stored UVW rather than lose it
            new_uvw, fix_stats, fix_err = stored, None, str(e)
    t = _lap(timings, "uvw", t)
    timings["total"] = {"wall": t[0] - t_all[0], "cpu": t[1] - t_all[1]}

    stem = Path(str(raw_ms)).name
    stem = stem[:-3] if stem.endswith(".ms") else stem
    meta = {"freqs": freqs, "chan_width": widths, "phase_dir": r["phase_dir"],
            "ant_pos": r["ant_pos"], "telescope": r["telescope"], "observer": r["observer"],
            "field_name": r["field_name"], "chanbin": int(chanbin), "badants": badants,
            "fix_uvw_mode": fix_uvw_mode,
            "fix_uvw": None if fix_stats is None else fix_stats.__dict__.copy(),
            "fix_uvw_error": fix_err, "timings": timings, "raw_ms": str(raw_ms)}
    return SnapArrays(stem=stem, antenna1=a1, antenna2=a2, uvw=new_uvw, data=avg, flag=aflag,
                      weight=weight, sigma=sigma, flag_row=flag_row, time=r["TIME"],
                      time_centroid=r["TIME_CENTROID"], exposure=r["EXPOSURE"],
                      interval=r["INTERVAL"], meta=meta)


# -- MS writing -------------------------------------------------------------------------

def make_template(raw_ms: str, out: str, chanbin: int) -> Path:
    """mstransform of a raw snapshot: the structure write_ms_like copies (one CASA call)."""
    shutil.rmtree(str(out), ignore_errors=True)
    return fastcal.average(Path(raw_ms), Path(out), chanbin)


def _copy_column(src, dst, col: str) -> None:
    """Every cell of ``src``'s column into ``dst`` (same row count); a cell undefined in
    ``src`` must be undefined in ``dst`` too, so no template value survives."""
    n = src.nrows()
    if n == 0:
        return
    try:
        dst.putcol(col, src.getcol(col))        # all cells defined, fixed shape
        return
    except Exception:
        pass
    for r in range(n):
        if src.iscelldefined(col, r):
            dst.putcell(col, r, src.getcell(col, r))
        elif dst.iscelldefined(col, r):
            raise ValueError(f"{col} row {r}: undefined in the raw MS, defined in the template")


def _fill_subtable(src_path: str, dst_path: str) -> None:
    """Values of ``src_path`` into the (template-structured) ``dst_path``."""
    from casacore.tables import table
    with table(src_path, ack=False) as s, table(dst_path, readonly=False, ack=False) as d:
        missing = sorted(set(d.colnames()) - set(s.colnames()))
        if missing:
            raise ValueError(f"{src_path} lacks template column(s) {missing}")
        ns, nd = s.nrows(), d.nrows()
        if nd > ns:
            d.removerows(list(range(ns, nd)))
        elif ns > nd:
            d.addrows(ns - nd)
        for col in d.colnames():
            _copy_column(s, d, col)


# SPECTRAL_WINDOW columns mstransform recomputes; every other one it copies
_SW_SUMMED = ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION")


def check_template(template_avg_ms: str, raw_ms: str, chanbin: int) -> None:
    """Raise ValueError unless the template is what mstransform makes of ``raw_ms``.

    Cheap (small subtables only): main-table row count; SPECTRAL_WINDOW (every column:
    averaged CHAN_FREQ, summed widths, NUM_CHAN, REF_FREQUENCY = first output channel,
    the others as in the raw MS); POLARIZATION and DATA_DESCRIPTION values; and the
    column keywords (frames, units) of those three subtables.
    """
    from casacore.tables import table
    from .mscompare import _column_values, _eq
    tpl, raw = str(template_avg_ms), str(raw_ms)
    with table(tpl, ack=False) as t, table(raw, ack=False) as r:
        if t.nrows() != r.nrows():
            raise ValueError(f"template has {t.nrows()} rows, {raw} {r.nrows()}")
    for sub in TEMPLATE_SUBTABLES:
        with table(f"{tpl}/{sub}", ack=False) as t, table(f"{raw}/{sub}", ack=False) as r:
            if t.nrows() != r.nrows():
                raise ValueError(f"{sub}: template {t.nrows()} rows, raw {r.nrows()}")
            missing = sorted(set(t.colnames()) - set(r.colnames()))
            if missing:
                raise ValueError(f"{sub}: raw MS lacks {missing}")
            want = {c: _column_values(r, c) for c in t.colnames()}
            if sub == "SPECTRAL_WINDOW":
                if r.nrows() != 1:
                    raise ValueError(f"{raw}: {r.nrows()} spectral windows")
                f, w = r.getcell("CHAN_FREQ", 0), r.getcell("CHAN_WIDTH", 0)
                if f.size % chanbin:
                    raise ValueError(f"{f.size} channels are not a multiple of chanbin {chanbin}")
                freqs, _ = average_channels(f, w, chanbin)
                want["CHAN_FREQ"] = freqs[None, :]
                for c in _SW_SUMMED:
                    want[c] = average_channels(f, r.getcell(c, 0), chanbin)[1][None, :]
                want["NUM_CHAN"] = np.array([freqs.size], want["NUM_CHAN"].dtype)
                want["REF_FREQUENCY"] = np.array([freqs[0]])
            for c in t.colnames():
                if not _eq(_column_values(t, c), want[c]):
                    raise ValueError(f"template {sub}/{c} does not match {raw}")
                if not _eq(t.getcolkeywords(c), r.getcolkeywords(c)):
                    raise ValueError(f"template {sub}/{c} keywords differ from {raw}")


def write_ms_like(template_avg_ms: str, raw_ms: str, snap: SnapArrays, out: str) -> Path:
    """Write ``snap`` as the MS v2's mstransform (+ fix_uvw) would have written.

    ``template_avg_ms``: an mstransform output of the same band, chanbin and row count
    (any snapshot; ``make_template``).  It is copied file by file, so the table and
    subtable structure is mstransform's, and its SPECTRAL_WINDOW / DATA_DESCRIPTION /
    POLARIZATION are kept.  Every other subtable gets the values of ``raw_ms`` (as
    mstransform copies them), every main column is overwritten from ``snap`` or, for the
    ID columns a SnapArrays does not hold, from ``raw_ms``.
    """
    from casacore.tables import table
    out_p, tpl, raw = Path(out), str(template_avg_ms), str(raw_ms)
    shutil.rmtree(str(out_p), ignore_errors=True)
    check_template(tpl, raw, int(snap.meta.get("chanbin", 0)) or 1)
    with table(tpl + "/SPECTRAL_WINDOW", ack=False) as s:
        tf = s.getcol("CHAN_FREQ")
    want = np.asarray(snap.meta.get("freqs"))
    if tf.shape != (1, snap.nchan) or want.shape != (snap.nchan,) or not np.array_equal(tf[0], want):
        raise ValueError(f"template {tpl} SPECTRAL_WINDOW does not match {snap.stem}")
    # A file copy keeps the storage-manager layout exactly (a casacore copy with
    # copynorows re-derives bucket sizes and empties the subtables); every main-table
    # column is then overwritten.
    with table(tpl, ack=False) as t:
        if t.nrows() != snap.nrow:
            raise ValueError(f"template {tpl}: {t.nrows()} rows, {snap.stem} has {snap.nrow}")
    shutil.copytree(tpl, str(out_p))
    try:
        with table(raw, ack=False) as r, table(str(out_p), readonly=False, ack=False) as t:
            if r.nrows() != snap.nrow:
                raise ValueError(f"{raw}: {r.nrows()} rows, snapshot has {snap.nrow}")
            for col in t.colnames():
                if col in SNAP_COLUMNS:
                    t.putcol(col, getattr(snap, SNAP_COLUMNS[col]))
                elif col in UNSET_COLUMNS:
                    continue
                elif col in r.colnames():
                    t.putcol(col, r.getcol(col))
                else:
                    raise ValueError(f"template column {col} has no source")
        for sub in sorted(p.name for p in out_p.iterdir() if (p / "table.dat").is_file()):
            if sub in TEMPLATE_SUBTABLES:
                continue
            if not (Path(raw) / sub / "table.dat").is_file():
                # mstransform then makes one up and edits others (FIELD SOURCE_ID without
                # a SOURCE table): not reproduced, the caller falls back to v2
                raise ValueError(f"{raw} has no {sub} subtable")
            _fill_subtable(f"{raw}/{sub}", str(out_p / sub))
    except Exception:
        shutil.rmtree(str(out_p), ignore_errors=True)
        raise
    return out_p


def read_ms_arrays(ms: str) -> Dict[str, np.ndarray]:
    """The SnapArrays main columns of an averaged MS (for comparisons)."""
    from casacore.tables import table
    with table(str(ms), ack=False) as t:
        return {c: t.getcol(c) for c in SNAP_COLUMNS}


def _ndiff_bits(a: np.ndarray, b: np.ndarray) -> int:
    """Number of elements whose bytes differ (so NaN == NaN and -0.0 != 0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype.kind not in "fc":
        return int(np.count_nonzero(a != b))
    if a.itemsize <= 8:
        u = np.dtype(f"u{a.itemsize}")
        return int(np.count_nonzero(a.view(u) != b.view(u)))
    ne = a.view(np.uint64).reshape(a.shape + (-1,)) != b.view(np.uint64).reshape(b.shape + (-1,))
    return int(np.count_nonzero(ne.any(axis=-1)))


def compare_snap_ms(snap: SnapArrays, ms: str) -> Dict[str, Any]:
    """Differences between ``snap`` and the main table of ``ms`` (e.g. v2's output).

    DATA: max |diff|, rms(diff)/rms(ms DATA), number of differing values (bitwise);
    the other columns: bitwise equality (and max |diff| for float columns).
    """
    m = read_ms_arrays(ms)
    out: Dict[str, Any] = {"nrow": snap.nrow, "nrow_ms": int(m["DATA"].shape[0])}
    if m["DATA"].shape != snap.data.shape:
        out["shape_mismatch"] = f"{snap.data.shape} vs {m['DATA'].shape}"
        return out
    d = snap.data.astype(np.complex128) - m["DATA"]
    ref = float(np.sqrt(np.mean(np.abs(m["DATA"].astype(np.complex128)) ** 2)))
    out["data_max_abs"] = float(np.max(np.abs(d))) if d.size else 0.0
    out["data_rel_rms"] = float(np.sqrt(np.mean(np.abs(d) ** 2))) / ref if ref > 0 else 0.0
    del d
    out["data_ndiff"] = _ndiff_bits(snap.data, m["DATA"])
    for col, attr in SNAP_COLUMNS.items():
        if col == "DATA":
            continue
        a, b = np.asarray(getattr(snap, attr)), m[col]
        key = col.lower()
        if a.shape != b.shape or a.dtype != b.dtype:
            out[key + "_ndiff"] = -1
            out[key + "_mismatch"] = f"{a.dtype}{a.shape} vs {b.dtype}{b.shape}"
            continue
        out[key + "_ndiff"] = _ndiff_bits(a, b)
        if a.dtype.kind == "f" and a.size:
            out[key + "_max_abs"] = float(np.max(np.abs(a.astype(np.float64) - b)))
    out["identical"] = (out["data_ndiff"] == 0 and
                        all(v == 0 for k, v in out.items() if k.endswith("_ndiff")))
    return out


# -- process-pool tasks (spawn: module-level and picklable) -----------------------------

def prep_task(raw_ms: str, bp: str, xy: str, badants: str, chanbin: int, fix_uvw_mode: str,
              sample_rows: int, min_amplitude: float, template: Optional[str] = None,
              out_ms: Optional[str] = None) -> Dict[str, Any]:
    """prep_arrays in a pool process; with ``out_ms`` also write it (write_ms_like).

    Returns {"meta", "stats" (quality.VisibilityStats as dict), "snap" (only without
    out_ms, i.e. shadow mode: the arrays travel back to the caller)}.
    """
    t_task = time.perf_counter()
    if out_ms is not None:                  # a wrong template: fail before the work
        check_template(template, raw_ms, chanbin)
    snap = prep_arrays(raw_ms, bp, xy, badants, chanbin, fix_uvw_mode)
    st = quality.visibility_stats_arrays(snap.data, snap.flag, sample_rows, min_amplitude)
    res: Dict[str, Any] = {"stats": st.__dict__.copy(), "pid": os.getpid()}
    if out_ms is not None:
        t0 = _clock()
        write_ms_like(template, raw_ms, snap, out_ms)
        _lap(snap.meta["timings"], "write", t0)
    else:
        res["snap"] = snap
    res["meta"] = {k: v for k, v in snap.meta.items() if k not in ("ant_pos",)}
    import resource
    res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    res["task_wall"] = time.perf_counter() - t_task
    return res


def prep_slot_task(raw_ms: str, bp: str, xy: str, badants: str, chanbin: int, fix_uvw_mode: str,
                   sample_rows: int, min_amplitude: float, slot: str,
                   min_nonzero_fraction: float = 0.0) -> Dict[str, Any]:
    """gsi v3 steps 2+4 (prep_inmem=on, peel_io=shm, snapshot_io=shm): prep_arrays in a pool
    process and write the result straight into the shm slot ``slot`` (state PREPPED), no MS.

    The slot is not written when the visibility statistics are below ``min_nonzero_fraction``
    (the worker skips such a snapshot before peeling, as v2).  Returns {"meta" (without the
    arrays), "stats", "slot_written", "slot_nbytes", "task_wall", "maxrss_mb", "pid"}."""
    from . import slots
    t_task = time.perf_counter()
    snap = prep_arrays(raw_ms, bp, xy, badants, chanbin, fix_uvw_mode)
    timings = snap.meta.pop("timings", {})       # the slot JSON stays small; the timings travel back here
    st = quality.visibility_stats_arrays(snap.data, snap.flag, sample_rows, min_amplitude)
    res: Dict[str, Any] = {"stats": st.__dict__.copy(), "pid": os.getpid(), "slot_written": False}
    if st.nonzero_fraction >= min_nonzero_fraction:
        t0 = _clock()
        h = slots.write_slot(slot, snap)
        _lap(timings, "slot_write", t0)
        res["slot_written"], res["slot_nbytes"] = True, int(h["nbytes"])
    res["meta"] = {k: v for k, v in snap.meta.items() if k not in ("ant_pos", "freqs", "chan_width")}
    res["meta"]["timings"] = timings
    import resource
    res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    res["task_wall"] = time.perf_counter() - t_task
    return res


def kept_from_slot_task(template_avg_ms: str, raw_ms: str, slot: str, out: str) -> Dict[str, Any]:
    """Write the (peeled) slot as the kept MS ``out`` (write_ms_like) in a pool process: the MS
    v2 keeps for the same snapshot (the slot holds what the zest daemon's MS write would)."""
    from . import slots
    t0 = time.perf_counter()
    snap = slots.view_slot(slot, writable=False)
    write_ms_like(template_avg_ms, raw_ms, snap, out)
    del snap
    return {"wall": time.perf_counter() - t0, "pid": os.getpid()}
