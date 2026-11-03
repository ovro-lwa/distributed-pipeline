"""Peeling solutions (peel.save_solutions): per-snapshot NPZ and one merged NPZ per band-hour.

workers/zest_daemon.jl (both engines) writes <target>.peelsol/<pass>.bin + <pass>.tsv after each
peel, where <target> is the MS (v2) or the /dev/shm slot (v3).  :func:`convert` turns them into
<band dir>/peeling_solutions/.snapshots/<stem>/<pass>.npz with the main pipeline's schema 1
(orca/wrapper/ttcal.py: gains (source, jones, antenna, frequency, time), source_indices,
source_names, frequencies_hz, times_mjd_seconds, antenna_indices, jones_order, metadata_json)
plus per-source elevation_deg, accepted, converged and niters.  Gains are complex64, the MS
visibility precision.  :func:`merge` combines one pass of a band-hour into schema 2
(orca/transform/peeling_solutions.py: time concatenated, sources as the union, NaN gains and
solved=False where a source was not peeled) with the per-source arrays as (source, time):
<band dir>/peeling_solutions/<band>MHz_<date>_<LST>_<pass>.npz (as feature/gsi-streaming).

Source indices are 0-based within the pass's catalog (the sky model for "sky", the RFI model for
"rfi"; with peel.rfi_mode=joint one "sky" pass over the sky then the RFI model).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PASSES = ("sky", "rfi")
_JONES = np.array(["xx", "xy", "yx", "yy"])
PARTS = ".snapshots"          # per-snapshot NPZs below <band dir>/peeling_solutions/


def _read_tsv(path: Path):
    lines = path.read_text().splitlines()
    shape = tuple(int(x) for x in lines[0].split()[2:])
    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    return shape, rows


def ms_axes(ms: Path) -> Tuple[np.ndarray, float]:
    """(channel frequencies [Hz], first TIME [MJD s]) of an MS."""
    from casacore.tables import table
    with table(str(ms) + "/SPECTRAL_WINDOW", ack=False) as s:
        freq = np.asarray(s.getcell("CHAN_FREQ", 0), dtype=np.float64)
    with table(str(ms), ack=False) as t:
        return freq, float(t.getcell("TIME", 0))


def slot_axes(slot: Path) -> Tuple[np.ndarray, float]:
    """The same axes of a v3 shm slot (its meta.freqs and first TIME row, as the MS's)."""
    from ..v3 import slots
    snap = slots.view_slot(slot, writable=False)
    return np.array(snap.meta["freqs"], dtype=np.float64), float(snap.time[0])


def sol_dir(target: Path) -> Path:
    """Where the daemon writes the solutions of a peeled MS or slot."""
    return Path(f"{target}.peelsol")


def parts_dir(band_dir: Path) -> Path:
    return Path(band_dir) / "peeling_solutions" / PARTS


def convert(src: Path, out_dir: Path, meta: Dict, axes: Tuple[np.ndarray, float],
            ms_name: str) -> List[Path]:
    """Daemon output `src` of one snapshot -> <out_dir>/<pass>.npz; removes `src`.  Returns the files.

    axes: (channel frequencies [Hz], TIME [MJD s]); ms_name: the snapshot's MS name in the
    metadata (v2: the peeled <stamp>_avg.ms; v3 writes the same name, there is no MS)."""
    src, out_dir = Path(src), Path(out_dir)
    if not src.is_dir():
        return []
    freq, time_s = axes
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in PASSES:
        tsv, binf = src / f"{p}.tsv", src / f"{p}.bin"
        if not tsv.exists():
            continue
        shape, rows = _read_tsv(tsv)
        g = np.fromfile(binf, dtype="<c8").reshape(shape, order="F")[..., None]
        if g.shape[3] != len(freq):
            raise ValueError(f"{binf}: {g.shape[3]} channels, the snapshot has {len(freq)}")
        md = dict(meta, schema_version=1, ms=ms_name, stage=p, routine="zest",
                  gain_axes=["source", "jones", "antenna", "frequency", "time"],
                  gains_dtype="complex64", convergence_flags_available=True,
                  backstop_reverted=(src / f"{p}.reverted").exists())
        out = out_dir / f"{p}.npz"
        tmp = out_dir / f".{p}.tmp.npz"
        np.savez_compressed(
            tmp, gains=g, invalid_gains=~np.isfinite(g).all(axis=1),
            source_indices=np.array([int(r[0]) for r in rows], dtype=np.int64),
            source_names=np.array([r[1] for r in rows], dtype=str),
            elevation_deg=np.array([float(r[2]) for r in rows]),
            accepted=np.array([r[3] == "true" for r in rows]),
            converged=np.array([r[4] == "true" for r in rows]),
            niters=np.array([int(r[5]) for r in rows], dtype=np.int64),
            frequencies_hz=freq, times_mjd_seconds=np.array([time_s]),
            antenna_indices=np.arange(g.shape[2]), jones_order=_JONES,
            metadata_json=np.array(json.dumps(md, default=str)))
        os.replace(tmp, out)
        written.append(out)
    shutil.rmtree(src, ignore_errors=True)
    return written


def merge(files: List[Path], output: Path) -> Path:
    """Merge one pass's per-snapshot NPZ files (schema 1) into one schema-2 NPZ."""
    parts = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
    parts.sort(key=lambda p: float(p["times_mjd_seconds"][0]))
    nant, nfreq = parts[0]["gains"].shape[2:4]
    for p in parts:
        if p["gains"].shape[2:4] != (nant, nfreq) or not np.array_equal(
                p["frequencies_hz"], parts[0]["frequencies_hz"]):
            raise ValueError(f"antenna/frequency axes differ in {json.loads(str(p['metadata_json']))['ms']}")
    names = {}
    for p in parts:
        names.update(zip(p["source_indices"].tolist(), p["source_names"].tolist()))
    idx = sorted(names)
    row = {i: k for k, i in enumerate(idx)}
    nt = len(parts)
    gains = np.full((len(idx), 4, nant, nfreq, nt), np.nan + 1j * np.nan, dtype=np.complex64)
    per = {k: np.zeros((len(idx), nt), dtype=dt) for k, dt in
           (("solved", bool), ("accepted", bool), ("converged", bool), ("niters", np.int64))}
    elev = np.full((len(idx), nt), np.nan)
    for t, p in enumerate(parts):
        for k, i in enumerate(p["source_indices"].tolist()):
            r = row[i]
            gains[r, ..., t] = p["gains"][k, ..., 0]
            per["solved"][r, t] = True
            for key in ("accepted", "converged", "niters"):
                per[key][r, t] = p[key][k]
            elev[r, t] = p["elevation_deg"][k]
    metas = [json.loads(str(p["metadata_json"])) for p in parts]
    md = {k: v for k, v in metas[0].items() if k not in ("ms", "backstop_reverted")}
    md.update(schema_version=2, n_ms=nt, gain_axes=["source", "jones", "antenna", "frequency", "time"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f".{output.name}.tmp.npz"
    np.savez_compressed(
        tmp, gains=gains, elevation_deg=elev, **per,
        backstop_reverted=np.array([m.get("backstop_reverted", False) for m in metas]),
        source_indices=np.array(idx, dtype=np.int64),
        source_names=np.array([names[i] for i in idx], dtype=str),
        antenna_indices=np.arange(nant), frequencies_hz=parts[0]["frequencies_hz"],
        times_mjd_seconds=np.concatenate([p["times_mjd_seconds"] for p in parts]),
        ms_names=np.array([m["ms"] for m in metas], dtype=str), jones_order=_JONES,
        metadata_json=np.array(json.dumps(md)))
    os.replace(tmp, output)
    return output


def merge_band_hour(band_dir: Path, stems: Optional[Iterable[str]], prefix: str) -> List[Path]:
    """Merge every pass of the band-hour's per-snapshot NPZs (those of `stems`; None = all present)
    into <band dir>/peeling_solutions/<prefix>_<pass>.npz; the merged parts are then removed.
    Nothing to merge (e.g. a rerun after the parts were merged): [] and the old files stay."""
    pdir = parts_dir(band_dir)
    if not pdir.is_dir():
        return []
    want = None if stems is None else set(stems)
    dirs = sorted(d for d in pdir.iterdir() if d.is_dir() and (want is None or d.name in want))
    written = []
    for p in PASSES:
        files = [d / f"{p}.npz" for d in dirs if (d / f"{p}.npz").exists()]
        if files:
            written.append(merge(files, Path(band_dir) / "peeling_solutions" / f"{prefix}_{p}.npz"))
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    try:
        pdir.rmdir()
    except OSError:
        pass                       # parts of snapshots outside `stems` stay
    return written


def peel_summary(files: List[Path]) -> Dict[str, Dict[str, int]]:
    """{"<pass>:<source>": {"selected": n, "subtracted": n}} over a band-hour's merged NPZs
    (a source counts as subtracted where it was accepted and its pass was not reverted)."""
    out: Dict[str, Dict[str, int]] = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            stage = json.loads(str(z["metadata_json"])).get("stage", Path(f).stem.rsplit("_", 1)[-1])
            kept = z["accepted"] & ~z["backstop_reverted"][None, :]
            for k, name in enumerate(z["source_names"].tolist()):
                out[f"{stage}:{name}"] = {"selected": int(z["solved"][k].sum()),
                                          "subtracted": int(kept[k].sum())}
    return out
