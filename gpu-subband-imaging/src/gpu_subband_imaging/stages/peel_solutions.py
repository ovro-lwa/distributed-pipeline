"""Peeling solutions (peel.save_solutions): per-MS NPZ and one merged NPZ per band-hour.

workers/zest_daemon.jl writes <ms>.peelsol/<pass>.bin + <pass>.tsv; :func:`convert`
turns them into <out>/<pass>.npz with the main pipeline's schema 1
(orca/wrapper/ttcal.py: gains (source, jones, antenna, frequency, time), source_indices,
source_names, frequencies_hz, times_mjd_seconds, antenna_indices, jones_order,
metadata_json) plus per-source elevation_deg, accepted, converged and niters.  Gains
are complex64, the MS visibility precision.  :func:`merge` combines one pass of a
band-hour into schema 2 (orca/transform/peeling_solutions.py: time concatenated,
sources as the union, NaN gains and solved=False where a source was not peeled) with
the per-source arrays as (source, time).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

PASSES = ("sky", "rfi")
_JONES = np.array(["xx", "xy", "yx", "yy"])


def _read_tsv(path: Path):
    lines = path.read_text().splitlines()
    shape = tuple(int(x) for x in lines[0].split()[2:])
    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    return shape, rows


def _ms_axes(ms: Path):
    from casacore.tables import table
    with table(str(ms) + "/SPECTRAL_WINDOW", ack=False) as s:
        freq = np.asarray(s.getcell("CHAN_FREQ", 0), dtype=np.float64)
    with table(str(ms), ack=False) as t:
        return freq, float(t.getcell("TIME", 0))


def convert(sol_dir: Path, ms: Path, out_dir: Path, meta: Dict, axes=None) -> List[Path]:
    """Daemon output of one MS -> <out_dir>/<pass>.npz; removes sol_dir.  Returns the files.

    axes: (channel frequencies [Hz], TIME [MJD s]); read from the MS when None."""
    sol_dir, out_dir = Path(sol_dir), Path(out_dir)
    if not sol_dir.is_dir():
        return []
    freq, time_s = axes if axes is not None else _ms_axes(ms)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in PASSES:
        tsv, binf = sol_dir / f"{p}.tsv", sol_dir / f"{p}.bin"
        if not tsv.exists():
            continue
        shape, rows = _read_tsv(tsv)
        g = np.fromfile(binf, dtype="<c8").reshape(shape, order="F")[..., None]
        if g.shape[3] != len(freq):
            raise ValueError(f"{binf}: {g.shape[3]} channels, MS has {len(freq)}")
        md = dict(meta, schema_version=1, ms=Path(ms).name, stage=p, routine="zest",
                  gain_axes=["source", "jones", "antenna", "frequency", "time"],
                  gains_dtype="complex64", convergence_flags_available=True,
                  backstop_reverted=(sol_dir / f"{p}.reverted").exists())
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
            metadata_json=np.array(json.dumps(md)))
        os.replace(tmp, out)
        written.append(out)
    shutil.rmtree(sol_dir, ignore_errors=True)
    return written


def merge(files: List[Path], output: Path) -> Path:
    """Merge one pass's per-MS NPZ files (schema 1) into one schema-2 NPZ."""
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


def sol_name(kept_ms: Path) -> Path:
    """Directory of a kept MS's solutions: <kept dir>/<stamp>.peelsol/ beside <stamp>.ms."""
    kept_ms = Path(kept_ms)
    return kept_ms.parent / (kept_ms.name[:-3] + ".peelsol")


def merge_band_hour(kept_ms: List[str], out_dir: Path, prefix: str) -> List[Path]:
    """Merge every pass of a band-hour's kept MSes into <out_dir>/<prefix>_<pass>.npz."""
    written = []
    for p in PASSES:
        files = [sol_name(Path(m)) / f"{p}.npz" for m in kept_ms]
        files = [f for f in files if f.exists()]
        if files:
            written.append(merge(files, Path(out_dir) / f"{prefix}_{p}.npz"))
    return written
