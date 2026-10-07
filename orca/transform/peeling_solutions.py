"""Consolidate per-MS TTCal peeling solutions into one NPZ per stage.

Phase 1 writes ``peeling_solutions/<ms>/{sky,rfi}.npz`` (schema 1) for every
integration.  Before archiving, Phase 2 merges them into
``peeling_solutions/<prefix>_{sky,rfi}.npz`` (schema 2): one file per stage
per subband-hour, complex64 gains (the MS visibility precision), and the
source model stored once instead of in every file.
"""
import os
import glob
import json
import shutil
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

STAGES = ('sky', 'rfi')
# Keys that differ between integrations; everything else in metadata is shared.
_PER_MS_METADATA = ('ms', 'source_indices', 'source_names',
                    'times_mjd_seconds', 'column')


def _load_per_ms(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {
            'gains': z['gains'],
            'source_indices': z['source_indices'].astype(np.int64),
            'source_names': z['source_names'].astype(str),
            'frequencies_hz': z['frequencies_hz'],
            'times_mjd_seconds': z['times_mjd_seconds'],
            'metadata': json.loads(str(z['metadata_json'])),
        }


def merge_stage(files: List[str], output: str) -> str:
    """Merge schema-1 per-MS NPZ files of one stage into one schema-2 NPZ.

    ``gains`` keeps the schema-1 axes ``(source, jones, antenna, frequency,
    time)`` with time concatenated over integrations and sources as the union
    over the hour.  Sources below the horizon for an integration are NaN and
    ``solved[source, time]`` is False.

    Raises:
        ValueError: if antenna or frequency axes differ between files.
    """
    parts = sorted((_load_per_ms(f) for f in files),
                   key=lambda p: float(p['times_mjd_seconds'][0]))
    first = parts[0]
    nant, nfreq = first['gains'].shape[2:4]
    for p in parts:
        if p['gains'].shape[2:4] != (nant, nfreq) or not np.array_equal(
                p['frequencies_hz'], first['frequencies_hz']):
            raise ValueError(f"Antenna/frequency axes differ in {p['metadata'].get('ms')}")

    names = {}
    for p in parts:
        names.update(zip(p['source_indices'].tolist(), p['source_names'].tolist()))
    source_indices = np.array(sorted(names), dtype=np.int64)
    row = {idx: i for i, idx in enumerate(source_indices.tolist())}
    ntimes = [p['gains'].shape[4] for p in parts]

    gains = np.full((len(source_indices), 4, nant, nfreq, sum(ntimes)),
                    np.nan + 1j * np.nan, dtype=np.complex64)
    solved = np.zeros((len(source_indices), sum(ntimes)), dtype=bool)
    t0 = 0
    for p, nt in zip(parts, ntimes):
        for k, idx in enumerate(p['source_indices'].tolist()):
            gains[row[idx], ..., t0:t0 + nt] = p['gains'][k]
            solved[row[idx], t0:t0 + nt] = True
        t0 += nt

    metadata = {k: v for k, v in first['metadata'].items() if k not in _PER_MS_METADATA}
    metadata.update(
        schema_version=2, gains_dtype='complex64', n_ms=len(parts),
        columns=sorted({p['metadata'].get('column', '') for p in parts}),
        gain_axes=['source', 'jones', 'antenna', 'frequency', 'time'],
    )
    ms_names = np.repeat([p['metadata'].get('ms', '') for p in parts], ntimes)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    pending = output + '.tmp.npz'
    np.savez_compressed(
        pending, gains=gains, solved=solved,
        source_indices=source_indices,
        source_names=np.array([names[i] for i in source_indices.tolist()], dtype=str),
        antenna_indices=np.arange(nant),
        frequencies_hz=first['frequencies_hz'].astype(np.float64),
        times_mjd_seconds=np.concatenate([p['times_mjd_seconds'] for p in parts]).astype(np.float64),
        ms_names=ms_names.astype(str),
        jones_order=np.array(['xx', 'xy', 'yx', 'yy']),
        metadata_json=np.array(json.dumps(metadata)),
    )
    os.replace(pending, output)
    return output


def consolidate_peeling_solutions(solutions_root: str, prefix: str) -> List[str]:
    """Merge every stage under *solutions_root* and remove the per-MS dirs.

    Per-MS files are only removed after all stages were written; on any
    error they are left in place (and archived as-is) and the error is logged.

    Returns:
        Paths of the consolidated files (empty if nothing to merge or on error).
    """
    ms_dirs = sorted(d for d in glob.glob(os.path.join(solutions_root, '*'))
                     if os.path.isdir(d))
    if not ms_dirs:
        return []
    written = []
    try:
        for stage in STAGES:
            files = [os.path.join(d, f'{stage}.npz') for d in ms_dirs
                     if os.path.isfile(os.path.join(d, f'{stage}.npz'))]
            if files:
                out = os.path.join(solutions_root, f'{prefix}_{stage}.npz')
                written.append(merge_stage(files, out))
                logger.info(f"Peeling solutions: merged {len(files)} {stage} files → "
                            f"{os.path.basename(out)} ({os.path.getsize(out) / 1e6:.1f} MB)")
    except Exception as e:
        logger.error(f"Peeling solution consolidation failed, keeping per-MS files: {e}")
        for f in written:
            os.remove(f)
        return []
    for d in ms_dirs:
        shutil.rmtree(d, ignore_errors=True)
    return written
