"""Merging per-MS peeling solutions into one NPZ per stage (no CASA/Julia)."""
import json

import numpy as np
import pytest

from orca.transform.peeling_solutions import consolidate_peeling_solutions

NANT, NFREQ = 3, 2
FREQS = np.array([50e6, 50.1e6])


def _write_per_ms(root, ms, stage, t, source_indices, fill, freqs=FREQS):
    d = root / ms
    d.mkdir(parents=True, exist_ok=True)
    n = len(source_indices)
    gains = np.full((n, 4, NANT, len(freqs), 1), fill, dtype=np.complex128)
    for k in range(n):
        gains[k] += k
    meta = {'ms': ms, 'column': 'CORRECTED_DATA', 'sources': [{'name': 'big model'}],
            'maxiter': 5, 'source_indices': source_indices,
            'source_names': [f's{i}' for i in source_indices],
            'times_mjd_seconds': [t]}
    np.savez_compressed(
        d / f'{stage}.npz', gains=gains,
        source_indices=np.array(source_indices, dtype=np.int64),
        source_names=np.array([f's{i}' for i in source_indices], dtype=str),
        frequencies_hz=freqs, times_mjd_seconds=np.array([t], dtype=np.float64),
        metadata_json=np.array(json.dumps(meta)))


def test_merge_unions_sources_and_removes_per_ms_dirs(tmp_path):
    root = tmp_path / 'peeling_solutions'
    # Listed out of time order; source 0 sets before the second integration.
    _write_per_ms(root, 'b.ms', 'sky', 20.0, [1], 2 + 0j)
    _write_per_ms(root, 'a.ms', 'sky', 10.0, [0, 1], 1 + 1j)
    _write_per_ms(root, 'a.ms', 'rfi', 10.0, [3], 5 + 0j)
    _write_per_ms(root, 'b.ms', 'rfi', 20.0, [], 0j)

    out = consolidate_peeling_solutions(str(root), '50MHz')

    assert sorted(p.name for p in root.iterdir()) == ['50MHz_rfi.npz', '50MHz_sky.npz']
    assert len(out) == 2
    with np.load(root / '50MHz_sky.npz', allow_pickle=False) as z:
        assert z['gains'].dtype == np.complex64
        assert z['gains'].shape == (2, 4, NANT, NFREQ, 2)
        assert z['source_indices'].tolist() == [0, 1]
        assert z['source_names'].tolist() == ['s0', 's1']
        assert z['times_mjd_seconds'].tolist() == [10.0, 20.0]
        assert z['ms_names'].tolist() == ['a.ms', 'b.ms']
        assert z['solved'].tolist() == [[True, False], [True, True]]
        assert z['gains'][0, 0, 0, 0, 0] == 1 + 1j        # s0 at t=10
        assert np.isnan(z['gains'][0, ..., 1]).all()        # s0 set at t=20
        assert z['gains'][1, 0, 0, 0, 0] == 2 + 1j          # s1 at t=10 (k=1 offset)
        assert z['gains'][1, 0, 0, 0, 1] == 2 + 0j          # s1 at t=20
        meta = json.loads(str(z['metadata_json']))
        assert meta['schema_version'] == 2
        assert meta['n_ms'] == 2
        assert meta['sources'] == [{'name': 'big model'}]
        assert 'ms' not in meta and 'source_indices' not in meta
    with np.load(root / '50MHz_rfi.npz', allow_pickle=False) as z:
        assert z['gains'].shape == (1, 4, NANT, NFREQ, 2)
        assert z['solved'].tolist() == [[True, False]]


def test_merge_failure_keeps_per_ms_files(tmp_path):
    root = tmp_path / 'peeling_solutions'
    _write_per_ms(root, 'a.ms', 'sky', 10.0, [0], 1 + 0j)
    _write_per_ms(root, 'b.ms', 'sky', 20.0, [0], 1 + 0j, freqs=FREQS + 1)

    assert consolidate_peeling_solutions(str(root), '50MHz') == []
    assert sorted(p.name for p in root.iterdir()) == ['a.ms', 'b.ms']


def test_nothing_to_merge(tmp_path):
    root = tmp_path / 'peeling_solutions'
    root.mkdir()
    assert consolidate_peeling_solutions(str(root), '50MHz') == []
