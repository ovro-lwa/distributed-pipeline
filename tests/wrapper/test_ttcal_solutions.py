"""Solution persistence tests; no CASA, Julia, broker, or worker required."""
import json
from pathlib import Path

import numpy as np
import pytest

from orca.wrapper import ttcal


def fake_solver(cmd, **kwargs):
    # The last argument is the private exchange directory, not the final output.
    exchange = Path(cmd[-1])
    gains = np.arange(32, dtype=np.float64).reshape(1, 4, 2, 2, 2) + 1j
    gains.ravel(order='F').astype('<c16').tofile(exchange / 'gains.bin')
    (exchange / 'metadata.json').write_text(json.dumps({
        'shape': [1, 4, 2, 2, 2], 'source_indices': [1],
        'source_names': ['visible'], 'frequencies_hz': [40e6, 41e6],
        'times_mjd_seconds': [5e9, 5e9 + 10], 'column': 'CORRECTED_DATA',
    }))


def test_save_compressed_solutions_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(ttcal.subprocess, 'run', fake_solver)
    sources = tmp_path / 'sources.json'
    sources.write_text('[{"name":"below horizon"},{"name":"visible"}]')
    output = tmp_path / 'solutions' / 'sky.npz'
    assert ttcal.zest_with_ttcal('input.ms', str(sources), solutions_path=str(output)) == 'input.ms'
    with np.load(output, allow_pickle=False) as saved:
        assert saved['gains'].shape == (1, 4, 2, 2, 2)
        assert saved['gains'][0, 3, 1, 1, 1] == 31 + 1j
        assert saved['source_indices'].tolist() == [1]
        assert saved['source_names'].tolist() == ['visible']
        assert saved['jones_order'].tolist() == ['xx', 'xy', 'yx', 'yy']
        assert saved['invalid_gains'].sum() == 0
        metadata = json.loads(str(saved['metadata_json']))
        assert metadata['sources'][1]['name'] == 'visible'
        assert metadata['maxiter'] == 30
    assert sorted(p.name for p in output.parent.iterdir()) == ['sky.npz']


def test_failed_solver_does_not_publish_partial_output(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ttcal.subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(ttcal.subprocess, 'run', fail)
    sources = tmp_path / 'sources.json'
    sources.write_text('[]')
    output = tmp_path / 'solutions' / 'rfi.npz'
    with pytest.raises(ttcal.subprocess.CalledProcessError):
        ttcal.zest_with_ttcal('input.ms', str(sources), solutions_path=str(output))
    assert not output.exists()
    assert not list(output.parent.iterdir())


def test_empty_visible_source_list_is_explicit(tmp_path, monkeypatch):
    def empty(cmd, **kwargs):
        exchange = Path(cmd[-1])
        (exchange / 'gains.bin').write_bytes(b'')
        (exchange / 'metadata.json').write_text(json.dumps({
            'shape': [0, 4, 8, 12, 1], 'source_indices': [], 'source_names': [],
            'frequencies_hz': list(range(12)), 'times_mjd_seconds': [5e9], 'column': 'DATA',
        }))
    monkeypatch.setattr(ttcal.subprocess, 'run', empty)
    sources = tmp_path / 'sources.json'
    sources.write_text('[]')
    output = tmp_path / 'sky.npz'
    ttcal.zest_with_ttcal('input.ms', str(sources), solutions_path=str(output))
    with np.load(output, allow_pickle=False) as saved:
        assert saved['gains'].shape == (0, 4, 8, 12, 1)
        assert saved['source_indices'].size == 0
        assert saved['source_names'].dtype.kind == 'U'


def test_malformed_exchange_preserves_previous_output(tmp_path, monkeypatch):
    def truncated(cmd, **kwargs):
        fake_solver(cmd, **kwargs)
        (Path(cmd[-1]) / 'gains.bin').write_bytes(b'')
    monkeypatch.setattr(ttcal.subprocess, 'run', truncated)
    sources = tmp_path / 'sources.json'
    sources.write_text('[]')
    output = tmp_path / 'sky.npz'
    output.write_bytes(b'previous valid output')
    with pytest.raises(ValueError):
        ttcal.zest_with_ttcal('input.ms', str(sources), solutions_path=str(output))
    assert output.read_bytes() == b'previous valid output'
    assert not list(tmp_path.glob('.peelsol-*'))


def test_ttcal_dev_prefers_consistent_casacore(tmp_path, monkeypatch):
    seen = {}

    def solver(cmd, env=None, **kwargs):
        seen['env'] = env
        exchange = Path(cmd[-1].rsplit(' ', 1)[-1].strip("'"))
        fake_solver([str(exchange)])

    monkeypatch.setattr(ttcal.subprocess, 'run', solver)
    monkeypatch.setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    sources = tmp_path / 'sources.json'
    sources.write_text('[{"name":"below horizon"},{"name":"visible"}]')
    ttcal.zest_with_ttcal('input.ms', str(sources), julia_env='ttcal_dev',
                          solutions_path=str(tmp_path / 'rfi.npz'))
    assert seen['env']['LD_LIBRARY_PATH'] == '/opt/lib:/usr/local/cuda/lib64'
