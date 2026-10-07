"""Requires the normal pipeline Python environment, but no Celery broker."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest


def test_solution_archive_survives_workdir_cleanup(tmp_path):
    pytest.importorskip('casacore.tables')
    # Load the working-tree implementation even when orca is installed elsewhere.
    source = Path(__file__).resolve().parents[2] / 'orca/transform/subband_processing.py'
    spec = importlib.util.spec_from_file_location('subband_processing_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / 'work'
    solutions = work / 'peeling_solutions' / 'integration.ms'
    solutions.mkdir(parents=True)
    np.savez_compressed(solutions / 'sky.npz', gains=np.array([1+2j]))
    archive = tmp_path / 'archive'
    module.archive_results(str(work), str(archive), cleanup_workdir=True)
    assert not work.exists()
    with np.load(archive / 'peeling_solutions/integration.ms/sky.npz') as saved:
        assert saved['gains'].tolist() == [1+2j]
