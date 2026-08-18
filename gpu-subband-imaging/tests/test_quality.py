import sys
from types import SimpleNamespace

import numpy as np

from gpu_subband_imaging.stages.quality import visibility_stats


class FakeTable:
    def __init__(self, data, flags):
        self.data = data
        self.flags = flags

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __len__(self):
        return self.data.shape[0]

    def getcol(self, column, startrow, nrow):
        values = self.flags if column == "FLAG" else self.data
        return values[startrow:startrow + nrow]


def install_fake_casacore(monkeypatch, data, flags):
    tables = SimpleNamespace(table=lambda *args, **kwargs: FakeTable(data, flags))
    monkeypatch.setitem(sys.modules, "casacore", SimpleNamespace(tables=tables))
    monkeypatch.setitem(sys.modules, "casacore.tables", tables)


def test_visibility_stats_detects_zero_input(monkeypatch, tmp_path):
    data = np.zeros((30, 4, 2), dtype=np.complex64)
    flags = np.zeros_like(data, dtype=bool)
    install_fake_casacore(monkeypatch, data, flags)

    stats = visibility_stats(tmp_path / "zero.ms", "DATA", 12, 1e-12)

    assert stats.usable_values > 0
    assert stats.nonzero_fraction == 0.0
    assert stats.max_amplitude == 0.0


def test_visibility_stats_ignores_flags_and_samples_real_data(monkeypatch, tmp_path):
    data = np.ones((30, 4, 2), dtype=np.complex64)
    flags = np.zeros_like(data, dtype=bool)
    flags[:, :2, :] = True
    install_fake_casacore(monkeypatch, data, flags)

    stats = visibility_stats(tmp_path / "valid.ms", "DATA", 12, 1e-12)

    assert stats.usable_values > 0
    assert stats.nonzero_fraction == 1.0
    assert stats.median_amplitude == 1.0
