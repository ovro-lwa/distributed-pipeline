"""Night-time averaged inputs, peeling-solution export and the backlog config checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpu_subband_imaging.config import ConfigError, PeelParams
from gpu_subband_imaging.stages import peel_solutions
from gpu_subband_imaging.worker import _stamp


def test_stamp_of_averaged_input():
    p = "/lustre/pipeline/night-time/averaged/55MHz/2024-12-20/03/20241220_030005_55MHz_averaged.ms"
    assert _stamp(p) == "20241220_030005_55MHz"
    assert _stamp("/x/20261026_031144_55MHz.ms.tar") == "20261026_031144_55MHz"


def test_save_solutions_validation():
    assert PeelParams(save_solutions="sky, rfi").solution_passes == ["sky", "rfi"]
    assert PeelParams().solution_passes == []
    assert PeelParams(save_solutions="sky").daemon_env(None)["ZEST_SAVE_SOLUTIONS"] == "sky"
    with pytest.raises(ConfigError):
        PeelParams(save_solutions="sky,cal")


def _daemon_output(d: Path, names, gains, accepted, elev=(30.0, 12.5)):
    """What zest_daemon.jl write_solutions leaves for one pass."""
    d.mkdir(parents=True, exist_ok=True)
    gains.astype("<c8").ravel(order="F").tofile(d / "sky.bin")
    lines = ["# shape " + " ".join(map(str, gains.shape)),
             "index\tname\televation_deg\taccepted\tconverged\tniters"]
    for k, (n, a) in enumerate(zip(names, accepted)):
        lines.append(f"{k * 2}\t{n}\t{elev[k]:.4f}\t{str(a).lower()}\ttrue\t{5 + k}")
    (d / "sky.tsv").write_text("\n".join(lines) + "\n")


def test_convert_and_merge(tmp_path):
    rng = np.random.default_rng(1)
    nant, nfreq = 3, 4
    freq = np.linspace(50e6, 51e6, nfreq)
    g1 = (rng.normal(size=(2, 4, nant, nfreq)) + 1j * rng.normal(size=(2, 4, nant, nfreq))).astype(np.complex64)
    g2 = (rng.normal(size=(1, 4, nant, nfreq)) + 1j * rng.normal(size=(1, 4, nant, nfreq))).astype(np.complex64)
    kept = []
    for t, (names, g, acc) in enumerate(((["Cas A", "Cyg A"], g1, [True, False]), (["Cas A"], g2, [True]))):
        ms = tmp_path / f"2024122{t}_030005_55MHz.ms"
        ms.mkdir()
        sol = Path(f"{ms}.peelsol")
        _daemon_output(sol, names, g, acc)
        out = peel_solutions.sol_name(ms)
        files = peel_solutions.convert(sol, ms, out, {"band": "55MHz"}, axes=(freq, 1000.0 + 10 * t))
        assert [f.name for f in files] == ["sky.npz"] and not sol.exists()
        with np.load(files[0]) as z:
            np.testing.assert_array_equal(z["gains"][..., 0], g)
            assert z["gains"].shape == g.shape + (1,)
            assert z["accepted"].tolist() == acc
            assert json.loads(str(z["metadata_json"]))["schema_version"] == 1
        kept.append(str(ms))
    merged = peel_solutions.merge_band_hour(kept, tmp_path / "peeling_solutions", "55MHz_x_03h")
    assert [p.name for p in merged] == ["55MHz_x_03h_sky.npz"]
    with np.load(merged[0]) as z:
        assert z["source_names"].tolist() == ["Cas A", "Cyg A"]
        assert z["gains"].shape == (2, 4, nant, nfreq, 2)
        np.testing.assert_array_equal(z["gains"][0, ..., 1], g2[0])
        assert np.isnan(z["gains"][1, ..., 1]).all()
        assert z["solved"].tolist() == [[True, True], [True, False]]
        assert z["accepted"].tolist() == [[True, True], [False, False]]
        assert z["niters"][0].tolist() == [5, 5]
        assert z["times_mjd_seconds"].tolist() == [1000.0, 1010.0]


def test_backlog_window():
    import importlib.util
    from datetime import datetime, timezone
    spec = importlib.util.spec_from_file_location(
        "run_backlog", Path(__file__).resolve().parents[1] / "scripts" / "run_backlog.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    t = lambda hh, mm: datetime(2026, 10, 29, hh, mm, tzinfo=timezone.utc)
    assert rb.may_start(t(16, 0), "05:30", "15:00", 60)
    assert rb.may_start(t(4, 0), "05:30", "15:00", 60)
    assert not rb.may_start(t(4, 45), "05:30", "15:00", 60)     # would run into the nightly
    assert not rb.may_start(t(9, 0), "05:30", "15:00", 60)      # nightly hours
