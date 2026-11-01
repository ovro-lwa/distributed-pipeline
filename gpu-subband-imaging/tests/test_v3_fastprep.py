"""gsi v3 in-memory prep: mstransform-exact averaging, equivalence with the v2 prep."""
import importlib.util
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_subband_imaging.stages import quality
from gpu_subband_imaging.v3 import fastprep

HAVE_CASACORE = importlib.util.find_spec("casacore") is not None
HAVE_CASA = HAVE_CASACORE and importlib.util.find_spec("casatasks") is not None
needs_casa = pytest.mark.skipif(not HAVE_CASA, reason="needs python-casacore and casatasks")


# -- averaging rules (pure numpy) ----------------------------------------------------

def _reference_average(data, flag, sigma, nb):
    """Element by element, float32 scalars: the mstransform rules of the module doc."""
    f32 = np.float32
    n, c, p = data.shape
    m = c // nb
    out = np.zeros((n, m, p), np.complex64)
    oflag = np.zeros((n, m, p), bool)
    wout = np.zeros((n, p), np.float32)
    sout = np.zeros((n, p), np.float32)
    for r in range(n):
        for q in range(p):
            wch = f32(1) / (sigma[r, q] * sigma[r, q])
            ws = np.zeros(m, np.float32)
            for j in range(m):
                re, im, cnt, w_ok, w_all = f32(0), f32(0), 0, f32(0), f32(0)
                re_a, im_a = f32(0), f32(0)
                for k in range(nb):
                    v = data[r, j * nb + k, q]
                    re_a, im_a, w_all = re_a + f32(v.real), im_a + f32(v.imag), w_all + wch
                    if not flag[r, j * nb + k, q]:
                        re, im, cnt, w_ok = re + f32(v.real), im + f32(v.imag), cnt + 1, w_ok + wch
                if cnt:
                    out[r, j, q] = complex(re / f32(cnt), im / f32(cnt))
                    ws[j] = w_ok
                else:
                    out[r, j, q] = complex(re_a / f32(nb), im_a / f32(nb))
                    ws[j] = w_all
                    oflag[r, j, q] = True
            use = ~oflag[r, :, q] if not oflag[r, :, q].all() else np.ones(m, bool)
            sw, ss = f32(0), f32(0)
            for j in range(m):
                if use[j]:
                    sw, ss = sw + ws[j], ss + f32(1) / np.sqrt(ws[j])
            wout[r, q], sout[r, q] = sw / f32(use.sum()), ss / f32(use.sum())
    return out, oflag, wout, sout


def _random_case(seed, n=7, c=12, p=4, flag_frac=0.4):
    rng = np.random.default_rng(seed)
    data = ((rng.normal(size=(n, c, p)) + 1j * rng.normal(size=(n, c, p))) * 37).astype(np.complex64)
    flag = rng.random((n, c, p)) < flag_frac
    flag[0] = True                               # every bin of a row flagged
    flag[1, :3] = True                           # one bin flagged in every correlation
    flag[2] = False                              # no flags in a row
    flag[3, :, 1] = False                        # nor in one correlation of a row
    flag[3, 6:, 2] = True                        # whole bins flagged, none partial
    sigma = rng.uniform(0.3, 2.5, size=(n, p)).astype(np.float32)
    return data, flag, sigma


@pytest.mark.parametrize("seed,nb", [(1, 3), (2, 4), (3, 2), (4, 6)])
def test_average_matches_reference_bitwise(seed, nb):
    data, flag, sigma = _random_case(seed)
    got = fastprep.average_like_mstransform(data, flag, sigma, nb)
    want = _reference_average(data, flag, sigma, nb)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape
        assert g.tobytes() == w.tobytes()


def test_average_by_hand():
    # one row, one correlation, chanbin 3: a partial bin and a fully flagged bin
    data = np.array([[[1 + 1j], [2 + 4j], [9 + 9j], [3 + 0j], [6 + 3j], [0 + 3j]]], np.complex64)
    flag = np.array([[[False], [False], [True], [True], [True], [True]]])
    sigma = np.array([[0.5]], np.float32)         # channel weight 4
    d, f, w, s = fastprep.average_like_mstransform(data, flag, sigma, 3)
    assert d[0, 0, 0] == np.complex64(1.5 + 2.5j)        # mean of the 2 unflagged
    assert d[0, 1, 0] == np.complex64(3 + 2j)            # all flagged: mean of all 3
    assert f[0, :, 0].tolist() == [False, True]
    assert w[0, 0] == np.float32(8)                      # 2 x 4, flagged bin left out
    assert s[0, 0] == np.float32(1) / np.sqrt(np.float32(8))
    # every bin flagged: the weights of all channels and bins
    d, f, w, s = fastprep.average_like_mstransform(data, np.ones_like(flag), sigma, 3)
    assert f.all() and w[0, 0] == np.float32(12)


def test_average_divides_real_and_imaginary_parts():
    # numpy complex64 / float multiplies by the reciprocal, mstransform divides each part
    data = np.zeros((1, 3, 1), np.complex64)
    data[0, :, 0] = [0.1 + 0.7j, 0.2 + 0.3j, 0.4 + 0.1j]
    d, *_ = fastprep.average_like_mstransform(data, np.zeros((1, 3, 1), bool),
                                              np.ones((1, 1), np.float32), 3)
    s = data[0, 0, 0] + data[0, 1, 0] + data[0, 2, 0]
    assert d[0, 0, 0].real == np.float32(s.real) / np.float32(3)
    assert d[0, 0, 0].imag == np.float32(s.imag) / np.float32(3)


def test_average_rejects_unsupported_input():
    data, flag, sigma = _random_case(5)
    with pytest.raises(ValueError, match="multiple"):
        fastprep.average_like_mstransform(data, flag, sigma, 5)
    bad = sigma.copy()
    bad[2, 1] = 0
    with pytest.raises(ValueError, match="SIGMA"):
        fastprep.average_like_mstransform(data, flag, bad, 4)
    with pytest.raises(TypeError):
        fastprep.average_like_mstransform(data.astype(np.complex128), flag, sigma, 4)


def test_average_channels():
    f = 22.5e6 + 23925.78125 * np.arange(8)
    w = np.full(8, 23925.78125)
    freqs, widths = fastprep.average_channels(f, w, 4)
    np.testing.assert_array_equal(freqs, [f[:4].mean(), f[4:].mean()])
    np.testing.assert_array_equal(widths, [4 * 23925.78125] * 2)


# -- quality stats on arrays ---------------------------------------------------------------

class _FakeTable:
    def __init__(self, data, flags):
        self.data, self.flags = data, flags

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def __len__(self):
        return self.data.shape[0]

    def getcol(self, column, startrow, nrow):
        return (self.flags if column == "FLAG" else self.data)[startrow:startrow + nrow]


@pytest.mark.parametrize("nrow,sample", [(1000, 120), (50, 1024), (7, 3)])
def test_visibility_stats_arrays_equals_ms_version(monkeypatch, nrow, sample):
    rng = np.random.default_rng(nrow)
    data = (rng.normal(size=(nrow, 6, 4)) + 1j * rng.normal(size=(nrow, 6, 4))).astype(np.complex64)
    data[: nrow // 3] = 0
    data[nrow // 2, 0, 0] = np.nan
    flags = rng.random(data.shape) < 0.3
    tables = SimpleNamespace(table=lambda *a, **k: _FakeTable(data, flags))
    monkeypatch.setitem(sys.modules, "casacore", SimpleNamespace(tables=tables))
    monkeypatch.setitem(sys.modules, "casacore.tables", tables)
    want = quality.visibility_stats("x.ms", "DATA", sample, 1e-12)
    got = quality.visibility_stats_arrays(data, flags, sample, 1e-12)
    assert got == want


# -- against the v2 prep on a synthetic MS (python-casacore + CASA mstransform) -----------

def _v2_prep(raw, work, badants, chanbin, mode):
    from gpu_subband_imaging.stages import fastcal, uvw
    v2 = work / "v2_raw.ms"
    shutil.copytree(raw, v2)
    fastcal.flag_calibrate(v2, "bp", "xy", badants)
    avg = fastcal.average(v2, work / "v2_avg.ms", chanbin)
    if mode != "off":
        uvw.fix_uvw(avg, epoch=mode)
    return avg


def _patch_jones(monkeypatch, nant, nchan):
    from gpu_subband_imaging.stages import fastcal
    J, dead = __import__("v3_synth").jones_for(nant, nchan)
    monkeypatch.setattr(fastcal, "jones", lambda ms, bp, xy, n, freq: (J, dead))


@needs_casa
@pytest.mark.parametrize("case", [
    dict(f0=54.74e6, mode="centroid", weights=False),
    dict(f0=84.95e6, mode="time", weights=False),     # 85 MHz edge inside a bin; rephasing
    dict(f0=54.74e6, mode="off", weights=True),       # non-uniform WEIGHT -> SIGMA
])
def test_prep_arrays_equals_v2_prep(tmp_path, monkeypatch, case):
    from casacore.tables import table
    from v3_synth import INTERVAL, make_raw_ms
    nant, nchan, chanbin, bad = 10, 16, 4, "2,7"
    raw = tmp_path / "raw.ms"
    make_raw_ms(raw, nant=nant, nchan=nchan, f0=case["f0"], time_centroid_offset=INTERVAL / 2)
    if case["weights"]:
        with table(str(raw), readonly=False, ack=False) as t:
            rng = np.random.default_rng(9)
            t.putcol("WEIGHT", rng.uniform(0.2, 3.0, (t.nrows(), 4)).astype(np.float32))
    _patch_jones(monkeypatch, nant, nchan)
    avg = _v2_prep(raw, tmp_path, bad, chanbin, case["mode"])

    snap = fastprep.prep_arrays(str(raw), "bp", "xy", bad, chanbin, case["mode"])
    c = fastprep.compare_snap_ms(snap, str(avg))
    assert c["identical"], c
    assert snap.meta["fix_uvw_error"] is None
    if case["mode"] == "time":
        assert snap.meta["fix_uvw"]["rephased"]
    assert snap.flag.any() and not snap.flag.all()

    # write_ms_like from a template of *another* snapshot: same structure and values as v2
    from gpu_subband_imaging.v3.mscompare import table_diff
    other = tmp_path / "other_raw.ms"
    make_raw_ms(other, nant=nant, nchan=nchan, f0=case["f0"], seed=11)
    tpl = fastprep.make_template(str(other), str(tmp_path / "tpl.ms"), chanbin)
    out = fastprep.write_ms_like(str(tpl), str(raw), snap, str(tmp_path / "v3_avg.ms"))
    assert table_diff(str(avg), str(out)) == []
    assert fastprep.compare_snap_ms(snap, str(out))["identical"]


@needs_casa
def test_write_ms_like_refuses_wrong_template(tmp_path, monkeypatch):
    from v3_synth import make_raw_ms
    raw, other = tmp_path / "raw.ms", tmp_path / "other.ms"
    make_raw_ms(raw, nchan=16, f0=54.74e6)
    make_raw_ms(other, nchan=16, f0=60.0e6)
    _patch_jones(monkeypatch, 10, 16)
    snap = fastprep.prep_arrays(str(raw), "bp", "xy", "", 4, "centroid")
    tpl = fastprep.make_template(str(other), str(tmp_path / "tpl.ms"), 4)
    with pytest.raises(ValueError, match="SPECTRAL_WINDOW"):
        fastprep.write_ms_like(str(tpl), str(raw), snap, str(tmp_path / "out.ms"))
    assert not (tmp_path / "out.ms").exists()


@needs_casa
def test_prep_task_modes(tmp_path, monkeypatch):
    from v3_synth import make_raw_ms
    raw = tmp_path / "raw.ms"
    make_raw_ms(raw)
    _patch_jones(monkeypatch, 10, 16)
    args = (str(raw), "bp", "xy", "3", 4, "centroid", 1024, 1e-12)
    shadow = fastprep.prep_task(*args)
    assert "snap" in shadow and set(shadow["meta"]["timings"]) >= {"read", "cal", "average", "uvw", "total"}
    tpl = fastprep.make_template(str(raw), str(tmp_path / "tpl.ms"), 4)
    on = fastprep.prep_task(*args, str(tpl), str(tmp_path / "out.ms"))
    assert "snap" not in on and "write" in on["meta"]["timings"]
    assert on["stats"] == shadow["stats"]
    assert fastprep.compare_snap_ms(shadow["snap"], str(tmp_path / "out.ms"))["identical"]


@needs_casa
def test_prep_arrays_keeps_stored_uvw_when_fix_fails(tmp_path, monkeypatch):
    from casacore.tables import table
    from v3_synth import make_raw_ms
    raw = tmp_path / "raw.ms"
    make_raw_ms(raw)
    with table(str(raw / "FIELD"), readonly=False, ack=False) as f:   # PHASE_DIR off by 1 deg
        pd = f.getcol("PHASE_DIR")
        pd[0, 0, 1] += np.radians(1.0)
        f.putcol("PHASE_DIR", pd)
    with table(str(raw), ack=False) as t:
        stored = t.getcol("UVW")
    _patch_jones(monkeypatch, 10, 16)
    snap = fastprep.prep_arrays(str(raw), "bp", "xy", "", 4, "centroid")
    assert "disagree" in snap.meta["fix_uvw_error"]
    np.testing.assert_array_equal(snap.uvw, stored)
