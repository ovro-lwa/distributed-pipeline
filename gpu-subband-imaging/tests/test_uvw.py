"""UVW recomputation (snapshot astrometry fix) and its worker/config wiring."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import Config, ConfigError
from gpu_subband_imaging.stages import uvw
from gpu_subband_imaging.worker import Worker

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"


def _rot_w(x, deg):
    a = np.radians(deg)
    out = x.copy()
    out[:, 0] = np.cos(a) * x[:, 0] - np.sin(a) * x[:, 1]
    out[:, 1] = np.sin(a) * x[:, 0] + np.cos(a) * x[:, 1]
    return out


def test_rotation_about_w_recovers_angle():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 3)) * 1000
    assert uvw.rotation_about_w_deg(x, _rot_w(x, 0.34)) == pytest.approx(0.34, abs=1e-9)
    assert uvw.rotation_about_w_deg(x, x) == pytest.approx(0.0, abs=1e-12)


# -- config ------------------------------------------------------------------

def test_fix_uvw_defaults_to_centroid():
    assert Config.load(CONFIG).pipeline.fix_uvw == "centroid"


def _pipeline_with(tmp_path, value):
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    d["fix_uvw"] = value
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_fix_uvw_yaml_off_and_time(tmp_path):
    p = tmp_path / "pipeline.yaml"
    text = (CONFIG / "pipeline.yaml").read_text().replace("fix_uvw: centroid", "fix_uvw: off")
    p.write_text(text)
    assert config_mod.load_pipeline(p).fix_uvw == "off"   # bare YAML off -> False -> "off"
    assert config_mod.load_pipeline(_pipeline_with(tmp_path, "time")).fix_uvw == "time"


def test_fix_uvw_rejects_unknown_mode(tmp_path):
    with pytest.raises(ConfigError, match="fix_uvw"):
        config_mod.load_pipeline(_pipeline_with(tmp_path, "geometric"))


# -- worker hook ---------------------------------------------------------------

def _worker(tmp_path, mode):
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(date="2026-10-24", output_root=str(tmp_path / "out"),
                                 movie=SimpleNamespace(), fix_uvw=mode,
                                 cal=SimpleNamespace(bp="bp", xy="xy")),
        cluster=SimpleNamespace(state_dir=str(tmp_path / "s"), work_root=str(tmp_path / "w")),
        subbands=SimpleNamespace(geom=lambda band: SimpleNamespace()),
    )
    return Worker(cfg, 55, 0, 0)


def test_worker_fix_uvw_modes(tmp_path, monkeypatch):
    calls = []

    def fake(ms, epoch):
        calls.append((ms, epoch))
        return uvw.UVWFixStats(1, -0.34, 3.5, 1e-5, False)

    monkeypatch.setattr(uvw, "fix_uvw", fake)
    _worker(tmp_path, "off")._fix_uvw("s", Path("a.ms"))
    assert calls == []
    _worker(tmp_path, "centroid")._fix_uvw("s", Path("a.ms"))
    _worker(tmp_path, "time")._fix_uvw("s", Path("b.ms"))
    assert calls == [(Path("a.ms"), "centroid"), (Path("b.ms"), "time")]


def test_worker_fix_uvw_failure_keeps_snapshot(tmp_path, monkeypatch):
    def boom(ms, epoch):
        raise ValueError("PHASE_DIR and stored UVW disagree")

    monkeypatch.setattr(uvw, "fix_uvw", boom)
    _worker(tmp_path, "centroid")._fix_uvw("s", Path("a.ms"))  # logs, does not raise


# -- regression on a synthetic OVRO-LWA-like snapshot MS (needs casacore) ----

OVRO = np.array([-2409150.0, -4478573.0, 3838617.0])     # ITRF, m
T_START = 5299554608.706455                               # raw MS TIME (integration start)
INTERVAL = 10.031020408163265


def _make_ms(path, stored_rot_deg):
    tables = pytest.importorskip("casacore.tables")
    from casacore.measures import measures
    from casacore.quanta import quantity as q

    rng = np.random.default_rng(2)
    nant = 12
    enu = rng.uniform(-600, 600, size=(nant, 3)) * [1, 1, 0.003]
    lat, lon = np.radians(37.2398), np.radians(-118.2817)
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.cross(east, north)
    pos = OVRO + enu[:, :1] * east + enu[:, 1:2] * north + enu[:, 2:] * up
    tc = T_START + INTERVAL / 2
    me = measures()
    me.do_frame(me.epoch("UTC", q(tc, "s")))
    me.do_frame(me.position("ITRF", *[q(x, "m") for x in pos.mean(0)]))
    z = me.measure(me.direction("AZEL", q(0, "deg"), q(90, "deg")), "J2000")
    ra, dec = z["m0"]["value"], z["m1"]["value"]

    a1, a2 = np.triu_indices(nant)
    A = uvw.antenna_uvw(pos, tc, ra, dec)
    true = A[a1] - A[a2]
    freqs = np.linspace(54.74e6, 59.31e6, 8)

    desc = tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, valuetype="complex", ndim=2)])
    t = tables.default_ms(str(path), desc)
    t.addrows(len(a1))
    t.putcol("ANTENNA1", a1)
    t.putcol("ANTENNA2", a2)
    t.putcol("UVW", _rot_w(true, stored_rot_deg))
    t.putcol("TIME", np.full(len(a1), T_START))
    t.putcol("TIME_CENTROID", np.full(len(a1), tc))
    t.putcol("INTERVAL", np.full(len(a1), INTERVAL))
    t.putcol("DATA", np.ones((len(a1), freqs.size, 4), complex))
    t.close()
    with tables.table(str(path / "ANTENNA"), readonly=False, ack=False) as an:
        an.addrows(nant)
        an.putcol("POSITION", pos)
    with tables.table(str(path / "FIELD"), readonly=False, ack=False) as f:
        f.addrows(1)
        for c in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            f.putcol(c, np.array([[[ra, dec]]]))
    with tables.table(str(path / "SPECTRAL_WINDOW"), readonly=False, ack=False) as s:
        s.addrows(1)
        s.putcol("NUM_CHAN", np.array([freqs.size]))
        s.putcol("CHAN_FREQ", freqs[None, :])
    return true, freqs


def test_fix_uvw_removes_stored_rotation(tmp_path):
    ms = tmp_path / "snap.ms"
    true, _ = _make_ms(ms, stored_rot_deg=-0.34)
    before = uvw.fix_uvw(ms, dry_run=True)
    assert before.rot_deg == pytest.approx(0.34, abs=1e-6)
    assert before.max_dw_m < 1e-6
    st = uvw.fix_uvw(ms, epoch="centroid")
    assert st.rot_deg == pytest.approx(0.34, abs=1e-6) and not st.rephased
    tables = pytest.importorskip("casacore.tables")
    with tables.table(str(ms), ack=False) as t:
        np.testing.assert_allclose(t.getcol("UVW"), true, atol=1e-6)
        np.testing.assert_array_equal(t.getcol("DATA"), 1.0)
    after = uvw.fix_uvw(ms, dry_run=True)    # consistency check now passes
    assert abs(after.rot_deg) < 1e-6 and after.rms_duv_m < 1e-6


def test_fix_uvw_time_epoch_rephases_like_chgcentre(tmp_path):
    ms = tmp_path / "snap.ms"
    _, freqs = _make_ms(ms, stored_rot_deg=0.0)
    tables = pytest.importorskip("casacore.tables")
    with tables.table(str(ms), ack=False) as t:
        w_old = t.getcol("UVW")[:, 2]
    st = uvw.fix_uvw(ms, epoch="time")
    assert st.rephased and 0.01 < st.max_dw_m < 1.0
    with tables.table(str(ms), ack=False) as t:
        w_new = t.getcol("UVW")[:, 2]
        d = t.getcol("DATA")
    want = np.exp(2j * np.pi * (w_old - w_new)[:, None] * freqs[None, :] / uvw.C_LIGHT)
    np.testing.assert_allclose(d[:, :, 0], want, atol=1e-5)


def test_fix_uvw_refuses_inconsistent_phase_centre(tmp_path):
    ms = tmp_path / "snap.ms"
    _make_ms(ms, stored_rot_deg=0.0)
    tables = pytest.importorskip("casacore.tables")
    with tables.table(str(ms / "FIELD"), readonly=False, ack=False) as f:
        pd = f.getcol("PHASE_DIR")
        pd[0, 0, 1] += np.radians(1.0)
        f.putcol("PHASE_DIR", pd)
    with tables.table(str(ms), ack=False) as t:
        old = t.getcol("UVW")
    with pytest.raises(ValueError, match="disagree"):
        uvw.fix_uvw(ms)
    with tables.table(str(ms), ack=False) as t:
        np.testing.assert_array_equal(t.getcol("UVW"), old)
