"""gsi v3 steps 1-2 in the worker: untar_to, prep_inmem off/shadow/on, the spawn pool."""
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import ConfigError, PeelParams, V3Params
from gpu_subband_imaging.stages import fastcal, quality, stage_ms, uvw
from gpu_subband_imaging.v3 import fastprep
from gpu_subband_imaging.worker import Worker

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"
STAMP = "20261031_101007_23MHz"
STATS = quality.VisibilityStats(100, 90, 80, 80 / 90, 1.5, 9.0)


def _worker(tmp_path, prep_inmem="off", untar_to="", prep="numpy", do_aoflag=False):
    peel = PeelParams()
    pl = SimpleNamespace(
        date="2026-10-31", output_root=str(tmp_path / "out"), movie=SimpleNamespace(),
        fix_uvw="centroid", chanbin=4, prep=prep, do_aoflag=do_aoflag,
        cal=SimpleNamespace(bp="bp", xy="xy"), peel_for_band=lambda band: peel,
        v3=V3Params(prep_inmem=prep_inmem, untar_to=untar_to, prep_procs=2))
    cfg = SimpleNamespace(
        pipeline=pl,
        cluster=SimpleNamespace(state_dir=str(tmp_path / "s"), work_root=str(tmp_path / "w")),
        subbands=SimpleNamespace(geom=lambda band: SimpleNamespace()))
    return Worker(cfg, 23, 0, 0)


class _Calls(list):
    def rec(self, name, ret=None):
        def f(*a, **k):
            self.append(name)
            return ret(*a) if callable(ret) else ret
        return f


def _fake_stage(calls):
    def stage(tar, work_dir, untar_to=None):
        calls.append(("stage", untar_to))
        work_dir.mkdir(parents=True, exist_ok=True)
        root = untar_to if untar_to is not None else work_dir
        ms = Path(root) / f"{STAMP}.ms"
        ms.mkdir(parents=True)
        return ms
    return stage


def _patch_v2(monkeypatch, calls):
    def average(ms, out, chanbin):
        calls.append("average")
        Path(out).mkdir()
        return Path(out)
    monkeypatch.setattr(fastcal, "flag_calibrate", calls.rec("flag_calibrate"))
    monkeypatch.setattr(fastcal, "average", average)
    monkeypatch.setattr(uvw, "fix_uvw", calls.rec("fix_uvw", uvw.UVWFixStats(1, 0.3, 2.0, 1e-5, False)))
    monkeypatch.setattr(quality, "visibility_stats", calls.rec("visibility_stats", STATS))


def _meta():
    return {"timings": {"read": {"wall": 0.2, "cpu": 0.2}, "total": {"wall": 1.0, "cpu": 1.0}},
            "fix_uvw": {"n_times": 1, "rot_deg": 0.3, "rms_duv_m": 2.0, "max_dw_m": 1e-5,
                        "rephased": False}, "fix_uvw_error": None}


# -- config ---------------------------------------------------------------------------

def _pipeline_yaml(tmp_path, **kw):
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    d.update(kw)
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_v3_defaults_are_v2(tmp_path):
    pl = config_mod.load_pipeline(_pipeline_yaml(tmp_path, prep="numpy"))
    assert pl.v3.untar_to == "" and pl.v3.prep_inmem == "off"
    w = _worker(tmp_path)
    assert w._untar_dir(STAMP) is None and w._prep_inmem() == "off"


def test_v3_prep_config_validation(tmp_path):
    ok = config_mod.load_pipeline(_pipeline_yaml(
        tmp_path, prep="numpy", do_aoflag=False,
        v3={"prep_inmem": "shadow", "untar_to": "/dev/shm/gsi3", "prep_procs": 3}))
    assert ok.v3.prep_inmem == "shadow" and ok.v3.prep_procs == 3
    with pytest.raises(ConfigError, match="prep=numpy"):
        config_mod.load_pipeline(_pipeline_yaml(tmp_path, prep="casa", v3={"prep_inmem": "on"}))
    with pytest.raises(ConfigError, match="do_aoflag"):
        config_mod.load_pipeline(_pipeline_yaml(tmp_path, prep="numpy", do_aoflag=True,
                                                v3={"prep_inmem": "on"}))
    with pytest.raises(ConfigError, match="node-local"):
        V3Params(untar_to="/lustre/x")
    with pytest.raises(ConfigError, match="prep_inmem"):
        V3Params(prep_inmem="yes")


def test_prep_inmem_only_on_numpy_path(tmp_path):
    assert _worker(tmp_path, "on")._prep_inmem() == "on"
    assert _worker(tmp_path, "on", prep="casa")._prep_inmem() == "off"
    assert _worker(tmp_path, "on", do_aoflag=True)._prep_inmem() == "off"


# -- _preprocess -----------------------------------------------------------------------

def test_off_is_v2(tmp_path, monkeypatch):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path)
    monkeypatch.setattr(w, "_v3_run", lambda *a: pytest.fail("v3 used with prep_inmem=off"))
    stamp, avg = w._preprocess(f"/lustre/x/{STAMP}.ms.tar")
    assert calls == [("stage", None), "flag_calibrate", "average", "fix_uvw", "visibility_stats"]
    assert avg == tmp_path / "w" / "b23_0" / STAMP / f"{STAMP}_avg.ms"


def test_untar_to_streams_into_tmpfs_and_cleans_up(tmp_path, monkeypatch):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    shm = tmp_path / "shm"
    w = _worker(tmp_path, untar_to=str(shm))
    stamp, avg = w._preprocess(f"/lustre/x/{STAMP}.ms.tar")
    assert calls[0] == ("stage", shm / STAMP)
    assert avg.is_dir() and not (shm / STAMP).exists()


def test_on_uses_v3_output(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    shm = tmp_path / "shm"
    w = _worker(tmp_path, "on", untar_to=str(shm))
    tpl = tmp_path / "tpl.ms"
    monkeypatch.setattr(w, "_v3_template_for", lambda ms: tpl)
    seen = {}

    def run(fn, *args):
        seen["fn"], seen["args"] = fn, args
        Path(args[-1]).mkdir()
        return {"meta": _meta(), "stats": STATS.__dict__.copy()}

    monkeypatch.setattr(w, "_v3_run", run)
    stamp, avg = w._preprocess(f"/lustre/x/{STAMP}.ms.tar")
    assert seen["fn"] is fastprep.prep_task
    assert seen["args"][:6] == (str(shm / STAMP / f"{STAMP}.ms"), "bp", "xy", "", 4, "centroid")
    assert seen["args"][-2:] == (str(tpl), str(avg))
    assert calls == [("stage", shm / STAMP)]                     # no v2 step
    assert "phase=inmem_read" in caplog.text and "fix_uvw 23MHz" in caplog.text
    assert "input_quality 23MHz" in caplog.text
    assert not (shm / STAMP).exists()


def test_on_falls_back_to_v2(tmp_path, monkeypatch, caplog):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path, "on")
    monkeypatch.setattr(w, "_v3_template_for", lambda ms: tmp_path / "tpl.ms")

    def boom(fn, *args):
        Path(args[-1]).mkdir()           # a half-written MS must not survive
        raise RuntimeError("child died")

    monkeypatch.setattr(w, "_v3_run", boom)
    stamp, avg = w._preprocess(f"/lustre/x/{STAMP}.ms.tar")
    assert "V3_FALLBACK" in caplog.text
    assert calls[1:] == ["flag_calibrate", "average", "fix_uvw", "visibility_stats"]
    # no template (mstransform of the first file failed): v2 without trying the pool
    calls.clear()
    w2 = _worker(tmp_path / "b", "on")
    monkeypatch.setattr(w2, "_v3_template_for", lambda ms: None)
    monkeypatch.setattr(w2, "_v3_run", lambda *a: pytest.fail("pool used without template"))
    assert w2._preprocess(f"/lustre/x/{STAMP}.ms.tar") is not None
    assert "flag_calibrate" in calls


def test_shadow_runs_v3_first_and_keeps_v2(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path, "shadow")

    def run(fn, *args):
        calls.append("v3")
        assert len(args) == 8             # no template / output: the arrays come back
        return {"meta": _meta(), "stats": STATS.__dict__.copy(), "snap": "SNAP"}

    def compare(snap, ms):
        calls.append("compare")
        assert snap == "SNAP" and Path(ms).name == f"{STAMP}_avg.ms"
        return {"identical": True, "data_max_abs": 0.0, "data_rel_rms": 0.0, "data_ndiff": 0,
                "flag_ndiff": 0, "flag_row_ndiff": 0, "weight_ndiff": 0, "sigma_ndiff": 0,
                "uvw_ndiff": 0, "uvw_max_abs": 0.0}

    monkeypatch.setattr(w, "_v3_run", run)
    monkeypatch.setattr(fastprep, "compare_snap_ms", compare)
    stamp, avg = w._preprocess(f"/lustre/x/{STAMP}.ms.tar")
    assert calls[1:] == ["v3", "flag_calibrate", "average", "fix_uvw", "visibility_stats", "compare"]
    line = [r.getMessage() for r in caplog.records if "V3_SHADOW" in r.getMessage()]
    assert len(line) == 1 and "identical=True" in line[0] and "stats_equal=True" in line[0]


def test_shadow_failure_does_not_stop_v2(tmp_path, monkeypatch, caplog):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path, "shadow")
    monkeypatch.setattr(w, "_v3_run", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    assert w._preprocess(f"/lustre/x/{STAMP}.ms.tar") is not None
    assert "V3_SHADOW" in caplog.text and "flag_calibrate" in calls


# -- the spawn pool -------------------------------------------------------------------------

def test_spawn_pool_runs_in_other_processes(tmp_path):
    w = _worker(tmp_path, "on")
    w._v3_start_pool()
    try:
        pids = {w._v3_run(os.getpid) for _ in range(4)}
        assert os.getpid() not in pids
        assert w._v3_pool._mp_context.get_start_method() == "spawn"
        assert w._v3_run(np.add, 2, 3) == 5
    finally:
        w._v3_stop_pool()
    assert w._v3_pool is None
    with pytest.raises(RuntimeError, match="not running"):
        w._v3_run(os.getpid)


def test_off_starts_no_pool(tmp_path):
    w = _worker(tmp_path)
    w._v3_start_pool()
    assert w._v3_pool is None
