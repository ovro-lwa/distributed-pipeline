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


def _worker(tmp_path, prep_inmem="off", untar_to="", prep="numpy", do_aoflag=False,
            prep_timeout_s=900.0, prep_procs=2):
    peel = PeelParams()
    pl = SimpleNamespace(
        date="2026-10-31", output_root=str(tmp_path / "out"), movie=SimpleNamespace(),
        fix_uvw="centroid", chanbin=4, prep=prep, do_aoflag=do_aoflag,
        cal=SimpleNamespace(bp="bp", xy="xy"), peel_for_band=lambda band: peel,
        v3=V3Params(prep_inmem=prep_inmem, untar_to=untar_to, prep_procs=prep_procs,
                    prep_timeout_s=prep_timeout_s),
        imaging=SimpleNamespace(imager="wsclean"), do_badants=False)
    cfg = SimpleNamespace(
        pipeline=pl,
        cluster=SimpleNamespace(state_dir=str(tmp_path / "s"), work_root=str(tmp_path / "w"),
                                max_feeders=2, archive_workers=1, zest_recycle_files=25),
        subbands=SimpleNamespace(geom=lambda band: SimpleNamespace()))
    return Worker(cfg, 23, 0, 0)


def _tar(tmp_path):
    t = tmp_path / "lustre" / f"{STAMP}.ms.tar"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(b"\0" * 4096)
    return str(t)


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
    stamp, avg = w._preprocess(_tar(tmp_path))
    assert calls == [("stage", None), "flag_calibrate", "average", "fix_uvw", "visibility_stats"]
    assert avg == tmp_path / "w" / "b23_0" / STAMP / f"{STAMP}_avg.ms"


def test_untar_to_streams_into_tmpfs_and_cleans_up(tmp_path, monkeypatch):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    shm = tmp_path / "shm"
    w = _worker(tmp_path, untar_to=str(shm))
    root = w._untar_root()
    assert root.parent == shm and root.name == f".gsi3w_{os.getuid()}_{os.uname().nodename}_{os.getpid()}"
    stamp, avg = w._preprocess(_tar(tmp_path))
    assert calls[0] == ("stage", root / STAMP)
    assert avg.is_dir() and not (root / STAMP).exists()


def test_on_uses_v3_output(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    shm = tmp_path / "shm"
    w = _worker(tmp_path, "on", untar_to=str(shm))
    tpl = tmp_path / "tpl.ms"
    monkeypatch.setattr(w, "_v3_template_for", lambda ms: tpl)
    monkeypatch.setattr(fastprep, "check_template", calls.rec("check_template"))
    seen = {}

    def run(fn, *args):
        seen["fn"], seen["args"] = fn, args
        Path(args[-1]).mkdir()
        return {"meta": _meta(), "stats": STATS.__dict__.copy(), "task_wall": 0.0}

    monkeypatch.setattr(w, "_v3_run", run)
    root = w._untar_root()
    stamp, avg = w._preprocess(_tar(tmp_path))
    assert seen["fn"] is fastprep.prep_task
    assert seen["args"][:6] == (str(root / STAMP / f"{STAMP}.ms"), "bp", "xy", "", 4, "centroid")
    assert seen["args"][-2:] == (str(tpl), str(avg))
    assert calls == [("stage", root / STAMP), "check_template"]   # no v2 step
    assert "phase=inmem_read" in caplog.text and "fix_uvw 23MHz" in caplog.text
    for phase in ("flag_cal", "average", "fix_uvw", "flag_cal_avg", "inmem_wait"):
        assert f"phase={phase} elapsed=" in caplog.text
    assert "input_quality 23MHz" in caplog.text
    assert not (root / STAMP).exists()


def test_on_falls_back_to_v2(tmp_path, monkeypatch, caplog):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path, "on")
    monkeypatch.setattr(w, "_v3_template_for", lambda ms: tmp_path / "tpl.ms")
    monkeypatch.setattr(fastprep, "check_template", lambda *a: None)

    def boom(fn, *args):
        Path(args[-1]).mkdir()           # a half-written MS must not survive
        raise RuntimeError("child died")

    monkeypatch.setattr(w, "_v3_run", boom)
    stamp, avg = w._preprocess(_tar(tmp_path))
    assert "V3_FALLBACK" in caplog.text
    assert calls[1:] == ["flag_calibrate", "average", "fix_uvw", "visibility_stats"]
    # no template (mstransform of the first file failed): v2 without trying the pool
    calls.clear()
    w2 = _worker(tmp_path / "b", "on")
    monkeypatch.setattr(w2, "_v3_template_for", lambda ms: None)
    monkeypatch.setattr(w2, "_v3_run", lambda *a: pytest.fail("pool used without template"))
    assert w2._preprocess(_tar(tmp_path)) is not None
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
    stamp, avg = w._preprocess(_tar(tmp_path))
    assert calls[1:] == ["v3", "flag_calibrate", "average", "fix_uvw", "visibility_stats", "compare"]
    line = [r.getMessage() for r in caplog.records if "V3_SHADOW" in r.getMessage()]
    assert len(line) == 1 and "identical=True" in line[0] and "stats_equal=True" in line[0]


def test_shadow_failure_does_not_stop_v2(tmp_path, monkeypatch, caplog):
    calls = _Calls()
    monkeypatch.setattr(stage_ms, "stage", _fake_stage(calls))
    _patch_v2(monkeypatch, calls)
    w = _worker(tmp_path, "shadow")
    monkeypatch.setattr(w, "_v3_run", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    assert w._preprocess(_tar(tmp_path)) is not None
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


# -- review fixes: tmpfs sweep, staging fallback, pool crash / timeout, template ---------

import errno
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HAVE_CASA = (importlib.util.find_spec("casacore") is not None
             and importlib.util.find_spec("casatasks") is not None)


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


class _FakeDaemon:
    def zest(self, avg):
        pass

    def close(self):
        pass


def test_run_batch_sweeps_stale_untar_dirs(tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    uid, host = os.getuid(), os.uname().nodename
    stale = shm / f".gsi3w_{uid}_{host}_{_dead_pid()}"
    (stale / "20261031_000000_23MHz" / "x.ms").mkdir(parents=True)
    live_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        live = shm / f".gsi3w_{uid}_{host}_{live_proc.pid}"
        other_user = shm / f".gsi3w_{uid + 1}_{host}_{_dead_pid()}"
        other_host = shm / f".gsi3w_{uid}_elsewhere_{_dead_pid()}"
        unrelated = shm / "somebody_elses"
        for d in (live, other_user, other_host, unrelated):
            d.mkdir()
        target = tmp_path / "precious"
        target.mkdir()
        (target / "keep").write_text("x")
        link2 = shm / f".gsi3w_{uid}_{host}_{_dead_pid()}"
        link2.symlink_to(target)                             # ours by name, but a link
        w = _worker(tmp_path, untar_to=str(shm))
        monkeypatch.setattr(w, "_stage_cal", lambda src: src)
        monkeypatch.setattr(w, "_start_daemon", lambda: _FakeDaemon())
        monkeypatch.setattr(w, "_image_archive", lambda stamp, avg: None)
        seen = {}

        def prep(tar):
            stamp = Path(tar).name[:-7]
            u = w._untar_dir(stamp)
            u.mkdir(parents=True)
            (u / "left_by_a_crash").write_bytes(b"x")       # as if the finally never ran
            seen["stale_gone_at_start"] = not stale.exists()
            return stamp, tmp_path / "avg.ms"

        monkeypatch.setattr(w, "_preprocess", prep)
        w.run_batch([_tar(tmp_path)])
        assert seen["stale_gone_at_start"]
        assert not w._untar_root().exists()                  # own dir removed at batch end
        for d in (live, other_user, other_host, unrelated):
            assert d.is_dir(), d
        assert link2.is_symlink() and (target / "keep").exists()
    finally:
        live_proc.kill()
        live_proc.wait()


def _real_tar(tmp_path):
    ms = tmp_path / "src" / f"{STAMP}.ms"
    ms.mkdir(parents=True)
    (ms / "table.dat").write_bytes(b"m" * 3000)
    tar = tmp_path / "lustre" / f"{STAMP}.ms.tar"
    tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "w") as t:
        t.add(ms, arcname=ms.name)
    return str(tar)


def test_enospc_in_tmpfs_falls_back_to_work_root(tmp_path, monkeypatch, caplog):
    tar = _real_tar(tmp_path)
    calls = []

    def full(source, dest, name):
        calls.append(1)
        (dest / name).mkdir(parents=True)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(stage_ms, "_extract_stream", full)
    w = _worker(tmp_path, untar_to=str(tmp_path / "shm"))
    d = tmp_path / "w" / STAMP
    ms = w._stage(tar, d, STAMP)
    assert calls == [1]                                      # ENOSPC is not retried
    assert ms == d / f"{STAMP}.ms" and (ms / "table.dat").is_file()
    assert "V3_FALLBACK" in caplog.text and "STAGE_RETRY" not in caplog.text
    assert not (w._untar_root() / STAMP).exists()


def test_short_tmpfs_uses_work_root_without_streaming(tmp_path, monkeypatch, caplog):
    tar = _real_tar(tmp_path)
    monkeypatch.setattr(stage_ms, "_extract_stream", lambda *a: pytest.fail("streamed"))
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: real(p)._replace(free=100) if "shm" in str(p) else real(p))
    w = _worker(tmp_path, untar_to=str(tmp_path / "shm"))
    ms = w._stage(tar, tmp_path / "w" / STAMP, STAMP)
    assert (ms / "table.dat").is_file() and "GB free" in caplog.text


def test_symlinked_untar_root_is_refused(tmp_path, caplog):
    tar = _real_tar(tmp_path)
    w = _worker(tmp_path, untar_to=str(tmp_path / "shm"))
    target = tmp_path / "elsewhere"
    target.mkdir()
    w._untar_root().parent.mkdir(parents=True)
    w._untar_root().symlink_to(target)
    ms = w._stage(tar, tmp_path / "w" / STAMP, STAMP)
    assert ms.parent == tmp_path / "w" / STAMP and list(target.iterdir()) == []
    assert "V3_FALLBACK" in caplog.text


def test_pool_is_replaced_after_a_child_crash(tmp_path, capfd):
    from concurrent.futures.process import BrokenProcessPool
    w = _worker(tmp_path, "on")
    w._v3_start_pool()
    try:
        w._v3_run(os.getpid)
        old = w._v3_pool
        procs = list(old._processes.values())
        with pytest.raises(BrokenProcessPool):
            w._v3_run(os._exit, 3)
        assert w._v3_pool is not old and w._v3_pool is not None
        assert all(not p.is_alive() for p in procs)
        assert w._v3_run(np.add, 1, 2) == 3
    finally:
        w._v3_stop_pool()
    time.sleep(0.5)
    assert "Queue is closed" not in capfd.readouterr().err


def test_pool_timeout_kills_and_replaces(tmp_path):
    w = _worker(tmp_path, "on", prep_timeout_s=2.0)
    w._v3_start_pool()
    try:
        w._v3_run(os.getpid)
        old = w._v3_pool
        procs = list(old._processes.values())
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            w._v3_run(time.sleep, 120)
        assert time.monotonic() - t0 < 30
        assert all(not p.is_alive() for p in procs)
        assert w._v3_pool is not old
        assert w._v3_run(np.add, 2, 2) == 4
    finally:
        t0 = time.monotonic()
        w._v3_stop_pool()
        assert time.monotonic() - t0 < 30


def test_discard_pool_cannot_hang(tmp_path):
    w = _worker(tmp_path, "on")
    w._v3_start_pool()
    pool = w._v3_pool
    pool.submit(os.getpid).result()
    pool.submit(time.sleep, 120)                             # a stuck task at batch end
    procs = list(pool._processes.values())
    t0 = time.monotonic()
    Worker._v3_discard_pool(pool, kill=False, wait_s=1.0)
    assert time.monotonic() - t0 < 30
    assert all(not p.is_alive() for p in procs)


def test_template_made_once_and_failure_remembered(tmp_path, monkeypatch, caplog):
    calls = []

    def make(raw, out, chanbin):
        calls.append(out)
        time.sleep(0.2)
        Path(out).mkdir(parents=True)
        return Path(out)

    monkeypatch.setattr(fastprep, "make_template", make)
    w = _worker(tmp_path, "on")
    got = []
    th = [threading.Thread(target=lambda: got.append(w._v3_template_for(Path("raw.ms"))))
          for _ in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert len(calls) == 1 and len(set(got)) == 1 and got[0] == w.work / "_v3_template_avg.ms"

    def broken(raw, out, chanbin):
        calls.append("broken")
        raise RuntimeError("CASA died")

    monkeypatch.setattr(fastprep, "make_template", broken)
    w2 = _worker(tmp_path / "b", "on")
    assert w2._v3_template_for(Path("raw.ms")) is None
    assert w2._v3_template_for(Path("raw.ms")) is None
    assert calls.count("broken") == 1 and "V3_TEMPLATE" in caplog.text


@pytest.mark.skipif(not HAVE_CASA, reason="needs python-casacore and casatasks")
def test_template_for_on_a_synthetic_ms(tmp_path, monkeypatch, caplog):
    from v3_synth import make_raw_ms
    raw, other = tmp_path / "raw.ms", tmp_path / "other.ms"
    make_raw_ms(raw)
    make_raw_ms(other, f0=60e6)
    w = _worker(tmp_path, "on")
    tpl = w._v3_template_for(raw)
    assert (tpl / "table.dat").is_file()
    fastprep.check_template(str(tpl), str(raw), 4)
    with pytest.raises(ValueError, match="SPECTRAL_WINDOW"):
        fastprep.check_template(str(tpl), str(other), 4)
    # a snapshot that does not fit the template goes to v2 before any v3 work
    monkeypatch.setattr(w, "_v3_run", lambda *a: pytest.fail("prep ran on a wrong template"))
    assert w._v3_prep_on(STAMP, other, tmp_path / "d") is None
    assert "template does not fit" in caplog.text
