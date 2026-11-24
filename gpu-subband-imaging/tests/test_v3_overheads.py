"""v3.zest_persistent (one zest daemon per GPU across batches, v3/zest_server.py) and v3.aoflag_async_io
(overlapped slot copies in the aoflag=hour passes): no GPU, no Julia (a stand-in daemon script)."""
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gpu_subband_imaging.config import ConfigError, V3Params
from gpu_subband_imaging.stages.peel import ZestTimeout
from gpu_subband_imaging.v3 import hourstore, slots, zest_server
from gpu_subband_imaging.v3.worker_v3 import V3Batch

# -- the persistent daemon ------------------------------------------------------------------------------

FAKE = '''
import os, sys, time
log = open(os.environ["FAKE_LOG"], "a")
print("DAEMON_READY sources=1 device=fake pid=%d" % os.getpid(), flush=True)
for line in sys.stdin:
    cmd = line.strip()
    log.write("%d %s\\n" % (os.getpid(), cmd)); log.flush()
    if cmd == "QUIT":
        break
    parts = cmd.split()
    if parts[0] == "RESET_WARM":
        print(f"RESET_WARM_OK id={parts[1]}", flush=True)
        continue
    if parts[0] == "RELEASE":
        print(f"RELEASE_OK id={parts[1]}", flush=True)
        continue
    name, rid = parts[1], parts[2]
    if name.endswith("hang"):
        time.sleep(3600)
    if name.endswith("bad"):
        print(f"FAILED {name} id={rid} boom", flush=True)
        continue
    print("PEEL_SUMMARY pass=all selected=5 accepted=5 action=sky_rfi", flush=True)
    print(f"ZESTED {name} 0.25 id={rid}", flush=True)
print("DAEMON_EXIT", flush=True)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_daemon.py"
    script.write_text(FAKE)
    julia = tmp_path / "julia"
    julia.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
    julia.chmod(julia.stat().st_mode | stat.S_IEXEC)
    d = Path(f"/tmp/gsi_zest_test_{os.getpid()}_{time.time_ns() % 10**9}")
    monkeypatch.setenv("GSI_ZEST_SERVER_DIR", str(d))
    flog = tmp_path / "daemon.log"
    env = dict(os.environ, FAKE_LOG=str(flog))
    yield dict(julia=str(julia), env=env, log=flog, dir=d)
    for sp in d.glob("*.sock"):                     # stop every server this test started
        try:
            c = zest_server.PersistentZest._connect(str(sp), 5.0)
            if c is not None:
                zest_server.PersistentZest._quit(c, zest_server.PersistentZest._hello(c))
        except Exception:
            pass
    shutil.rmtree(d, ignore_errors=True)


def _attach(fake, gpu=0, zest_env=None, **kw):
    return zest_server.PersistentZest.attach(gpu, julia=fake["julia"], project="/nonexistent/TTCalX",
                                             daemon_jl="daemon.jl", sources="s.json", env=fake["env"],
                                             zest_env=zest_env or {"ZEST_ENGINE": "sky_rfi"}, start_timeout=60,
                                             python=sys.executable, **kw)


def _cmds(fake):
    return [ln.split(" ", 1)[1] for ln in fake["log"].read_text().splitlines()]


def test_server_is_reused_across_batches_and_released_at_detach(fake, caplog):
    caplog.set_level(logging.INFO, logger="gsi.peel")
    a = _attach(fake)
    assert a.zest_slot(Path("/dev/shm/x_0"), timeout=10) == pytest.approx(0.25)
    with pytest.raises(RuntimeError, match="boom"):
        a.zest_slot(Path("/dev/shm/x_bad"), timeout=10)
    a.reset_warm()
    pid, dpid = a._p.pid, a.pid
    a.close()
    assert "PEEL_SUMMARY pass=all" in caplog.text          # the daemon's lines reach the worker log
    b = _attach(fake)                                       # the next batch: same server, same daemon
    try:
        assert (b._p.pid, b.pid) == (pid, dpid) and b.hello["clients"] == 2 and b.hello["slots"] == 1
        assert b.zest_slot(Path("/dev/shm/x_1"), timeout=10) == pytest.approx(0.25)
    finally:
        b.close()
    c = _cmds(fake)
    assert [x.split()[0] for x in c] == ["SHM", "SHM", "RESET_WARM", "RELEASE", "SHM", "RELEASE"]
    assert all(x.endswith(" hard") for x in c if x.startswith("RELEASE"))        # v3.zest_release_hard default


def test_another_key_replaces_the_server(fake):
    a = _attach(fake)
    pid = a._p.pid
    a.close()
    b = _attach(fake, zest_env={"ZEST_ENGINE": "sky_rfi", "ZEST_MAXITER": "7"})
    try:
        assert b._p.pid != pid
        assert not zest_server._alive(pid)                   # the old server (and its daemon) are gone
    finally:
        b.close()
    assert "QUIT" in _cmds(fake)


def test_second_batch_on_the_gpu_gets_a_private_daemon(fake):
    a = _attach(fake)
    try:
        assert _attach(fake) is None                         # attached elsewhere: no wait, caller starts its own
        other = _attach(fake, gpu=1)                         # another GPU has its own server
        assert other is not None and other._p.pid != a._p.pid
        other.close()
    finally:
        a.close()


def test_lost_client_is_released_by_the_server(fake):
    a = _attach(fake)
    a.zest_slot(Path("/dev/shm/x_0"), timeout=10)
    a._drop()                                                # a killed worker: no RELEASE sent
    b = _attach(fake)
    try:
        assert b._p.pid == a._p.pid
    finally:
        b.close()
    assert [x.split()[0] for x in _cmds(fake)] == ["SHM", "RELEASE", "RELEASE"]


def test_timeout_restarts_the_daemon_inside_the_server(fake):
    a = _attach(fake)
    try:
        dpid = a.pid
        with pytest.raises(ZestTimeout):
            a.zest_slot(Path("/dev/shm/x_hang"), timeout=1)
        assert a.restarts == 1
        assert a.zest_slot(Path("/dev/shm/x_2"), timeout=10) == pytest.approx(0.25)
        assert a._request(None, op="hello")["daemon_pid"] != dpid
    finally:
        a.close()


def test_idle_server_exits(fake):
    a = _attach(fake, idle_s=1.0)
    pid = a._p.pid
    a.close()
    for _ in range(100):
        if not zest_server._alive(pid):
            break
        time.sleep(0.1)
    else:
        pytest.fail("idle server did not exit")
    assert _cmds(fake)[-1] == "QUIT" and not list(fake["dir"].glob("*.sock"))


def test_budget_kill_takes_the_server_down(fake):
    a = _attach(fake)
    pid = a._p.pid
    a._p.kill()
    time.sleep(0.5)
    assert not zest_server._alive(pid)
    a._drop()


def test_server_key():
    k = zest_server.server_key(["julia", "d.jl"], {"ZEST_A": "1", "HOME": "/h"}, [])
    assert k == zest_server.server_key(["julia", "d.jl"], {"ZEST_A": "1", "HOME": "/x"}, [])     # not in the key
    assert k != zest_server.server_key(["julia", "d.jl"], {"ZEST_A": "2", "HOME": "/h"}, [])
    assert k != zest_server.server_key(["julia", "d.jl"], {"ZEST_A": "1", "CUDA_VISIBLE_DEVICES": "1"}, [])


def test_worker_falls_back_to_a_private_daemon(monkeypatch, tmp_path):
    from gpu_subband_imaging import worker as wmod
    made = []
    monkeypatch.setattr(wmod, "ZestDaemon", lambda **kw: made.append(kw) or "private")
    monkeypatch.setattr(zest_server.PersistentZest, "attach", classmethod(lambda cls, gpu, **kw: None))
    monkeypatch.setenv("GSI_WORKERS", str(tmp_path))
    w = wmod.Worker.__new__(wmod.Worker)
    w.band, w.gpu = 23, 0
    pl = type("PL", (), {})()
    pl.v3 = V3Params(prep_inmem="on", peel_io="shm", zest_persistent=True)
    pl.env = type("E", (), dict(julia_bin="/j", ttcalx="/t"))()
    pl.cal = type("C", (), dict(sources="s.json", rfi_sources=None))()
    pl.imaging = type("I", (), dict(imager="ttcalx"))()
    pl.peel_for_band = lambda band: type("P", (), dict(daemon_env=lambda self, rfi: {}))()
    w.cfg = type("Cfg", (), dict(pipeline=pl))()
    assert w._start_daemon() == "private" and len(made) == 1


def test_config_rules():
    v = V3Params()
    assert not v.zest_persistent and v.zest_release_hard and not v.aoflag_async_io
    with pytest.raises(ConfigError, match="zest_persistent"):
        V3Params(zest_persistent=True)                       # peel_io ms
    with pytest.raises(ConfigError, match="aoflag_load_threads"):
        V3Params(aoflag_load_threads=0)


# -- v3.aoflag_async_io: the same bytes ----------------------------------------------------------------------

from test_v3_batch import FakeDaemon, STAMPS, _bdir, _config, _worker, env  # noqa: E402,F401
from test_v3_engine import needs_v3_imager  # noqa: E402


def _fake_aoflag(seen):
    def run(self, cdir, saved):
        out = []
        for it in saved:
            v = slots.view_slot(it.slot)
            v.flag[..., :] |= (np.abs(v.data) > 1.5)          # deterministic "flags" from the peeled data
            del v
            out.append((it.slot.name, hashlib.md5(it.slot.read_bytes()).hexdigest()))
        seen.append(out)
        return dict(fake=True)
    return run


@needs_v3_imager
def test_aoflag_async_io_gives_the_same_bytes(env, monkeypatch):
    tmp = env.tmp
    res = {}
    for mode in (False, True):
        seen = []
        monkeypatch.setattr(V3Batch, "_aoflag_hour", _fake_aoflag(seen))
        cfg = _config(tmp, delete_checkpoint=False, aoflag="hour", aoflag_min_snaps=1, aoflag_async_io=mode,
                      aoflag_load_threads=2)
        d = FakeDaemon(sleep=0.05)
        _worker(cfg, d).run_batch(env.tars)
        prov = json.loads((_bdir(tmp) / "provenance.json").read_text())
        assert prov["hour"]["status"] == "ok" and prov["hour"]["n_loaded"] == len(STAMPS)
        recs = {r.stem: {k: a.tobytes() for k, a in r.arrays.items()}
                for r in hourstore.read_records(Path(prov["v3"]["store_path"]))[2]}
        res[mode] = (seen[0], recs, d.calls)
        assert not list(Path("/dev/shm").glob(f"gsi3_{os.getpid()}_*"))       # every ring slot given back
        for p in ("out", "state", "spill", "work"):
            shutil.rmtree(tmp / p, ignore_errors=True)
    assert res[True][0] == res[False][0] and [n for n, _ in res[True][0]] == \
        [f"{k:04d}_{s}.slot" for k, s in enumerate(STAMPS)]
    assert res[True][1] == res[False][1] and sorted(res[True][1]) == STAMPS
    assert res[True][2] == res[False][2] == STAMPS


# -- v3.prep_block_rows: the same arrays --------------------------------------------------------------------

@pytest.mark.parametrize("cache", ["0", "1"])
@pytest.mark.parametrize("block", [1, 7, 64, 10_000])
def test_blocked_cal_average_is_bitwise_the_whole_array_prep(monkeypatch, cache, block):
    from gpu_subband_imaging.stages import fastcal
    from gpu_subband_imaging.v3 import fastprep
    monkeypatch.setenv("GSI_PREP_GAIN_CACHE", cache)
    rng = np.random.default_rng(3)
    nant, nchan, chanbin = 12, 16, 4
    a1, a2 = (x.astype(np.int32) for x in np.triu_indices(nant))
    n = a1.size
    data = (rng.normal(size=(n, nchan, 4)) + 1j * rng.normal(size=(n, nchan, 4))).astype(np.complex64)
    flag = rng.random((n, nchan, 4)) < 0.2
    flag[5] = True                                            # a fully flagged row
    flag_row = np.zeros(n, bool)
    freq = np.linspace(80e6, 104e6, nchan)                     # some channels in 85-100 MHz
    J = (rng.normal(size=(nant, nchan, 2)) + 1j * rng.normal(size=(nant, nchan, 2))).astype(np.complex128)
    dead = rng.random((nant, 2)) < 0.15
    sigma = rng.uniform(0.5, 2.0, (n, 4)).astype(np.float32)
    ref_in = [x.copy() for x in (data, flag, flag_row)]
    fastcal._gp_cache.clear()
    fastcal.calibrate_arrays(*ref_in, a1, a2, freq, J, dead, "3,7")
    ref = fastprep.average_like_mstransform(ref_in[0], ref_in[1], sigma, chanbin)
    fastcal._gp_cache.clear()
    got_in = [x.copy() for x in (data, flag, flag_row)]
    got = fastprep._cal_average_blocked(*got_in, a1, a2, freq, J, dead, "3,7", sigma, chanbin, block)
    for r, g in zip(ref, got):
        assert r.dtype == g.dtype and r.tobytes() == g.tobytes()
    assert got_in[2].tobytes() == ref_in[2].tobytes()          # FLAG_ROW as well


def test_prep_block_rows_config():
    assert V3Params().prep_block_rows == 0
    with pytest.raises(ConfigError, match="prep_block_rows"):
        V3Params(prep_block_rows=-1)
