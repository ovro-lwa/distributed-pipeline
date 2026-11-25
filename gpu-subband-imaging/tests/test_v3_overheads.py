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
    assert all(len(x.split()) == 2 for x in c if x.startswith("RELEASE"))     # RELEASE <id>: the context stays


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


def test_server_key_covers_the_files_zest_settings_name(fake, tmp_path):
    """ZEST_RFI_SOURCES (read once at the daemon's start) rewritten at the same path: a new server, not the old
    daemon with the old model."""
    rfi = tmp_path / "rfi.json"
    rfi.write_text("[1]")
    zenv = {"ZEST_ENGINE": "sky_rfi", "ZEST_RFI_SOURCES": str(rfi)}
    assert zest_server._env_files(dict(zenv, HOME=str(rfi), ZEST_X="rel/path")) == [str(rfi)]
    a = _attach(fake, zest_env=zenv)
    pid = a._p.pid
    a.close()
    b = _attach(fake, zest_env=zenv)                         # unchanged: reused
    assert b._p.pid == pid
    b.close()
    rfi.write_text("[1, 2]")                                 # another size (mtime may be the same second)
    c = _attach(fake, zest_env=zenv)
    try:
        assert c._p.pid != pid
    finally:
        c.close()


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
    assert not v.zest_persistent and not v.aoflag_async_io
    with pytest.raises(ConfigError, match="zest_persistent"):
        V3Params(zest_persistent=True)                       # peel_io ms
    with pytest.raises(ConfigError, match="aoflag_load_threads"):
        V3Params(aoflag_load_threads=0)
    with pytest.raises(ConfigError, match="aoflag_async_io needs v3.aoflag=hour"):
        V3Params(aoflag_async_io=True)
    V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", aoflag="hour",
             aoflag_async_io=True)


@pytest.mark.parametrize("recycle,persistent,ok", [(25, True, False), (0, True, True), (25, False, True)])
def test_zest_persistent_needs_no_recycling(recycle, persistent, ok):
    from types import SimpleNamespace
    from gpu_subband_imaging.config import _check_zest_persistent
    cl = SimpleNamespace(zest_recycle_files=recycle)
    pl = SimpleNamespace(v3=SimpleNamespace(zest_persistent=persistent))
    if ok:
        _check_zest_persistent(cl, pl)
    else:
        with pytest.raises(ConfigError, match="zest_recycle_files: 0"):
            _check_zest_persistent(cl, pl)


# -- faults (verifier's fault injection) -------------------------------------------------------------------

def test_server_killed_mid_batch_the_rest_of_the_batch_uses_a_private_daemon(fake):
    """The server dies (not its daemon): the request in flight fails, the remaining snapshots are peeled by a
    private daemon, close() quits it, and the next batch gets a new server."""
    import signal
    a = _attach(fake)
    a.zest_slot(Path("/dev/shm/x_0"), timeout=10)
    spid = a._p.pid
    os.kill(spid, signal.SIGKILL)
    time.sleep(0.3)
    with pytest.raises(RuntimeError, match="zest server"):
        a.zest_slot(Path("/dev/shm/x_1"), timeout=10)
    assert a.closed
    assert [a.zest_slot(Path(f"/dev/shm/x_{i}"), timeout=10) for i in (2, 3)] == [pytest.approx(0.25)] * 2
    a.reset_warm()
    priv = a._private
    assert priv is not None and a.pid == priv.pid and a._p is priv._p      # the budget kill reaches it
    a.close()
    assert a._private is None and priv._p.poll() is not None              # the private daemon quit
    with pytest.raises(RuntimeError, match="closed"):
        a.zest_slot(Path("/dev/shm/x_4"), timeout=10)
    assert [x.split()[0] for x in _cmds(fake)][-4:] == ["SHM", "SHM", "RESET_WARM", "QUIT"]
    b = _attach(fake)                                        # the next batch gets a new server
    try:
        assert b is not None and b._p.pid != spid
        assert b.zest_slot(Path("/dev/shm/x_5"), timeout=10) == pytest.approx(0.25)
    finally:
        b.close()


def test_daemon_killed_mid_batch_is_restarted_inside_the_server(fake):
    import signal
    a = _attach(fake)
    try:
        a.zest_slot(Path("/dev/shm/x_0"), timeout=10)
        spid = a._p.pid
        os.kill(a.pid, signal.SIGKILL)
        time.sleep(0.3)
        res = []
        for i in range(1, 4):
            try:
                res.append(a.zest_slot(Path(f"/dev/shm/x_{i}"), timeout=10))
            except RuntimeError:
                res.append("err")
        # at most the first request after the death fails (as a private ZestDaemon); the same server goes on
        assert res[1:] == [pytest.approx(0.25)] * 2 and a.restarts == 1
        assert not a.closed and a._private is None and a._p.pid == spid
    finally:
        a.close()


def _slow_julia(fake, tmp_path):
    slow = tmp_path / "slow"
    script = Path(fake["julia"]).read_text()
    j2 = tmp_path / "julia2"
    j2.write_text(f'#!/bin/sh\n[ -f {slow} ] && sleep 3\n' + script.split("\n", 1)[1])
    j2.chmod(0o755)
    return j2, slow


def _attach_julia(fake, julia, **kw):
    return zest_server.PersistentZest.attach(0, julia=str(julia), project="/nonexistent/TTCalX", daemon_jl="daemon.jl",
                                             sources="s.json", env=fake["env"], zest_env={"ZEST_ENGINE": "sky_rfi"},
                                             start_timeout=60, python=sys.executable, **kw)


def test_short_io_margin_kills_a_restarting_server_and_falls_back(fake, tmp_path):
    """peel timeout 1 s, the restarted daemon needs 3 s, client margin 1 s: the server is killed; the rest of
    the batch still peels (private daemon)."""
    j2, slow = _slow_julia(fake, tmp_path)
    a = _attach_julia(fake, j2, io_margin_s=1.0)
    slow.write_text("1")
    res = []
    for name in ("x_hang", "x_1", "x_2"):
        try:
            res.append(a.zest_slot(Path(f"/dev/shm/{name}"), timeout=1))
        except Exception as e:                                  # noqa: BLE001
            res.append(type(e).__name__)
    a.close()
    assert res == ["RuntimeError", pytest.approx(0.25), pytest.approx(0.25)]


def test_default_io_margin_waits_for_the_daemon_restart(fake, tmp_path):
    """The same with the default margin (start_timeout + 90 s): a ZestTimeout, the server lives on."""
    j2, slow = _slow_julia(fake, tmp_path)
    a = _attach_julia(fake, j2)
    assert a._margin == 60 + 90
    slow.write_text("1")
    try:
        spid = a._p.pid
        with pytest.raises(ZestTimeout):
            a.zest_slot(Path("/dev/shm/x_hang"), timeout=1)
        assert a.zest_slot(Path("/dev/shm/x_1"), timeout=10) == pytest.approx(0.25)
        assert not a.closed and a._p.pid == spid and a.restarts == 1
    finally:
        a.close()


def test_private_daemon_that_cannot_start_fails_fast(fake, monkeypatch):
    import signal
    from gpu_subband_imaging.stages import peel
    a = _attach(fake)
    os.kill(a._p.pid, signal.SIGKILL)
    time.sleep(0.3)
    with pytest.raises(RuntimeError):
        a.zest_slot(Path("/dev/shm/x_0"), timeout=10)
    calls = []

    def boom(**kw):
        calls.append(kw)
        raise RuntimeError("no julia")
    monkeypatch.setattr(peel, "ZestDaemon", boom)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="no julia"):
            a.zest_slot(Path("/dev/shm/x_1"), timeout=10)
    assert len(calls) == 1                                       # one start attempt, not one per snapshot
    a.close()


def test_server_log_keeps_only_server_lines(fake):
    """The server's log (appended across nights) has no per-snapshot PEEL_* lines; the worker still gets them."""
    a = _attach(fake)
    for i in range(5):
        a.zest_slot(Path(f"/dev/shm/x_{i}"), timeout=10)
    a.close()
    text = (fake["dir"] / "gpu0.log").read_text()
    assert "ZEST_SERVER" in text and "PEEL_SUMMARY" not in text


def test_server_key_covers_peel_py():
    files = [Path(f).name for f in zest_server._SERVER_FILES]
    assert files == ["zest_server.py", "peel.py"] and all(Path(f).exists() for f in zest_server._SERVER_FILES)


def test_cleanup_nodes_stops_the_zest_servers(monkeypatch, tmp_path):
    import re
    from types import SimpleNamespace
    from gpu_subband_imaging import dispatch, orchestrator
    sent = []
    monkeypatch.setattr(dispatch, "_ssh", lambda node, cmd, timeout: sent.append(cmd) or SimpleNamespace(ok=True, tail=""))
    for persistent in (False, True):
        o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        o.cfg = SimpleNamespace(cluster=SimpleNamespace(work_root=str(tmp_path)),
                                pipeline=SimpleNamespace(keep_ms_local=None, v3=SimpleNamespace(zest_persistent=persistent)))
        o.slots = [SimpleNamespace(node="calimX", gpu=0)]
        o.cleanup_nodes()
    assert "pkill" not in sent[0] and "pkill" in sent[1]
    pat = re.search(r"-f '([^']+)'", sent[1]).group(1)
    assert re.search(pat, f"{sys.executable} -m gpu_subband_imaging.v3.zest_server /tmp/gsi_zest_1/gpu0.spec.json")
    assert not re.search(pat, f"bash -c {sent[1]}")             # pkill does not match its own ssh command line


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
