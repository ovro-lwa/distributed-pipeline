"""v3.prefetch: the slot's next batch starts during the hour stage of its current batch, stages + preps its first
snapshots while the GPU is held, and peels only after the GPU's flock is free (no GPU, no Julia)."""
import fcntl
import json
import logging
import os
import shutil
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_subband_imaging import dispatch
from gpu_subband_imaging.config import ConfigError, Slot, V3Params
from gpu_subband_imaging.ledger import Job, Ledger
from gpu_subband_imaging.orchestrator import Orchestrator, prefetch_marker
from gpu_subband_imaging.v3 import fastprep, hourstore, memory
from gpu_subband_imaging.v3.worker_v3 import V3Batch

from test_v3_batch import FakeDaemon, STAMPS, _bdir, _config, _worker, env, fake_prep_slot_task  # noqa: F401
from test_v3_engine import needs_v3_imager


def test_prefetch_config():
    assert V3Params().prefetch is False
    with pytest.raises(ConfigError, match="prefetch needs v3.hour_source=memory"):
        V3Params(prefetch=True)
    V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", prefetch=True)


def _orch(tmp_path, prefetch, streaming=True):
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(max_retries=0, keep_ms_local=None, lst_hours=[5], date="2026-10-31",
                                       stream_hour_images=streaming, layout="lst",
                                       v3=V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm",
                                                   hour_source="memory", prefetch=prefetch)),
        cluster=types.SimpleNamespace(free_core_budget=0, state_dir=str(tmp_path / f"st{prefetch}")))
    orch.slots = [Slot("gpu-node-01", 1)]
    orch.ledger = Ledger(tmp_path / f"ledger{prefetch}.sqlite")
    orch.ledger.seed([Job(82, 0, 1), Job(55, 0, 1), Job(23, 0, 1)])
    orch.group_of = {(82, 0): "05h", (55, 0): "05h", (23, 0): "05h"}
    orch._lock = threading.Lock()
    orch._maybe_wideband = lambda: None
    return orch


@pytest.mark.parametrize("prefetch", [True, False])
def test_streaming_prefetch_dispatches_the_next_batch_during_the_hour_stage(monkeypatch, tmp_path, prefetch):
    """A slot whose batch wrote its hour marker takes the next batch while it still runs (never a third one);
    without v3.prefetch the next batch waits for the slot."""
    orch = _orch(tmp_path, prefetch)
    monkeypatch.setattr(dispatch, "node_free_cores", lambda node: None)
    started = {b: threading.Event() for b in (23, 55, 82)}
    concurrent, max_running = [], [0]
    running = set()
    lk = threading.Lock()

    def run_one(slot, job):
        with lk:
            running.add(job.band)
            max_running[0] = max(max_running[0], len(running))
        started[job.band].set()
        if job.band == 23:                             # dispatched first (lowest band): "hour stage" now
            m = prefetch_marker(orch.cfg, 23, 0)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("x")
            concurrent.append(started[55].wait(2))
            time.sleep(0.3)                            # a third batch would start now if the rule were wrong
            m.unlink()
        with lk:
            running.discard(job.band)
        orch.ledger.mark_done(job)
        return True
    orch._run_one = run_one
    with ThreadPoolExecutor(max_workers=3) as pool:
        orch._run_streaming(pool, poll=0.01)
    assert concurrent == [prefetch]
    assert max_running[0] == (2 if prefetch else 1)
    assert all(e.is_set() for e in started.values())


def test_prefetch_needs_streaming(tmp_path):
    assert _orch(tmp_path, True)._prefetch and not _orch(tmp_path, True, streaming=False)._prefetch
    assert not _orch(tmp_path, False)._prefetch


def test_stale_marker_of_a_killed_worker_is_removed(monkeypatch, tmp_path):
    """The orchestrator removes the marker of a batch whose worker ended without removing it."""
    orch = _orch(tmp_path, True)
    monkeypatch.setattr(dispatch, "node_free_cores", lambda node: None)

    def run_one(slot, job):
        m = prefetch_marker(orch.cfg, job.band, job.batch)
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("x")                              # killed in its hour stage: the marker stays
        orch.ledger.mark_done(job)
        return True
    orch._run_one = run_one
    with ThreadPoolExecutor(max_workers=2) as pool:
        orch._run_streaming(pool, poll=0.01)
    assert not list(Path(prefetch_marker(orch.cfg, 23, 0)).parent.glob("*.hour"))


# -- the worker ------------------------------------------------------------------------------------------------

def _products(tmp):
    b = _bdir(tmp)
    prov = json.loads((b / "provenance.json").read_text())
    stored = hourstore.read_records(Path(prov["v3"]["store_path"]))[2]
    recs = {r.stem: {k: a.tobytes() for k, a in r.arrays.items()} for r in stored}
    fits = {p.name: p.read_bytes() for p in sorted((b / "snapshots_clean").iterdir())}
    return prov, recs, fits, [r.stem for r in stored]


def _reset(tmp):
    for p in ("out", "state", "spill", "work"):
        shutil.rmtree(tmp / p, ignore_errors=True)


@needs_v3_imager
@pytest.mark.parametrize("tight", [False, True])
def test_prefetched_batch_stages_before_the_gpu_and_gives_the_same_bytes(env, monkeypatch, caplog, tight):
    tmp = env.tmp
    lock = tmp / "gpu0.lock"
    monkeypatch.setattr(V3Batch, "_gpu_lock_path", lambda self: str(lock))
    # the reference: the same batch, nobody holds the GPU
    cfg = _config(tmp, delete_checkpoint=False, prefetch=True)
    d0 = FakeDaemon()
    _worker(cfg, d0).run_batch(env.tars)
    ref = _products(tmp)
    assert ref[0]["hour"]["status"] == "ok"
    _reset(tmp)
    # the prefetched run: the GPU is held by "the previous batch" (another open file description)
    events = []
    preps = []

    def prep(raw_ms, *a, **k):
        preps.append(Path(raw_ms).name[:-3])
        events.append(("prep", Path(raw_ms).name[:-3]))
        return fake_prep_slot_task(raw_ms, *a, **k)
    monkeypatch.setattr(fastprep, "prep_slot_task", prep)
    if tight:                                              # (_prefetch_room itself: test below)
        monkeypatch.setattr(V3Batch, "_prefetch_room", lambda self, n: False)
    markers = []
    real_hour = V3Batch._hour_stage

    def hour_stage(self):
        r = real_hour(self)
        markers.append(prefetch_marker(self.cfg, self.band, self.batch).exists())
        return r
    monkeypatch.setattr(V3Batch, "_hour_stage", hour_stage)
    d = FakeDaemon()
    w = _worker(cfg, d)

    def start_daemon():
        events.append(("daemon", None))
        return d
    w._start_daemon = start_daemon
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    err = []

    def run():
        try:
            w.run_batch(env.tars)
        except BaseException as e:                        # noqa: BLE001
            err.append(e)
    with caplog.at_level(logging.INFO, logger="gsi"):
        t = threading.Thread(target=run)
        t.start()
        n_pre = 0 if tight else 3                          # lookahead 2: snapshots 0..2 before the GPU
        deadline = time.time() + 30
        while len(preps) < n_pre and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert sorted(preps) == STAMPS[:n_pre] and ("daemon", None) not in events
        os.close(fd)                                       # the previous batch exits
        t.join(120)
    assert not t.is_alive() and not err, err
    assert "V3_PREFETCH" in caplog.text and "phase=prefetch_gpu_wait" in caplog.text
    first_daemon = events.index(("daemon", None))
    assert sorted(e for e in events[:first_daemon] if e[0] == "prep") == [("prep", s) for s in STAMPS[:n_pre]]
    assert sorted(preps) == STAMPS                         # every snapshot prepped once
    assert d.calls == STAMPS
    got = _products(tmp)
    assert got[1] == ref[1] and got[2].keys() == ref[2].keys()
    # one record per snapshot, prefetched or not.  The store's append order is not fixed (the snapshot loop's 3
    # post threads race to the engine, with or without prefetch); the hour op takes the stems sorted
    # (test_v3_engine: test_hour_input_does_not_depend_on_the_record_arrival_order)
    assert sorted(got[3]) == sorted(ref[3]) and len(set(got[3])) == len(got[3]) == len(STAMPS)
    assert got[0]["hour"]["status"] == "ok" and got[0]["hour"]["n_loaded"] == len(STAMPS)
    assert markers == [True]                               # written in the hour stage
    assert not prefetch_marker(cfg, 23, 0).exists()        # removed at the end of the batch
    assert not list(Path("/dev/shm").glob(f"gsi3_{os.getpid()}_*")) and not w.work.exists()
    # the GPU lock was given back: a later batch takes it at once
    fd2 = os.open(str(lock), os.O_RDWR)
    try:
        fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd2)


@needs_v3_imager
def test_prefetch_off_takes_no_gpu_lock(env, monkeypatch):
    tmp = env.tmp
    lock = tmp / "gpu0.lock"
    monkeypatch.setattr(V3Batch, "_gpu_lock_path", lambda self: str(lock))
    cfg = _config(tmp, delete_checkpoint=False)
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        _worker(cfg, FakeDaemon()).run_batch(env.tars)      # does not wait for the held lock
    finally:
        os.close(fd)
    assert json.loads((_bdir(tmp) / "provenance.json").read_text())["hour"]["status"] == "ok"
    assert not (Path(cfg.cluster.state_dir) / "prefetch").exists()


@needs_v3_imager
def test_prefetched_retry_with_fits_stages_nothing_early(env, monkeypatch, caplog):
    """A retry whose work list needs the store (FITS already there): no staging before the GPU."""
    tmp = env.tmp
    lock = tmp / "gpu0.lock"
    monkeypatch.setattr(V3Batch, "_gpu_lock_path", lambda self: str(lock))
    cfg = _config(tmp, delete_checkpoint=False, prefetch=True)
    v = V3Batch(_worker(cfg, FakeDaemon()), env.tars)
    v._prefetch_stage(env.tars, {env.tars[0]})
    assert v._pre is None and v.ring is None


def test_prefetch_room(env, monkeypatch, caplog):
    cfg = _config(env.tmp, prefetch=True, ram_reserve_gb=10)
    v = V3Batch(_worker(cfg, FakeDaemon()), env.tars)
    need = 10e9 + memory.fixed_bytes(v.v3.prep_procs, v.v3.prep_gain_cache) + 4 * 130e6 + 2 * 1 * v.w.UNTAR_SPACE_MARGIN
    monkeypatch.setattr(memory, "mem_available", lambda meminfo="/proc/meminfo": need + 1)
    assert v._prefetch_room(5)
    monkeypatch.setattr(memory, "mem_available", lambda meminfo="/proc/meminfo": need - 1)
    with caplog.at_level(logging.WARNING, logger="gsi"):
        assert not v._prefetch_room(5)
    assert "V3_PREFETCH_TIGHT" in caplog.text
