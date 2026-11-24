"""Scheduling switches of gsi v5 (both opt-in): v3.retry_on_checkpoint_node (a band-hour's first retry waits for the
node holding its local checkpoint) and pipeline.dispatch_band_order (band order inside an LST hour); and the tmpfs
raw-MS count of _plan_shm.  Ported from exp/gsi-v5-sched (676640f) without the lanes / gpu_share parts."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import Slot, V3Params
from gpu_subband_imaging.ledger import FAILED, QUEUED, Job
from gpu_subband_imaging.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "examples" / "config"

AOF = dict(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", write_kept_ms=False,
           aoflag="hour")


# -- config ---------------------------------------------------------------------------------------------------
def test_defaults_are_off():
    assert V3Params().retry_on_checkpoint_node is False
    pl = config_mod.load_pipeline(CONFIG / "pipeline.yaml")
    assert pl.dispatch_band_order == []


def test_dispatch_band_order_is_read(tmp_path):
    c = tmp_path / "cfg"
    shutil.copytree(CONFIG, c)
    d = yaml.safe_load((c / "pipeline.yaml").read_text())
    d["dispatch_band_order"] = [73, "55", 82]
    (c / "pipeline.yaml").write_text(yaml.safe_dump(d))
    assert config_mod.load_pipeline(c / "pipeline.yaml").dispatch_band_order == [73, 55, 82]


# -- orchestrator -----------------------------------------------------------------------------------------------
def _orch(order=(), v3=None):
    o = Orchestrator.__new__(Orchestrator)
    pl = SimpleNamespace(dispatch_band_order=list(order),
                         v3=V3Params(**(v3 or {**AOF, "checkpoint": "local", "retry_on_checkpoint_node": True})))
    o.cfg = SimpleNamespace(pipeline=pl)
    o.slots = [Slot("n1", 0), Slot("n2", 0)]
    o.group_of = {(b, 0): "00h" for b in (18, 55, 82)}
    o.group_of.update({(b, 1): "01h" for b in (18, 55, 82)})
    return o


def _st(jobs, node_of=None):
    node_of = node_of or {}
    return {(j.band, "00h" if j.batch == 0 else "01h"): {"batches": [j], "hour": None, "node": node_of.get(j.band)}
            for j in jobs}


def _up(node):            # node_free_cores of a reachable node
    return 48.0


def test_failed_band_hour_waits_for_its_checkpoint_node():
    o = _orch()
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    st = _st([failed, queued], {18: "n1"})
    order = {"00h": 0, "01h": 1}
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, order, _up) is queued     # not the redo elsewhere
    assert o._choose(Slot("n1", 1), [failed, queued], st, False, order, _up) is failed     # its node, any GPU
    second = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=2)                   # 2nd retry: anywhere
    assert o._choose(Slot("n2", 0), [second, queued], _st([second, queued], {18: "n1"}), False, order,
                     _up) is second
    gone = Job(18, 0, 358, status=FAILED, slot="n9:gpu0", attempts=1)                     # node left the cluster
    assert o._choose(Slot("n2", 0), [gone, queued], _st([gone, queued], {18: "n9"}), False, order, _up) is gone


def test_pinned_retry_does_not_wait_for_an_unreachable_node():
    o = _orch()
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    st = _st([failed, queued], {18: "n1"})
    fc = {"n1": None, "n2": 48.0}
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, {"00h": 0, "01h": 1}, fc.get) is failed


def test_pinned_retry_goes_anywhere_after_the_pin_timeout():
    o = _orch(v3={**AOF, "checkpoint": "local", "retry_on_checkpoint_node": True, "retry_pin_max_wait_s": 600})
    now = [1000.0]
    o._clock = lambda: now[0]
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    st = _st([failed, queued], {18: "n1"})
    order = {"00h": 0, "01h": 1}
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, order, _up) is queued    # first refusal: t=1000
    now[0] = 1599.0
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, order, _up) is queued
    now[0] = 1600.0
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, order, _up) is failed


# -- the streaming loop with the pin (fake ledger rows, dispatch and clock) -------------------------------------
class _ImmediatePool:
    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        return f

    def shutdown(self, wait=True):
        pass


def _stream(tmp_path, monkeypatch, max_retries, fails, fc, wait_s=1800.0):
    """One band-hour (55 MHz 00h, batch 0) on n1 / n2 (hour_source memory, local checkpoint, pinned retry).  `fails`:
    attempt numbers that fail; `fc(node, attempt)`: free cores (budget 10).  Returns [(attempt, node, clock)]."""
    import threading
    from gpu_subband_imaging import dispatch, orchestrator
    from gpu_subband_imaging.ledger import Ledger
    o = Orchestrator.__new__(Orchestrator)
    pl = SimpleNamespace(layout="lst", stream_hour_images=True, keep_ms_local=None, lst_hours=[0],
                         max_retries=max_retries, wideband_per_hour=False, calcache_cleanup="none",
                         batch_timeout_seconds=60, dispatch_band_order=[],
                         v3=V3Params(**AOF, checkpoint="local", retry_on_checkpoint_node=True,
                                     retry_pin_max_wait_s=wait_s))
    o.cfg = SimpleNamespace(pipeline=pl, cluster=SimpleNamespace(free_core_budget=10))
    o.slots = [Slot("n1", 0), Slot("n2", 0)]
    o.ledger = Ledger(tmp_path / "ledger.sqlite")
    o._lock = threading.Lock()
    o._wideband_done, o._wideband_pool = set(), None
    o._write_metadata = lambda *a, **k: None
    o._cleanup_manifests = lambda: None
    o.group_of = {(55, 0): "00h"}
    o.ledger.seed([Job(55, 0, 358)])
    now = [0.0]
    o._clock = lambda: now[0]
    runs = []

    def run_one(slot, j):
        runs.append((len(runs) + 1, slot.node, now[0]))
        if len(runs) in fails:
            o.ledger.mark_failed(j, "boom")
            return False
        o.ledger.mark_done(j)
        return True

    o._run_one = run_one
    monkeypatch.setattr(orchestrator, "ThreadPoolExecutor", _ImmediatePool)
    monkeypatch.setattr(dispatch, "node_free_cores", lambda node: fc(node, len(runs)))
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: now.__setitem__(0, now[0] + 60.0))
    o.run(poll=1)
    return runs, {(j.band, j.batch): j.status for j in Ledger(tmp_path / "ledger.sqlite").all()}


def test_stream_pinned_retry_waits_for_its_node_then_goes_elsewhere(tmp_path, monkeypatch):
    # n1 is over its core budget after the first attempt failed there: the retry waits 1800 s, then runs on n2
    runs, st = _stream(tmp_path, monkeypatch, 2, {1}, lambda node, n: 2.0 if (node == "n1" and n >= 1) else 48.0)
    assert [r[:2] for r in runs] == [(1, "n1"), (2, "n2")]
    assert runs[1][2] >= 1800.0 and st[(55, 0)] == "done"


def test_stream_pinned_retry_runs_on_its_node_when_it_is_free(tmp_path, monkeypatch):
    runs, st = _stream(tmp_path, monkeypatch, 2, {1}, lambda node, n: 48.0)
    assert [r[:2] for r in runs] == [(1, "n1"), (2, "n1")] and runs[1][2] == 0.0 and st[(55, 0)] == "done"


def test_stream_pinned_retry_skips_an_unreachable_node(tmp_path, monkeypatch):
    runs, st = _stream(tmp_path, monkeypatch, 2, {1}, lambda node, n: None if (node == "n1" and n >= 1) else 48.0)
    assert [r[:2] for r in runs] == [(1, "n1"), (2, "n2")] and runs[1][2] == 0.0 and st[(55, 0)] == "done"


def test_stream_two_retries_first_pinned_second_anywhere(tmp_path, monkeypatch):
    # attempt 2 (first retry) on n1; n1 then over budget: attempt 3 (second retry) goes to n2 without waiting
    runs, st = _stream(tmp_path, monkeypatch, 2, {1, 2},
                       lambda node, n: 2.0 if (node == "n1" and n >= 2) else 48.0)
    assert [r[:2] for r in runs] == [(1, "n1"), (2, "n1"), (3, "n2")] and runs[2][2] == 0.0
    assert st[(55, 0)] == "done"
    # max_retries 1 (one retry): the band-hour is given up after the pinned retry failed
    (tmp_path / "r1").mkdir()
    runs, st = _stream(tmp_path / "r1", monkeypatch, 1, {1, 2}, lambda node, n: 48.0)
    assert [r[:2] for r in runs] == [(1, "n1"), (2, "n1")] and st[(55, 0)] == "skipped"


@pytest.mark.parametrize("v3", [dict(AOF, checkpoint="local"),                       # default: off, as before
                                dict(AOF, checkpoint="local", retry_on_checkpoint_node=True, archive="iv",
                                     archive_root="/tmp/a")])
def test_no_pinning_by_default_or_without_local_checkpoints(v3):
    o = _orch(v3=v3)
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    assert o._choose(Slot("n2", 0), [failed, queued], _st([failed, queued], {18: "n1"}), False,
                     {"00h": 0, "01h": 1}, _up) is failed


def test_dispatch_band_order_inside_an_lst_hour():
    jobs = [Job(b, h, 358) for h in (0, 1) for b in (18, 55, 82)]
    order = {"00h": 0, "01h": 1}

    def seq(o):
        left, out = list(jobs), []
        while left:
            j = o._choose(Slot("n1", 0), left, _st(left), False, order, _up)
            out.append((j.band, j.batch))
            left.remove(j)
        return out
    assert seq(_orch()) == [(18, 0), (55, 0), (82, 0), (18, 1), (55, 1), (82, 1)]
    assert seq(_orch(order=[82, 55])) == [(82, 0), (55, 0), (18, 0), (82, 1), (55, 1), (18, 1)]


# -- _plan_shm ----------------------------------------------------------------------------------------------------
class _W:
    UNTAR_SPACE_MARGIN = 1.0

    def __init__(self, root):
        self.root = root
        self._untar_disabled = False

    def _untar_root(self):
        return self.root


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="needs /dev/shm")
@pytest.mark.parametrize("lookahead,feeders,raw", [(8, 3, 3), (2, 8, 3), (4, 5, 5)])
def test_plan_shm_counts_one_raw_ms_per_running_feeder(tmp_path, monkeypatch, lookahead, feeders, raw):
    """tmpfs holds min(lookahead + 1, max_feeders) raw MSes at once (not lookahead): untar stays in tmpfs when that
    fits next to the slots, and moves to work_root when it does not."""
    from gpu_subband_imaging.v3 import worker_v3
    tar_b = 1 << 30
    tar = tmp_path / "a.tar"
    with open(tar, "wb") as f:                     # sparse: only the size matters
        f.truncate(tar_b)
    need = (lookahead + 2) * 130e6 * 1.2 + raw * tar_b

    def run(free):
        monkeypatch.setattr(worker_v3.shutil, "disk_usage", lambda p: SimpleNamespace(free=free))
        b = worker_v3.V3Batch.__new__(worker_v3.V3Batch)
        b.w = _W(Path("/dev/shm") / "gsi-test-not-created" / "untar")      # parent missing: /dev/shm is used
        b.tars = [str(tar)]
        b.lookahead, b.write_kept, b.band, b.batch = lookahead, False, 55, 0
        b.cfg = SimpleNamespace(cluster=SimpleNamespace(max_feeders=feeders))
        b._plan_shm()
        return b.w._untar_disabled
    assert run(need + 1e6) is False
    assert run(need - 1e6) is True
