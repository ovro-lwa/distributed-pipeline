"""stream_hour_images: hour jobs start when their band-hour is done, on its node."""
import threading
import types
from concurrent.futures import Future
from pathlib import Path

import pytest

from gpu_subband_imaging import dispatch
from gpu_subband_imaging.config import ConfigError, PipelineConfig, Slot
from gpu_subband_imaging.ledger import DONE, SKIPPED, Job, Ledger
from gpu_subband_imaging.orchestrator import (HOUR_JOB_BASE, Orchestrator, hour_job_group,
                                              hour_job_id, is_hour_job)


class ImmediatePool:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args):
        f = Future()
        f.set_result(fn(*args))
        return f

    def shutdown(self, wait=True):
        pass


def _orch(tmp_path, slots, local=True, max_retries=1):
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(
            layout="lst", stream_hour_images=True, keep_ms_local="/fast/k" if local else None,
            lst_hours=[23, 0], max_retries=max_retries, wideband_per_hour=False,
            calcache_cleanup="none", batch_timeout_seconds=60,
            hour_imaging=types.SimpleNamespace(timeout_seconds=60)),
        cluster=types.SimpleNamespace(free_core_budget=0))
    orch.slots = slots
    orch.ledger = Ledger(Path(tmp_path) / "ledger.sqlite")
    orch._lock = threading.Lock()
    orch._wideband_done = set()
    orch._wideband_pool = None
    orch._write_metadata = lambda *a, **k: None
    orch._cleanup_manifests = lambda: None
    # band 55: hour 23h = batches 0, 1; hour 00h = batch 2.  band 82: 23h = batch 0
    orch.group_of = {(55, 0): "23h", (55, 1): "23h", (55, 2): "00h", (82, 0): "23h"}
    jobs = [Job(55, 0, 45), Job(55, 1, 45), Job(55, 2, 45), Job(82, 0, 45),
            Job(55, hour_job_id("23h"), 2), Job(55, hour_job_id("00h"), 1),
            Job(82, hour_job_id("23h"), 1)]
    orch.ledger.seed(jobs)
    return orch


def test_hour_job_ids_round_trip():
    assert hour_job_id("00h") == HOUR_JOB_BASE
    assert hour_job_group(hour_job_id("23h")) == "23h"
    assert is_hour_job(Job(55, hour_job_id("05h"), 1)) and not is_hour_job(Job(55, 0, 45))


def test_hour_jobs_follow_their_batches_on_the_same_node(monkeypatch, tmp_path):
    slots = [Slot("n1", 0), Slot("n2", 0)]
    orch = _orch(tmp_path, slots)
    monkeypatch.setattr("gpu_subband_imaging.orchestrator.ThreadPoolExecutor", ImmediatePool)
    monkeypatch.setattr(dispatch, "node_free_cores", lambda node: None)
    order = []

    def run_one(slot, j):
        order.append(("batch", j.band, j.batch, slot.node))
        orch.ledger.mark_done(j)
        return True

    def run_hour(slot, j):
        order.append(("hour", j.band, hour_job_group(j.batch), slot.node))
        orch.ledger.mark_done(j)
        return True

    orch._run_one, orch._run_hour_job = run_one, run_hour
    orch.run(poll=0)

    node_of = {}
    for kind, band, x, node in order:
        if kind == "batch":
            node_of.setdefault((band, orch.group_of[(band, x)]), set()).add(node)
    hours = [(band, g, node) for kind, band, g, node in order if kind == "hour"]
    assert sorted((b, g) for b, g, _ in hours) == [(55, "00h"), (55, "23h"), (82, "23h")]
    for band, g, node in hours:
        assert node_of[(band, g)] == {node}          # every batch and the hour job on one node
        last_batch = max(i for i, o in enumerate(order) if o[0] == "batch"
                         and o[1] == band and orch.group_of[(band, o[2])] == g)
        assert order.index(("hour", band, g, node)) > last_batch
    # 23h before 00h (night order), and the 55MHz 23h hour job before its 00h batch
    assert order.index(("hour", 55, "23h", node_of[(55, "23h")].pop())) < \
        next(i for i, o in enumerate(order) if o[:3] == ("batch", 55, 2))
    assert all(j.status == DONE for j in Ledger(Path(tmp_path) / "ledger.sqlite").all())


def test_hour_job_skipped_when_all_batches_skipped(monkeypatch, tmp_path):
    orch = _orch(tmp_path, [Slot("n1", 0)], max_retries=0)
    monkeypatch.setattr("gpu_subband_imaging.orchestrator.ThreadPoolExecutor", ImmediatePool)
    monkeypatch.setattr(dispatch, "node_free_cores", lambda node: None)
    ran_hours = []

    def run_one(slot, j):
        if j.band == 82:
            orch.ledger.mark_failed(j, "boom")
            return False
        orch.ledger.mark_done(j)
        return True

    orch._run_one = run_one
    orch._run_hour_job = lambda slot, j: (ran_hours.append(j.band), orch.ledger.mark_done(j), True)[-1]
    orch.run(poll=0)
    st = {(j.band, j.batch): j.status for j in Ledger(Path(tmp_path) / "ledger.sqlite").all()}
    assert st[(82, 0)] == SKIPPED and st[(82, hour_job_id("23h"))] == SKIPPED
    assert 82 not in ran_hours and ran_hours.count(55) == 2


def test_keep_ms_local_config_checks():
    base = dict(date="2026-10-28", hours="all", bands=[55], batch_size=45,
                batch_timeout_seconds=None, max_retries=1, data_root="/d", output_root="/o",
                chanbin=4, cal=None, peel=None, peel_overrides={}, imaging=None, movie=None,
                do_aoflag=False, do_badants=True, env=None, layout="lst", run_label="R")
    PipelineConfig(**base, stream_hour_images=True, keep_ms_local="/fast/k")
    with pytest.raises(ConfigError):
        PipelineConfig(**base, keep_ms_local="/fast/k")                 # needs streaming
    with pytest.raises(ConfigError):
        PipelineConfig(**base, stream_hour_images=True, keep_ms_local="/lustre/x")
    with pytest.raises(ConfigError):
        PipelineConfig(**base, prep="fast")
