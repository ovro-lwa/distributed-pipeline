import types
import threading
from concurrent.futures import Future
from pathlib import Path

from gpu_subband_imaging import dispatch
from gpu_subband_imaging.config import Slot
from gpu_subband_imaging.ledger import Job, Ledger, RUNNING
from gpu_subband_imaging.orchestrator import Orchestrator


def test_stitch_movies_uses_timeout_and_keeps_going(monkeypatch):
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(
            movie=types.SimpleNamespace(enabled=True),
            batch_timeout_seconds=123,
        )
    )
    orch.slots = [Slot("gpu-node-01", 0)]
    orch.worker_sh = "workers/run_worker.sh"
    orch.config_dir = "config-allbands"
    orch.worker_env = {}
    orch._bands_complete = lambda: [73]
    calls = []

    def fake_run_worker(*args, **kwargs):
        calls.append((args, kwargs))
        return dispatch.RunResult(False, -1, "ssh timeout")

    monkeypatch.setattr(dispatch, "run_worker", fake_run_worker)

    orch._stitch_movies()

    assert calls[0][0] == (
        "gpu-node-01",
        0,
        "workers/run_worker.sh",
        "config-allbands",
        73,
        -1,
        "STITCH",
    )
    assert calls[0][1]["timeout"] == 123


def test_cleanup_manifests_removes_only_run_date(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.state = tmp_path / "_state"
    orch.cfg = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(date="2026-07-08")
    )
    current = orch.state / "manifests" / "2026-07-08"
    other = orch.state / "manifests" / "2026-07-07"
    current.mkdir(parents=True)
    other.mkdir(parents=True)
    (current / "27_0.txt").write_text("current")
    (other / "27_0.txt").write_text("other")

    orch._cleanup_manifests()

    assert not current.exists()
    assert (other / "27_0.txt").exists()


def test_run_claims_batch_before_worker_thread_starts(monkeypatch, tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(
            max_retries=1,
            batch_timeout_seconds=60,
            movie=types.SimpleNamespace(enabled=False),
        ),
        cluster=types.SimpleNamespace(free_core_budget=0),
    )
    orch.slots = [Slot("gpu-node-01", 0)]
    orch.ledger = Ledger(Path(tmp_path) / "ledger.sqlite")
    orch.ledger.seed([Job(27, 0, 1)])
    orch._lock = threading.Lock()
    orch._write_metadata = lambda *args, **kwargs: None
    orch._stitch_movies = lambda: None
    orch._cleanup_calcache = lambda: None
    orch._cleanup_manifests = lambda: None
    orch._pick_slot = lambda free: free[0]

    class ImmediatePool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn, slot, job):
            claimed = orch.ledger.all()[0]
            assert claimed.status == RUNNING
            assert claimed.slot == str(slot)
            future = Future()
            future.set_result(fn(slot, job))
            return future

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(
        "gpu_subband_imaging.orchestrator.ThreadPoolExecutor", ImmediatePool
    )
    monkeypatch.setattr(orch, "_run_one", lambda slot, job: (
        orch.ledger.mark_done(job) or True
    ))

    orch.run(poll=0)
