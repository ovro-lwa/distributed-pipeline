import types

from gpu_subband_imaging import dispatch
from gpu_subband_imaging.config import Slot
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
