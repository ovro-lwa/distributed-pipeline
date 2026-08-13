from types import SimpleNamespace

from gpu_subband_imaging import worklist
from gpu_subband_imaging.worklist import _chunk


def test_chunk_splits_evenly_and_remainder():
    assert _chunk(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunk_empty():
    assert _chunk([], 50) == []


def test_list_inputs_includes_archives_and_raw_measurement_sets(monkeypatch):
    output = (
        "/data/27MHz/2026-04-19/05/a.ms\n"
        "/data/27MHz/2026-04-19/05/b.ms.tar\n"
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(worklist.subprocess, "run", fake_run)

    inputs = worklist._list_inputs(
        "gpu-node", 27, "/data/27MHz/2026-04-19", ["05"]
    )

    assert inputs == [
        "/data/27MHz/2026-04-19/05/a.ms",
        "/data/27MHz/2026-04-19/05/b.ms.tar",
    ]
    assert "*.ms.tar" in calls[0][-1]
    assert "*.ms" in calls[0][-1]
