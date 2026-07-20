from pathlib import Path

from gpu_subband_imaging.ledger import DONE, FAILED, SKIPPED, Job, Ledger


def test_seed_and_transitions(tmp_path: Path):
    led = Ledger(tmp_path / "l.sqlite")
    jobs = [Job(73, 0, 50), Job(73, 1, 30), Job(41, 0, 50)]
    led.seed(jobs)
    assert len(led.pending()) == 3

    led.mark_running(jobs[0], "gpu-node-01:gpu0")
    led.mark_done(jobs[0])
    led.mark_failed(jobs[1], "boom")
    # done drops out of pending; failed stays (for retry)
    keys = {j.key for j in led.pending(include_running=False)}
    assert "73:0" not in keys and "73:1" in keys and "41:0" in keys
    assert led.summary().get(DONE) == 1
    assert led.summary().get(FAILED) == 1


def test_seed_is_idempotent(tmp_path: Path):
    led = Ledger(tmp_path / "l.sqlite")
    led.seed([Job(73, 0, 50)])
    led.mark_done(Job(73, 0, 50))
    led.seed([Job(73, 0, 50)])  # resume: must not reset the done job
    assert led.summary().get(DONE) == 1


def test_seed_many_jobs_in_one_call(tmp_path: Path):
    led = Ledger(tmp_path / "l.sqlite")
    jobs = [Job(73, i, 45) for i in range(500)]
    led.seed(jobs)
    assert led.summary().get("queued") == 500


def test_ledger_uses_delete_journal_mode(tmp_path: Path):
    led = Ledger(tmp_path / "l.sqlite")
    with led._lock:
        mode = led._c.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "delete"


def test_skipped_jobs_are_terminal_not_pending(tmp_path: Path):
    led = Ledger(tmp_path / "l.sqlite")
    job = Job(73, 0, 50)
    led.seed([job])
    led.mark_skipped(job, "exhausted retries")
    assert led.pending(include_running=False) == []
    assert led.summary().get(SKIPPED) == 1


def test_ledger_used_from_threads(tmp_path):
    """Regression: ledger must be usable from pool threads (check_same_thread)."""
    from concurrent.futures import ThreadPoolExecutor
    led = Ledger(tmp_path / "l.sqlite")
    jobs = [Job(b, 0, 10) for b in range(20)]
    led.seed(jobs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda j: led.mark_done(j), jobs))
    assert led.summary().get(DONE) == 20
