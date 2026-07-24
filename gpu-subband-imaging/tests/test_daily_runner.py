from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_allsubbands.py"
SPEC = importlib.util.spec_from_file_location("run_daily_allsubbands", SCRIPT)
daily = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = daily
SPEC.loader.exec_module(daily)


def _make_cal(root: Path, date: str, hour: int, stamp: str) -> Path:
    path = (
        root
        / date
        / f"{hour:02d}h"
        / "successful"
        / stamp
        / "tables"
        / f"calibration_{date}_{hour:02d}h.B.flagged"
    )
    path.mkdir(parents=True)
    return path


def test_find_calibrations_returns_successful_flagged_tables_sorted_by_hour(tmp_path):
    cal_root = tmp_path / "calibration" / "results"
    later = _make_cal(cal_root, "2026-07-10", 22, "20260710_130000")
    earlier = _make_cal(cal_root, "2026-07-10", 21, "20260710_100000")
    _make_cal(cal_root, "2026-07-09", 22, "20260709_130000")

    candidates = daily.find_calibrations(cal_root, "2026-07-10")

    assert [cand.path for cand in candidates] == [earlier, later]
    assert [cand.hour for cand in candidates] == [21, 22]


def test_default_calibration_uses_latest_successful_hour(tmp_path):
    cal_root = tmp_path / "calibration" / "results"
    _make_cal(cal_root, "2026-07-10", 21, "20260710_100000")
    latest = _make_cal(cal_root, "2026-07-10", 22, "20260710_130000")

    candidates = daily.find_calibrations(cal_root, "2026-07-10")

    assert daily.default_calibration(candidates).path == latest


def test_deadline_utc_uses_requested_date_at_utc_hour():
    deadline = daily.deadline_utc("2026-07-10", 23)

    assert deadline.isoformat() == "2026-07-10T23:00:00+00:00"


def test_next_date_advances_one_day():
    assert daily.next_date("2026-07-10") == "2026-07-11"


def test_wait_for_calibration_uses_one_successful_table_immediately(tmp_path):
    cal_root = tmp_path / "calibration" / "results"
    only = _make_cal(cal_root, "2026-07-10", 21, "20260710_181903")

    choice = daily.wait_for_calibration(
        cal_root,
        "2026-07-10",
        deadline_hour=23,
        poll_seconds=1,
    )

    assert choice.path == only


def test_write_config_preserves_template_yaml_style_and_edits_run_fields(tmp_path):
    template = tmp_path / "config-allbands"
    output = tmp_path / "config-auto"
    template.mkdir()
    (template / "cluster.yaml").write_text("control:\n  state_dir: /state\n")
    (template / "subbands.yaml").write_text("subbands: {}\n")
    (template / "pipeline.yaml").write_text(
        "# One run's parameters.\n"
        "\n"
        "date: \"2026-07-08\"\n"
        "hours: [\"06\", \"07\", \"08\", \"09\"]        # \"all\", or a list like [\"09\", \"10\"]\n"
        "bands: [23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]\n"
        "batch_size: 45             # worker recycles zest daemon every 25 files (CUDA hang workaround)\n"
        "batch_timeout_seconds: 21600\n"
        "max_retries: 1\n"
        "\n"
        "data_root: /lustre/pipeline/slow          # <root>/<band>MHz/<date>/<hour>/*.ms.tar\n"
        "output_root: /lustre/pipeline/images_test      # old output comment\n"
        "\n"
        "peel_overrides:\n"
        "  23: {maxiter: 20}\n"
        "  32: {maxiter: 20}\n"
        "\n"
        "cal:\n"
        "  bp: /old/calibration.B.flagged\n"
        "  xy: /calibration/xyphase.Xf\n"
    )
    bp = (
        "/lustre/pipeline/calibration/results/2026-07-10/22h/successful/"
        "run/tables/calibration_2026-07-10_22h.B.flagged"
    )

    daily.write_config(
        template,
        output,
        "2026-07-10",
        bp,
        "/lustre/pipeline/snapshots_gpu_pipeline",
    )

    text = (output / "pipeline.yaml").read_text()
    assert 'date: "2026-07-10"\n' in text
    assert 'hours: ["06", "07", "08", "09"]        # "all", or a list like ["09", "10"]\n' in text
    assert "bands: [23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]\n" in text
    assert "data_root: /lustre/pipeline/slow          # <root>/<band>MHz/<date>/<hour>/*.ms.tar\n" in text
    assert "output_root: /lustre/pipeline/snapshots_gpu_pipeline      # old output comment\n" in text
    assert "peel_overrides:\n  23: {maxiter: 20}\n  32: {maxiter: 20}\n" in text
    assert f"  bp: {bp}\n" in text
    assert (output / "cluster.yaml").read_text() == "control:\n  state_dir: /state\n"
    assert (output / "subbands.yaml").read_text() == "subbands: {}\n"


def test_daily_run_lock_is_exclusive(tmp_path):
    path = tmp_path / "pipeline.lock"
    first = daily.acquire_run_lock(path)

    try:
        import fcntl

        second = path.open("a+")
        try:
            try:
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
            assert not acquired
        finally:
            second.close()
    finally:
        daily.release_run_lock(first)
