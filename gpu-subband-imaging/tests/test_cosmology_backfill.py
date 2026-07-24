from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cosmology_backfill.py"
SPEC = importlib.util.spec_from_file_location("run_cosmology_backfill", SCRIPT)
backfill = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = backfill
SPEC.loader.exec_module(backfill)


def _make_cal(root: Path, date: str, hour: int, stamp: str) -> Path:
    path = (
        root / date / f"{hour:02d}h" / "successful" / stamp / "tables"
        / f"calibration_{date}_{hour:02d}h.B.flagged"
    )
    path.mkdir(parents=True)
    return path


def test_discover_dates_uses_union_across_requested_bands(tmp_path):
    (tmp_path / "23MHz" / "2026-05-05").mkdir(parents=True)
    (tmp_path / "73MHz" / "2026-05-06").mkdir(parents=True)
    (tmp_path / "73MHz" / "2026-06-01").mkdir(parents=True)

    dates = backfill.discover_dates(
        tmp_path, [23, 73], ["2026-05-*", "2026-06-*"]
    )

    assert dates == ["2026-05-05", "2026-05-06", "2026-06-01"]


def test_first_calibration_uses_earliest_successful_solution(tmp_path):
    later = _make_cal(tmp_path, "2026-05-05", 19, "20260505_180000")
    earliest = _make_cal(tmp_path, "2026-05-05", 18, "20260505_160000")

    selected = backfill.first_calibration(tmp_path, "2026-05-05")

    assert selected is not None
    assert selected.path == earliest
    assert selected.path != later


def test_write_config_selects_all_hours_and_cosmology_paths(tmp_path):
    template = tmp_path / "template"
    output = tmp_path / "output"
    template.mkdir()
    (template / "cluster.yaml").write_text("nodes: []\n")
    (template / "subbands.yaml").write_text("bands: {}\n")
    (template / "pipeline.yaml").write_text(
        "date: '2026-01-01'\n"
        "hours: ['06']\n"
        "bands: [73]\n"
        "data_root: /old/input\n"
        "output_root: /old/output\n"
        "cal:\n"
        "  bp: /old/table\n"
    )
    bp = Path("/cal/2026-05-05/18h/first.B.flagged")

    backfill.write_config(
        template, output, "2026-05-05", bp,
        Path("/lustre/pipeline/cosmology"),
        Path("/lustre/pipeline/snapshots_gpu_pipeline"),
        [23, 73],
    )

    pipeline = yaml.safe_load((output / "pipeline.yaml").read_text())
    assert pipeline["date"] == "2026-05-05"
    assert pipeline["hours"] == "all"
    assert pipeline["bands"] == [23, 73]
    assert pipeline["data_root"] == "/lustre/pipeline/cosmology"
    assert pipeline["output_root"] == "/lustre/pipeline/snapshots_gpu_pipeline"
    assert pipeline["cal"]["bp"] == str(bp)
    assert (output / "cluster.yaml").read_text() == "nodes: []\n"


def test_write_config_can_limit_hours_and_use_fresh_state(tmp_path):
    template = tmp_path / "template"
    output = tmp_path / "output"
    template.mkdir()
    (template / "cluster.yaml").write_text(
        "control:\n  state_dir: /old/state\nnodes: []\n"
    )
    (template / "subbands.yaml").write_text("bands: {}\n")
    (template / "pipeline.yaml").write_text(
        "date: '2026-01-01'\n"
        "hours: all\n"
        "bands: [73]\n"
        "data_root: /old/input\n"
        "output_root: /old/output\n"
        "cal:\n"
        "  bp: /old/table\n"
    )

    backfill.write_config(
        template, output, "2026-05-05", Path("/cal/table"),
        Path("/lustre/pipeline/cosmology"),
        Path("/lustre/pipeline/snapshots_gpu_pipeline"),
        [73], hours=["04", "05"],
        state_dir=Path("/lustre/pipeline/snapshots_gpu_pipeline/_state/repair"),
    )

    pipeline = yaml.safe_load((output / "pipeline.yaml").read_text())
    cluster = yaml.safe_load((output / "cluster.yaml").read_text())
    assert pipeline["hours"] == ["04", "05"]
    assert cluster["control"]["state_dir"].endswith("/_state/repair")


def test_nonblocking_run_lock_yields_while_daily_lock_is_held(tmp_path):
    path = tmp_path / "pipeline.lock"
    daily_lock = backfill.acquire_run_lock(path, blocking=True)

    try:
        assert backfill.acquire_run_lock(path, blocking=False) is None
    finally:
        backfill.release_run_lock(daily_lock)

    available = backfill.acquire_run_lock(path, blocking=False)
    assert available is not None
    backfill.release_run_lock(available)
