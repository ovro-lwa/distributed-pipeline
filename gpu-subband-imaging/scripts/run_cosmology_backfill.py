#!/usr/bin/env python3
"""Run available cosmology dates sequentially with same-date calibration."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import yaml


DEFAULT_BANDS = [23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
DEFAULT_DATA_ROOT = Path("/lustre/pipeline/cosmology")
DEFAULT_CAL_ROOT = Path("/lustre/pipeline/calibration/results")
DEFAULT_OUTPUT_ROOT = Path("/lustre/pipeline/snapshots_gpu_pipeline")
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "run_logs" / "cosmology"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CalCandidate:
    hour: int
    path: Path


def _parse_cal_hour(path: Path) -> int:
    for part in path.parts:
        if part.endswith("h") and part[:-1].isdigit():
            return int(part[:-1])
    return -1


def discover_dates(data_root: Path, bands: Sequence[int],
                   patterns: Sequence[str]) -> List[str]:
    """Return sorted dates present in at least one requested subband."""
    dates = {
        path.name
        for band in bands
        for pattern in patterns
        for path in (data_root / f"{band}MHz").glob(pattern)
        if path.is_dir()
    }
    return sorted(dates)


def find_calibrations(cal_root: Path, date: str) -> List[CalCandidate]:
    candidates = []
    for path in (cal_root / date).glob("*h/successful/*/tables/*flagged"):
        if path.is_dir():
            candidates.append(CalCandidate(_parse_cal_hour(path), path))
    return sorted(candidates, key=lambda candidate: (candidate.hour,
                                                       str(candidate.path)))


def first_calibration(cal_root: Path, date: str) -> Optional[CalCandidate]:
    candidates = find_calibrations(cal_root, date)
    return candidates[0] if candidates else None


def write_config(template_dir: Path, output_dir: Path, date: str, bp: Path,
                 data_root: Path, output_root: Path,
                 bands: Sequence[int], hours: Optional[Sequence[str]] = None,
                 state_dir: Optional[Path] = None) -> Path:
    """Write one date config while preserving cluster resource isolation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cluster.yaml", "subbands.yaml"):
        shutil.copy2(template_dir / name, output_dir / name)

    pipeline = yaml.safe_load((template_dir / "pipeline.yaml").read_text())
    pipeline["date"] = date
    pipeline["hours"] = "all" if hours is None else list(hours)
    pipeline["bands"] = list(bands)
    pipeline["data_root"] = str(data_root)
    pipeline["output_root"] = str(output_root)
    pipeline["cal"]["bp"] = str(bp)
    (output_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(pipeline, sort_keys=False)
    )
    if state_dir is not None:
        cluster_path = output_dir / "cluster.yaml"
        cluster = yaml.safe_load(cluster_path.read_text())
        cluster.setdefault("control", {})["state_dir"] = str(state_dir)
        cluster_path.write_text(yaml.safe_dump(cluster, sort_keys=False))
    return output_dir


def run_pipeline(config_dir: Path, date: str, log_root: Path) -> int:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"run_cosmology_{date}.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    cmd = [sys.executable, "-m", "gpu_subband_imaging.cli", "run",
           "--config", str(config_dir.resolve())]

    print("Launching:", " ".join(cmd), flush=True)
    print("Logging to:", log_path, flush=True)
    with log_path.open("a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def acquire_run_lock(path: Path, blocking: bool):
    """Acquire the shared daily/backfill lock, or return None if busy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    if blocking:
        print(f"Waiting for cluster run lock: {path}", flush=True)
    try:
        fcntl.flock(handle, flags)
    except BlockingIOError:
        handle.close()
        return None
    if blocking:
        print(f"Acquired cluster run lock: {path}", flush=True)
    return handle


def release_run_lock(handle) -> None:
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="config-cosmology-template")
    parser.add_argument("--output-config", default="config-cosmology-auto")
    parser.add_argument("--date-glob", action="append", dest="date_globs",
                        help="repeatable date glob; defaults to May and June 2026")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cal-root", default=str(DEFAULT_CAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--bands", type=int, nargs="+", default=DEFAULT_BANDS)
    parser.add_argument("--hours", nargs="+",
                        help="UTC hours to process; defaults to every available hour")
    parser.add_argument("--state-dir",
                        help="override cluster state directory, useful for reprocessing")
    parser.add_argument("--lock-path",
                        help="shared lock path; defaults under output_root/_state")
    parser.add_argument("--plan", action="store_true",
                        help="print date/calibration choices without writing or running")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    cal_root = Path(args.cal_root)
    date_globs = args.date_globs or ["2026-05-*", "2026-06-*"]
    dates = discover_dates(data_root, args.bands, date_globs)
    if not dates:
        print(f"No cosmology dates match {date_globs!r} under {data_root}")
        return 0

    runnable = []
    for date in dates:
        calibration = first_calibration(cal_root, date)
        if calibration is None:
            print(f"SKIP {date}: no successful same-date flagged calibration",
                  flush=True)
            continue
        print(f"PLAN {date}: {calibration.path}", flush=True)
        runnable.append((date, calibration.path))

    if args.plan:
        print(f"Runnable dates: {len(runnable)}; skipped: {len(dates) - len(runnable)}")
        return 0

    failures = []
    lock_path = (Path(args.lock_path) if args.lock_path
                 else Path(args.output_root) / "_state" / "pipeline.lock")
    for date, bp in runnable:
        config_dir = write_config(
            Path(args.template), Path(args.output_config), date, bp,
            data_root, Path(args.output_root), args.bands,
            hours=args.hours,
            state_dir=Path(args.state_dir) if args.state_dir else None,
        )
        lock = acquire_run_lock(lock_path, blocking=True)
        assert lock is not None
        try:
            rc = run_pipeline(config_dir, date, Path(args.log_root))
        finally:
            release_run_lock(lock)
        if rc:
            failures.append((date, rc))
            print(f"Pipeline for {date} exited with rc={rc}; continuing",
                  flush=True)
        time.sleep(1.0)

    if failures:
        print("Failed dates: " + ", ".join(f"{date}(rc={rc})"
                                             for date, rc in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
