#!/usr/bin/env python3
"""Wait for daily calibration, write a run config, then launch the pipeline."""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence


DEFAULT_CAL_ROOT = Path("/lustre/pipeline/calibration/results")
DEFAULT_OUTPUT_ROOT = "/lustre/pipeline/snapshots_gpu_pipeline"
DEFAULT_LOG_ROOT = Path("/lustre/pipeline/snapshots_gpu_pipeline/run_logs")
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


def find_calibrations(cal_root: Path, date: str) -> List[CalCandidate]:
    root = cal_root / date
    candidates = []
    for path in root.glob("*h/successful/*/tables/*flagged"):
        if path.is_dir():
            candidates.append(CalCandidate(_parse_cal_hour(path), path))
    return sorted(candidates, key=lambda c: (c.hour, str(c.path)))


def default_calibration(candidates: Sequence[CalCandidate]) -> CalCandidate:
    if not candidates:
        raise RuntimeError("no successful calibration tables found")
    return sorted(candidates, key=lambda c: (c.hour, str(c.path)))[-1]


def deadline_utc(date: str, hour: int) -> datetime:
    day = datetime.strptime(date, "%Y-%m-%d").date()
    return datetime.combine(day, dt_time(hour=hour), tzinfo=timezone.utc)


def next_date(date: str) -> str:
    day = datetime.strptime(date, "%Y-%m-%d").date()
    return (day + timedelta(days=1)).isoformat()


def prompt_for_calibration(candidates: Sequence[CalCandidate],
                           deadline: datetime) -> CalCandidate:
    if not sys.stdin.isatty():
        choice = default_calibration(candidates)
        print(f"stdin is not interactive; choosing {choice.path}", flush=True)
        return choice

    while True:
        print("\nSuccessful calibration tables:", flush=True)
        for i, cand in enumerate(candidates, start=1):
            print(f"  {i}. {cand.hour:02d}h  {cand.path}", flush=True)
        fallback = default_calibration(candidates)
        remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
        print(f"Choose 1-{len(candidates)} before {deadline.isoformat()} UTC.", flush=True)
        print(f"Default at deadline: {fallback.path}", flush=True)
        print("> ", end="", flush=True)

        ready, _, _ = select.select([sys.stdin], [], [], remaining)
        if not ready:
            print(f"\nNo answer before deadline; choosing {fallback.path}", flush=True)
            return fallback

        answer = sys.stdin.readline().strip()
        try:
            idx = int(answer)
        except ValueError:
            print(f"Invalid choice {answer!r}", flush=True)
            continue
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print(f"Choice must be between 1 and {len(candidates)}", flush=True)


def wait_for_calibration(cal_root: Path, date: str, deadline_hour: int,
                         poll_seconds: int) -> CalCandidate:
    deadline = deadline_utc(date, deadline_hour)
    while True:
        candidates = find_calibrations(cal_root, date)
        if len(candidates) == 1:
            choice = candidates[0]
            print(f"Found one successful calibration table; choosing {choice.path}",
                  flush=True)
            return choice
        if len(candidates) >= 2:
            return prompt_for_calibration(candidates, deadline)

        now = datetime.now(timezone.utc)
        if now >= deadline:
            if candidates:
                choice = default_calibration(candidates)
                print(f"Deadline reached; choosing {choice.path}", flush=True)
                return choice
            raise RuntimeError(f"no successful calibration tables found by {deadline}")

        seen = ", ".join(str(c.path) for c in candidates) or "none"
        sleep_for = min(float(poll_seconds), (deadline - now).total_seconds())
        print(f"Waiting for calibration tables for {date}; found: {seen}", flush=True)
        time.sleep(max(1.0, sleep_for))


def replace_yaml_scalar_line(text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*:\s*)"
        rf"(?P<old>[^#\r\n]*?)"
        rf"(?P<comment>\s+#.*)?"
        rf"(?P<newline>\r?\n?)$"
    )
    changed = False
    out = []
    for line in text.splitlines(keepends=True):
        match = pattern.match(line)
        if match and not changed:
            comment = match.group("comment") or ""
            newline = match.group("newline") or ""
            out.append(f"{match.group('prefix')}{value}{comment}{newline}")
            changed = True
        else:
            out.append(line)
    if not changed:
        raise RuntimeError(f"could not find {key!r} in pipeline.yaml")
    return "".join(out)


def write_config(template_dir: Path, output_dir: Path, date: str,
                 bp: str, output_root: Optional[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cluster.yaml", "subbands.yaml"):
        shutil.copy2(template_dir / name, output_dir / name)

    pipeline_path = template_dir / "pipeline.yaml"
    pipeline_text = pipeline_path.read_text()
    pipeline_text = replace_yaml_scalar_line(pipeline_text, "date", f'"{date}"')
    if output_root is not None:
        pipeline_text = replace_yaml_scalar_line(pipeline_text, "output_root",
                                                 output_root)
    pipeline_text = replace_yaml_scalar_line(pipeline_text, "bp", bp)
    (output_dir / "pipeline.yaml").write_text(pipeline_text)
    return output_dir


def run_pipeline(config_dir: Path, date: str, log_root: Path) -> int:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"run_auto_{date}.log"
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


def acquire_run_lock(path: Path):
    """Wait for exclusive cluster access shared with backfill launchers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    print(f"Waiting for cluster run lock: {path}", flush=True)
    fcntl.flock(handle, fcntl.LOCK_EX)
    print(f"Acquired cluster run lock: {path}", flush=True)
    return handle


def release_run_lock(handle) -> None:
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="UTC date, e.g. 2026-07-10")
    ap.add_argument("--template", default="config-allbands",
                    help="config directory to copy YAML from")
    ap.add_argument("--output-config", default="config-allsubbands-auto",
                    help="config directory to write/update")
    ap.add_argument("--cal-root", default=str(DEFAULT_CAL_ROOT))
    ap.add_argument("--deadline-hour-utc", type=int, default=23)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--lock-path",
                    help="shared lock path; defaults under output_root/_state")
    ap.add_argument("--no-run", action="store_true",
                    help="write the config but do not launch the pipeline")
    ap.add_argument("--once", action="store_true",
                    help="process only --date and exit instead of waiting for tomorrow")
    args = ap.parse_args(argv)

    date = args.date
    while True:
        choice = wait_for_calibration(Path(args.cal_root), date,
                                      args.deadline_hour_utc, args.poll_seconds)
        config_dir = write_config(Path(args.template), Path(args.output_config),
                                  date, str(choice.path), args.output_root)
        print(f"Wrote config: {config_dir}", flush=True)
        print(f"Using bandpass: {choice.path}", flush=True)

        if args.no_run:
            rc = 0
        else:
            lock_path = (Path(args.lock_path) if args.lock_path
                         else Path(args.output_root) / "_state" / "pipeline.lock")
            lock = acquire_run_lock(lock_path)
            try:
                rc = run_pipeline(config_dir, date, Path(args.log_root))
            finally:
                release_run_lock(lock)
        if args.once or args.no_run:
            return rc
        if rc:
            print(f"Pipeline for {date} exited with rc={rc}; continuing to next date",
                  flush=True)
        date = next_date(date)
        print(f"Waiting for next date: {date}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
