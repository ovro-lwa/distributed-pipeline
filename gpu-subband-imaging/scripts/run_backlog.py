#!/usr/bin/env python
"""Re-process night-time averaged backlog nights with the GPU pipeline, one LST hour per run.

For each --dates night: the bandpass is the one the main pipeline (exopipe phase1) used,
<cal-root>/<date>/02h/successful/<newest>/tables/calibration_<date>_02h.B.flagged, and the
LST hours are those exopipe phase1 has for the date (or --lst-hours).  Each hour is one
``gsi run`` with a config written from --template (date, lst_hours, run_label, bandpass),
under the same cluster lock as scripts/run_nightly_lst.py, and only while there is time
before --stop-utc (the nightly run needs the cluster from about 06:30 UTC).  Done hours
are recorded in <output_root>/_state/backlog_<run_label>.json, so a re-run continues.
Inputs (night-time/averaged, exopipe) are only read.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from run_nightly_lst import run_gsi, write_config  # noqa: E402

DEFAULT_CAL_ROOT = "/lustre/pipeline/calibration/results"
DEFAULT_EXOPIPE = "/lustre/pipeline/exopipe/phase1"


def exopipe_bandpass(cal_root: str, date: str) -> Optional[str]:
    """The date's 02h bandpass (newest successful solve), as the exopipe phase1 runs used."""
    found = sorted(Path(cal_root, date, "02h", "successful").glob(
        f"*/tables/calibration_{date}_02h.B.flagged"))
    return str(found[-1]) if found else None


def exopipe_hours(root: str, date: str) -> List[int]:
    return sorted(int(p.parent.name[:-1]) for p in Path(root).glob(f"*h/{date}") if p.is_dir())


def exopipe_run(root: str, date: str, hour: int) -> Optional[str]:
    """Newest exopipe phase1 run label of an LST hour (the products a re-run is compared with)."""
    runs = sorted(p.name for p in Path(root, f"{hour:02d}h", date).glob("Run_*") if p.is_dir())
    return runs[-1] if runs else None


def load_badants(json_dir: str) -> Dict[str, Counter]:
    """{"<LST>h/<date>/<run>": Counter(csv -> n MSes)} merged over the per-log JSON files
    (lists exopipe's workers logged as "Flagging bad antennas")."""
    out: Dict[str, Counter] = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        for key, lists in json.loads(f.read_text()).items():
            out.setdefault(key, Counter()).update(lists)
    return out


def write_backlog_config(template: Path, out_dir: Path, date: str, hour: int, run_label: str,
                         bp: str, badants: Optional[str], note: str) -> Path:
    cfg = write_config(template, out_dir, date, [hour], run_label, bp)
    d = yaml.safe_load((cfg / "pipeline.yaml").read_text())
    if badants is not None:
        d["cal"]["badants"] = badants
    d["description"] = (d.get("description", "") + " " + note).strip()
    (cfg / "pipeline.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    return cfg


def may_start(now: datetime, stop_utc: str, resume_utc: str, need_min: float) -> bool:
    """A run of need_min minutes may start now: outside the nightly's hours
    [stop_utc, resume_utc) and ending before the next stop_utc."""
    def at(hhmm: str) -> datetime:
        hh, mm = map(int, hhmm.split(":"))
        return now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    stop, resume = at(stop_utc), at(resume_utc)
    if stop <= now < resume:
        return False
    if stop <= now:
        stop += timedelta(days=1)
    return now + timedelta(minutes=need_min) <= stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="config dir (pipeline/cluster/subbands.yaml)")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--output-config", required=True, help="config dir written for each run")
    ap.add_argument("--lst-hours", type=int, nargs="*", help="default: exopipe phase1 hours of the date")
    ap.add_argument("--cal-root", default=DEFAULT_CAL_ROOT)
    ap.add_argument("--exopipe-root", default=DEFAULT_EXOPIPE)
    ap.add_argument("--stop-utc", default="05:30", help="no run may end after this (nightly needs the cluster)")
    ap.add_argument("--resume-utc", default="15:00", help="earliest start after the nightly run")
    ap.add_argument("--run-minutes", type=float, default=60.0, help="expected wall time of one LST hour")
    ap.add_argument("--ignore-window", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="print the plan and the first config only")
    ap.add_argument("--lock-path")
    ap.add_argument("--badants-dir", help="JSON lists of exopipe's bad antennas per LST/date/run; "
                                          "hours without one are skipped")
    ap.add_argument("--allow-anthealth", action="store_true",
                    help="without --badants-dir: use the live anthealth lookup (the current list)")
    a = ap.parse_args(argv)
    if not a.badants_dir and not a.allow_anthealth:
        ap.error("give --badants-dir (exopipe's lists) or --allow-anthealth")
    badants = load_badants(a.badants_dir) if a.badants_dir else {}

    template = Path(a.template)
    tpl = yaml.safe_load((template / "pipeline.yaml").read_text())
    out_root = Path(tpl["output_root"])
    state_path = out_root / "_state" / f"backlog_{a.run_label}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "runs": []}
    log_dir = out_root / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(a.lock_path) if a.lock_path else out_root / "_state" / "pipeline.lock"

    plan = []
    for date in a.dates:
        bp = exopipe_bandpass(a.cal_root, date)
        hours = a.lst_hours if a.lst_hours else exopipe_hours(a.exopipe_root, date)
        if not bp or not hours:
            print(f"{date}: skipped (bandpass {bp}, LST hours {hours})", flush=True)
            continue
        done = set(state["done"].get(date, []))
        print(f"{date}: bandpass {bp}; LST hours {hours}; {len(done)} done", flush=True)
        for h in hours:
            if h in done:
                continue
            run = exopipe_run(a.exopipe_root, date, h)
            note = f"Compare with exopipe phase1 {h:02d}h/{date}/{run}."
            ba = None
            if a.badants_dir:
                lists = badants.get(f"{h:02d}h/{date}/{run}")
                if not lists:
                    print(f"  {h:02d}h: no bad-antenna list logged for exopipe {run}; skipped", flush=True)
                    continue
                ba, n = lists.most_common(1)[0]
                note += f" Bad antennas: exopipe {run}'s list ({n} of {sum(lists.values())} MSes)."
                print(f"  {h:02d}h: exopipe {run}, {len(ba.split(','))} bad antennas "
                      f"({len(lists)} distinct list(s) in its log)", flush=True)
            plan.append((date, h, bp, ba, note))
    if a.no_run:
        if plan:
            date, h, bp, ba, note = plan[0]
            cfg = write_backlog_config(template, Path(a.output_config), date, h, a.run_label, bp, ba, note)
            print((cfg / "pipeline.yaml").read_text(), flush=True)
        return 0

    for date, h, bp, ba, note in plan:
        now = datetime.now(timezone.utc)
        if not a.ignore_window and not may_start(now, a.stop_utc, a.resume_utc, a.run_minutes):
            print(f"{now:%H:%M} UTC: not enough time before {a.stop_utc} UTC (or inside "
                  f"{a.stop_utc}-{a.resume_utc}); stopping, {len(plan)} hour(s) left in the plan",
                  flush=True)
            return 0
        cfg = write_backlog_config(template, Path(a.output_config), date, h, a.run_label, bp, ba, note)
        with open(lock_path, "a+") as lock:
            print(f"{now:%H:%M:%S} {date} LST {h:02d}h: waiting for cluster lock {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            t0 = time.time()
            rc = run_gsi(cfg, log_dir / f"backlog_{a.run_label}_{date}.log")
            fcntl.flock(lock, fcntl.LOCK_UN)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {date} LST {h:02d}h rc={rc} in {time.time() - t0:.0f} s",
              flush=True)
        if rc == 0:
            state["done"].setdefault(date, []).append(h)
        state["runs"].append(dict(date=date, lst=h, rc=rc, bp=bp, start=now.isoformat(),
                                  wall_s=time.time() - t0))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
