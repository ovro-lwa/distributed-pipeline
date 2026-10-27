#!/usr/bin/env python
"""Process one night by LST hour as the data arrive (layout=lst).

Every --poll-seconds: an LST hour of --date whose Sun-cut window (lst.sun_ok) ended
at least --margin-min ago is ready; all ready hours not yet done go into one
``gsi run`` (config written from --template with date, lst_hours, run_label and the
newest successful bandpass under --cal-root at that moment).  The run label is
fixed for the night, so every run writes under
<output_root>/<LST>h/<date>/<run_label>/<band>MHz/.  Done hours are recorded in
<output_root>/_state/nightly_<date>.json; the script exits when every night hour
of the date is done (or failed --max-attempts times).  With --wait-cal the bandpass is the
date's own LST calibration (main 22h, else backup 21h; see tonight_bandpass), fixed for the night.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from gpu_subband_imaging import lst  # noqa: E402

DEFAULT_CAL_ROOT = "/lustre/pipeline/calibration/results"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def night_windows(date: str) -> Dict[int, Tuple[datetime, datetime]]:
    """{LST hour: (first, last) minute of the date passing the Sun cut}."""
    day = datetime.strptime(date, "%Y-%m-%d")
    names = [(day + timedelta(minutes=m)).strftime("%Y%m%d_%H%M%S") for m in range(24 * 60)]
    ok = lst.sun_ok(names)
    hours = lst.lst_hours(names)
    out: Dict[int, Tuple[datetime, datetime]] = {}
    for n, k, h in zip(names, ok, hours):
        if not k:
            continue
        t = lst.frame_time(n)
        a, b = out.get(h, (t, t))
        out[h] = (min(a, t), max(b, t))
    return out


def latest_bandpass(cal_root: str, date: str, days_back: int = 3) -> Optional[str]:
    """Newest successful *.B.flagged table for date or up to days_back earlier."""
    day = datetime.strptime(date, "%Y-%m-%d")
    for back in range(days_back + 1):
        d = Path(cal_root) / (day - timedelta(days=back)).strftime("%Y-%m-%d")
        found = sorted(d.glob("*h/successful/*/tables/*.B.flagged"),
                       key=lambda p: (p.parts[-5], p.parts[-3]))
        if found:
            return str(found[-1])
    return None


def tonight_bandpass(cal_root: str, date: str, state_dir: str, deadline: datetime,
                     main: str = "22h", backup: str = "21h",
                     rejected: frozenset = frozenset()) -> Tuple[Optional[str], str]:
    """Bandpass from the date's own LST calibration (run_daily_cal_lst.py): the main one when it
    succeeded, the backup when the main one is finished but failed or was rejected (`rejected`:
    tables that failed check_table); after `deadline`, or when neither is usable, the newest
    table of the previous days.  (None, reason) means wait."""
    def ok(lst_label: str) -> Optional[str]:
        found = sorted(Path(cal_root, date, lst_label, "successful").glob("*/tables/*.B.flagged"))
        return str(found[-1]) if found and str(found[-1]) not in rejected else None

    def finished(role: str, lst_label: str) -> bool:
        if any(f"/{date}/{lst_label}/" in r for r in rejected):
            return True
        p = Path(state_dir) / f"{date}_{role}.json"
        return p.exists() and json.loads(p.read_text()).get("status") is not None

    prev = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    if ok(main):
        return ok(main), f"{date} {main} calibration"
    if finished("main", main) and ok(backup):
        return ok(backup), f"{date} {backup} backup calibration (main failed or rejected)"
    if finished("main", main) and finished("backup", backup):
        return latest_bandpass(cal_root, prev), f"no usable {date} calibration; newest earlier table"
    if utcnow() >= deadline:
        bp = ok(backup) or latest_bandpass(cal_root, prev)
        return bp, f"calibration deadline {deadline:%H:%M} UTC passed; using {'backup' if ok(backup) else 'newest earlier'}"
    return None, f"waiting for the {date} {main} calibration"


def check_table(bp: str, ref: Optional[str], out_dir: Path, amp=(0.8, 1.25), max_phase_deg=15.0,
                max_flag=0.5) -> Tuple[bool, str]:
    """scripts/check_bandpass.py of bp against ref (an earlier night's table): every band both
    tables have must have a median gain ratio within `amp`, a phase-difference scatter below
    max_phase_deg and a flagged fraction below max(max_flag, ref + 0.2).  No ref -> accepted."""
    if not ref:
        return True, "no reference table; not checked"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "check_bandpass.py"), bp, ref,
                         "--out", str(out_dir)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
    (out_dir / "check_bandpass.log").write_text(rc.stdout)
    if rc.returncode != 0:
        return False, f"check_bandpass failed rc={rc.returncode} (see {out_dir})"
    rows = next(iter(json.loads((out_dir / "bandpass_check.json").read_text())["comparisons"].values()))
    bad = []
    for c in rows:
        a, ph = c["amp_ratio_median"], c["phase_diff_scatter_deg"]
        if a is None or not (amp[0] <= a <= amp[1]) or ph is None or ph > max_phase_deg \
                or c["flag_new"] > max(max_flag, c["flag_ref"] + 0.2):
            bad.append(f"{c['freq_mhz']:.0f} MHz (amp {a}, phase {ph}, flag {c['flag_new']:.2f})")
    if len(rows) < 6:
        return False, f"only {len(rows)} bands in common with {ref}"
    if bad:
        return False, f"{len(bad)}/{len(rows)} bands off vs {Path(ref).name}: " + "; ".join(bad)
    return True, f"{len(rows)} bands consistent with {Path(ref).name}"


def write_config(template: Path, out_dir: Path, date: str, hours: List[int],
                 run_label: str, bp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cluster.yaml", "subbands.yaml"):
        (out_dir / name).write_text((template / name).read_text())
    d = yaml.safe_load((template / "pipeline.yaml").read_text())
    d.update(date=date, lst_hours=sorted(hours), run_label=run_label, layout="lst", hours="all")
    d["cal"]["bp"] = bp
    (out_dir / "pipeline.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    return out_dir


def run_gsi(config_dir: Path, log_path: Path, cmd: str = "run") -> int:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "gpu_subband_imaging.cli", cmd, "--config", str(config_dir)]
    print("Launching:", " ".join(cmd), "->", log_path, flush=True)
    with open(log_path, "a") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def wideband(run_dir: Path, log_path: Path) -> None:
    """Main pipeline's pipeline/wideband_summary.py on one LST hour (V rms vs band, stacks, 3-colour)."""
    script = REPO_ROOT.parent / "pipeline" / "wideband_summary.py"
    if not run_dir.is_dir() or not script.exists():
        return
    with open(log_path, "a") as log:
        rc = subprocess.run([sys.executable, str(script), str(run_dir)], stdout=log,
                            stderr=subprocess.STDOUT).returncode
    print(f"wideband summary {run_dir} rc={rc}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--template", required=True, help="config dir to copy (layout=lst settings)")
    ap.add_argument("--output-config", required=True, help="config dir to write for each run")
    ap.add_argument("--run-label", default="Run_" + utcnow().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--cal-root", default=DEFAULT_CAL_ROOT)
    ap.add_argument("--bp", help="fixed bandpass table (default: newest at each launch)")
    ap.add_argument("--wait-cal", action="store_true",
                    help="use the date's own LST calibration (main, else backup), fixed for the night")
    ap.add_argument("--cal-state-dir", default="/fast/pipeline/calibration/watch_state_lst")
    ap.add_argument("--cal-deadline-utc", type=float, default=10.0,
                    help="--wait-cal: hour (UTC, on the date) after which to stop waiting")
    ap.add_argument("--no-check-bp", action="store_true",
                    help="--wait-cal: use tonight's table without comparing it with the previous night's")
    ap.add_argument("--hours", help="comma list of LST hours (default: every night hour)")
    ap.add_argument("--margin-min", type=float, default=10.0)
    ap.add_argument("--min-window-min", type=float, default=10.0,
                    help="skip LST hours with less Sun-cut data than this (e.g. a few minutes at dusk)")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--lock-path", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--no-wideband", action="store_true", help="skip pipeline/wideband_summary.py per LST hour")
    ap.add_argument("--no-run", action="store_true", help="print the plan and the first config, then exit")
    a = ap.parse_args(argv)

    template = Path(a.template).resolve()
    tpl = yaml.safe_load((template / "pipeline.yaml").read_text())
    out_root = Path(tpl["output_root"])
    # the orchestrator makes each LST hour's wideband summary itself (wideband_per_hour);
    # node caches kept across runs (calcache_cleanup: none / keep_ms_local) go after the night
    own_wideband = not tpl.get("wideband_per_hour", False)
    node_cleanup = tpl.get("calcache_cleanup", "run") == "none" or bool(tpl.get("keep_ms_local"))
    state_path = out_root / "_state" / f"nightly_{a.date}.json"
    log_dir = Path(a.log_dir) if a.log_dir else out_root / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(a.lock_path) if a.lock_path else out_root / "_state" / "pipeline.lock"

    windows = night_windows(a.date)
    short = {h: w for h, w in windows.items() if (w[1] - w[0]).total_seconds() < 60 * a.min_window_min}
    for h, w in short.items():
        print(f"  skipping {lst.label(h)}: only {w[0]:%H:%M}-{w[1]:%H:%M} UTC after the Sun cut", flush=True)
        windows.pop(h)
    if a.hours:
        windows = {h: w for h, w in windows.items() if h in {int(x) for x in a.hours.split(",")}}
    print(f"night {a.date}: run label {a.run_label}; LST hours (UTC window after the Sun cut):", flush=True)
    for h in sorted(windows, key=lambda k: windows[k][0]):
        print(f"  {lst.label(h)}: {windows[h][0]:%H:%M} - {windows[h][1]:%H:%M} UTC", flush=True)

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.setdefault("run_label", a.run_label)
    state.setdefault("done", [])
    state.setdefault("attempts", {})
    run_label = state["run_label"]           # keep one label for the night across restarts

    while True:
        now = utcnow()
        ready = [h for h, (_, end) in windows.items()
                 if now >= end + timedelta(minutes=a.margin_min + 1)
                 and h not in state["done"] and state["attempts"].get(str(h), 0) < a.max_attempts]
        remaining = [h for h in windows if h not in state["done"]
                     and state["attempts"].get(str(h), 0) < a.max_attempts]
        if not remaining:
            print(f"{now:%H:%M:%S} all night hours done: {sorted(state['done'])}", flush=True)
            if node_cleanup and (Path(a.output_config) / "pipeline.yaml").exists():
                rc = run_gsi(Path(a.output_config), log_dir / f"run_lst_{a.date}.log",
                             cmd="cleanup-nodes")
                print(f"{utcnow():%H:%M:%S} node cleanup (calibration caches, local kept MSes) "
                      f"rc={rc}", flush=True)
            report = log_dir / f"night_report_{a.date}.txt"
            with open(report, "w") as fh:
                subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "night_report.py"), str(out_root),
                                a.date, run_label, "--json", str(report.with_suffix(".json"))],
                               stdout=fh, stderr=subprocess.STDOUT)
            print(f"night report: {report}", flush=True)
            return 0
        if not ready:
            nxt = min(windows[h][1] for h in remaining) + timedelta(minutes=a.margin_min + 1)
            print(f"{now:%H:%M:%S} waiting; next hour ready at {nxt:%H:%M} UTC", flush=True)
            if a.no_run:
                return 0
            time.sleep(a.poll_seconds)
            continue
        ready.sort(key=lambda h: windows[h][0])
        if a.bp:
            bp = a.bp
        elif a.wait_cal:
            bp = state.get("bp")                 # one table for the whole night
            deadline = datetime.strptime(a.date, "%Y-%m-%d") + timedelta(hours=a.cal_deadline_utc)
            prev = (datetime.strptime(a.date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            while not bp:
                rejected = frozenset(state.setdefault("bp_rejected", {}))
                bp, why = tonight_bandpass(a.cal_root, a.date, a.cal_state_dir, deadline, rejected=rejected)
                print(f"{now:%H:%M:%S} bandpass: {why}: {bp}", flush=True)
                if not bp:
                    break
                if f"/{a.date}/" in bp and not a.no_check_bp:
                    # tonight's table: compare with the previous nights' before using it
                    good, msg = check_table(bp, latest_bandpass(a.cal_root, prev),
                                            log_dir / f"bpcheck_{a.date}" / Path(bp).parts[-5])
                    print(f"{now:%H:%M:%S} bandpass check: {'OK' if good else 'REJECTED'}: {msg}", flush=True)
                    if not good:
                        state["bp_rejected"][bp] = msg
                        state_path.parent.mkdir(parents=True, exist_ok=True)
                        state_path.write_text(json.dumps(state, indent=1))
                        bp = None
                        continue
                    why += f"; check: {msg}"
                state.update(bp=bp, bp_reason=why)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=1))
        else:
            bp = latest_bandpass(a.cal_root, a.date)
        if not bp:
            print("no bandpass table found; waiting", flush=True)
            time.sleep(a.poll_seconds)
            continue
        cfg = write_config(template, Path(a.output_config), a.date, ready, run_label, bp)
        print(f"{now:%H:%M:%S} LST hours {[lst.label(h) for h in ready]} ready; bandpass {bp}", flush=True)
        if a.no_run:
            print((cfg / "pipeline.yaml").read_text(), flush=True)
            return 0
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock:
            print(f"waiting for cluster lock {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            t0 = time.time()
            rc = run_gsi(cfg, log_dir / f"run_lst_{a.date}.log")
            fcntl.flock(lock, fcntl.LOCK_UN)
        print(f"{utcnow():%H:%M:%S} gsi run for {[lst.label(h) for h in ready]} rc={rc} "
              f"in {time.time() - t0:.0f} s", flush=True)
        if rc == 0 and not a.no_wideband and own_wideband:
            for h in ready:
                wideband(out_root / lst.label(h) / a.date / run_label, log_dir / f"wideband_{a.date}.log")
        for h in ready:
            state["attempts"][str(h)] = state["attempts"].get(str(h), 0) + 1
            if rc == 0:
                state["done"].append(h)
        state.setdefault("runs", []).append(dict(hours=ready, rc=rc, bp=bp, start=now.isoformat(),
                                                 wall_s=time.time() - t0))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    sys.exit(main())
