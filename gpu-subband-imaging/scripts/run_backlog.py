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
from run_nightly_lst import night_windows, run_gsi, write_config  # noqa: E402

DEFAULT_CAL_ROOT = "/lustre/pipeline/calibration/results"
DEFAULT_EXOPIPE = "/lustre/pipeline/exopipe/phase1"


def exopipe_bandpass(cal_root: str, date: str) -> Optional[str]:
    """The date's 02h bandpass (newest successful solve), as the exopipe phase1 runs used; a date
    without one (2024-12-18 has only 01h) falls back to its other LST calibrations, nearest first."""
    for lst_label in ["02h"] + [f"{h:02d}h" for h in (1, 3, 0, 4, 23, 5, 22)]:
        found = sorted(Path(cal_root, date, lst_label, "successful").glob(
            f"*/tables/calibration_{date}_{lst_label}.B.flagged"))
        if found:
            return str(found[-1])
    return None


def exopipe_hours(root: str, date: str) -> List[int]:
    return sorted(int(p.parent.name[:-1]) for p in Path(root).glob(f"*h/{date}") if p.is_dir())


def exopipe_run(root: str, date: str, hour: int, band: int) -> Optional[str]:
    """Newest exopipe phase1 run with this band-hour (the products a re-run is compared with;
    exopipe re-ran single bands under new run labels)."""
    runs = sorted(p.parent.name for p in Path(root, f"{hour:02d}h", date).glob(f"Run_*/{band}MHz")
                  if p.is_dir())
    return runs[-1] if runs else None


def load_badants(json_dir: str) -> Dict[str, Counter]:
    """{"<LST>h/<date>/<run>/<band>MHz": Counter(csv -> n MSes)} merged over the per-log JSON
    files (lists exopipe's workers logged as "Flagging bad antennas")."""
    out: Dict[str, Counter] = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        for key, lists in json.loads(f.read_text()).items():
            out.setdefault(key, Counter()).update(lists)
    return out


def write_backlog_config(template: Path, out_dir: Path, date: str, hours, run_label: str,
                         bp: str, bands: List[int], badants: Optional[Dict], note: str) -> Path:
    """hours: one LST hour or a list; badants keyed by band or "<LST>h/<band>"."""
    hours = [hours] if isinstance(hours, int) else list(hours)
    cfg = write_config(template, out_dir, date, hours, run_label, bp)
    d = yaml.safe_load((cfg / "pipeline.yaml").read_text())
    d["bands"] = bands
    if badants is not None:
        d["cal"]["badants"] = badants
    d["description"] = (d.get("description", "") + " " + note).strip()
    (cfg / "pipeline.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    return cfg


def nightly_busy(state_dir: Path) -> Optional[str]:
    """The date the nightly run is processing now, else None: a run_nightly_lst.py process that is
    running a round (its gsi run, config-lst-auto) or whose date already has rounds in
    <state_dir>/nightly_<date>.json (it releases the cluster lock between rounds, so the lock
    alone does not keep a backlog run out of its night)."""
    import re
    import subprocess
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    round_running = any("gpu_subband_imaging.cli run" in ln and "config-lst-auto" in ln
                        for ln in ps.splitlines())
    for line in ps.splitlines():
        m = re.search(r"run_nightly_lst\.py .*--date (\d{4}-\d\d-\d\d)", line)
        if m:
            if round_running:
                return m.group(1)
            f = state_dir / f"nightly_{m.group(1)}.json"
            try:
                if json.loads(f.read_text()).get("runs"):
                    return m.group(1)
            except (OSError, ValueError):
                pass
    return None


def nightly_start(margin_min: float = 10.0) -> Optional[datetime]:
    """When the waiting nightly (run_nightly_lst.py with nightly_wait_all) will start: the end of
    its date's last Sun-cut LST window plus its margin (UTC, aware); None without a nightly."""
    import re
    import subprocess
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    dates = re.findall(r"run_nightly_lst\.py .*--date (\d{4}-\d\d-\d\d)", ps)
    if not dates:
        return None
    # as run_nightly_lst.py: LST hours with < 10 min of Sun-cut data are skipped
    w = {h: v for h, v in night_windows(min(dates)).items() if (v[1] - v[0]).total_seconds() >= 600}
    if not w:
        return None
    last = max(end for _, end in w.values()) + timedelta(minutes=margin_min + 1)
    return last.replace(tzinfo=timezone.utc)


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
    ap.add_argument("--hours-per-run", type=int, default=1,
                    help="LST hours of one date per gsi run (more jobs than GPU slots pack better)")
    ap.add_argument("--ignore-window", action="store_true")
    ap.add_argument("--stop-before-nightly", action="store_true",
                    help="instead of --stop-utc/--resume-utc: start a run only if it ends before the "
                         "waiting nightly starts (all its data in; template nightly_wait_all)")
    ap.add_argument("--loop", action="store_true",
                    help="wait (instead of exiting) while outside the window or the nightly is "
                         "mid-night, until every planned hour is done")
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
            bands, ba, runs, missing, multi = [], {}, {}, [], []
            for band in tpl["bands"]:
                run = exopipe_run(a.exopipe_root, date, h, band)
                if run is None:
                    missing.append(f"{band}(no exopipe run)")
                    continue
                if a.badants_dir:
                    lists = badants.get(f"{h:02d}h/{date}/{run}/{band}MHz")
                    if not lists:
                        missing.append(f"{band}(no list for {run})")
                        continue
                    ba[band] = lists.most_common(1)[0][0]
                    if len(lists) > 1:
                        multi.append(band)
                bands.append(band)
                runs.setdefault(run, []).append(band)
            if not bands:
                print(f"  {h:02d}h: no band to run ({', '.join(missing)}); skipped", flush=True)
                continue
            note = ("Compare with exopipe phase1 " + "; ".join(
                f"{h:02d}h/{date}/{r} ({','.join(map(str, b))} MHz)" for r, b in runs.items()) + ".")
            if a.badants_dir:
                note += " Bad antennas per band: the list exopipe logged for that band-hour's run."
            print(f"  {h:02d}h: {len(bands)} bands, exopipe runs {sorted(runs)}"
                  + (f"; left out {missing}" if missing else "")
                  + (f"; several logged lists (most common used) for {multi}" if multi else ""),
                  flush=True)
            plan.append((date, h, bp, bands, ba if a.badants_dir else None, note))
    groups = group_plan(plan, a.hours_per_run)
    if a.no_run:
        if groups:
            cfg = write_group_config(template, Path(a.output_config), a.run_label, groups[0])
            print((cfg / "pipeline.yaml").read_text(), flush=True)
        return 0

    for k, grp in enumerate(groups):
        date, hours = grp[0][0], [g[1] for g in grp]
        label = "+".join(f"{h:02d}h" for h in hours)
        while True:
            now = datetime.now(timezone.utc)
            busy = nightly_busy(out_root / "_state")
            need = timedelta(minutes=a.run_minutes * len(hours))
            if a.stop_before_nightly:
                ns = nightly_start()
                ok = ns is None or now + need <= ns
            else:
                ok = a.ignore_window or may_start(now, a.stop_utc, a.resume_utc, a.run_minutes * len(hours))
            if ok and not busy:
                break
            why = f"nightly {busy} in progress" if busy else (
                f"would not end before the nightly starts ({nightly_start():%H:%M} UTC)"
                if a.stop_before_nightly else
                f"not enough time before {a.stop_utc} UTC (or inside {a.stop_utc}-{a.resume_utc})")
            if not a.loop:
                print(f"{now:%H:%M} UTC: {why}; stopping, {len(groups) - k} run(s) left in the plan",
                      flush=True)
                return 0
            print(f"{now:%H:%M} UTC: {why}; next: {date} LST {label}; checking again in 10 min", flush=True)
            time.sleep(600)
        cfg = write_group_config(template, Path(a.output_config), a.run_label, grp)
        with open(lock_path, "a+") as lock:
            print(f"{now:%H:%M:%S} {date} LST {label}: waiting for cluster lock {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            t0 = time.time()
            rc = run_gsi(cfg, log_dir / f"backlog_{a.run_label}_{date}.log")
            fcntl.flock(lock, fcntl.LOCK_UN)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {date} LST {label} rc={rc} in {time.time() - t0:.0f} s",
              flush=True)
        if rc == 0:
            state["done"].setdefault(date, []).extend(hours)
        state["runs"].append(dict(date=date, lst=hours, rc=rc, bp=grp[0][2], start=now.isoformat(),
                                  wall_s=time.time() - t0))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))
    return 0


def group_plan(plan, n: int):
    """Consecutive hours of one date with the same bands, at most n per run."""
    groups = []
    for item in plan:
        last = groups[-1] if groups else None
        if (last and len(last) < n and last[0][0] == item[0] and last[0][3] == item[3]):
            last.append(item)
        else:
            groups.append([item])
    return groups


def write_group_config(template: Path, out_dir: Path, run_label: str, grp) -> Path:
    date, bp, bands = grp[0][0], grp[0][2], grp[0][3]
    hours = [g[1] for g in grp]
    ba = None
    if grp[0][4] is not None:
        ba = {f"{g[1]:02d}h/{band}": csv for g in grp for band, csv in g[4].items()}
    note = " ".join(g[5] for g in grp)
    return write_backlog_config(template, out_dir, date, hours, run_label, bp, bands, ba, note)


if __name__ == "__main__":
    sys.exit(main())
