#!/usr/bin/env python
"""Daily calibration at fixed LST (replaces the UTC-hour watch_calib_auto.sh).

For each date: a main calibration at LST 22:30 and a backup at LST 21:30 (mean LST
at OVRO-LWA, when Cyg A and Cas A are both high in autumn).  If a target time fails
the main pipeline's Sun cut, the darkest-sky time with the highest min(Cyg A, Cas A)
elevation is used instead (backup: one hour earlier or later, also dark).

Each calibration takes the first --n-files (6, as calibration.sh) 10-s files at or
after the target time from every band in --bands (13 MHz skipped), waits until
they are --stable-seconds old, extracts them to <fast_base>/<date>/<HHMM>/, runs
calibration_master.py from <fast_base> (as calibration.sh), deletes the inputs and
moves the result (<fast_base>/<date>/<LST>h/<successful|unsuccessful>/<stamp>) to
<results>/<date>/<LST>h/<successful|unsuccessful>/<stamp>.  Main and backup run
in parallel.  State: <fast_base>/watch_state_lst/<date>_<role>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from gpu_subband_imaging import lst  # noqa: E402

BANDS = [18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
CAL_PY = "/opt/devel/nkosogor/nkosogor/ovro-lwa-calibration-pipeline/calibration_master.py"
_FINAL = re.compile(r"Moving working directory to final destination: (\S+)")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def log(msg: str) -> None:
    print(f"[{utcnow():%F %T}] {msg}", flush=True)


def _minutes(date: str) -> List[str]:
    day = datetime.strptime(date, "%Y-%m-%d")
    return [(day + timedelta(minutes=m)).strftime("%Y%m%d_%H%M%S") for m in range(24 * 60)]


def plan_times(date: str, main_lst: float, backup_lst: float) -> Dict[str, Tuple[datetime, str]]:
    """{'main'|'backup': (UTC time, how chosen)} for one date."""
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import AltAz, SkyCoord
    names = _minutes(date)
    t = lst._times(names)
    lsth = t.sidereal_time("mean", longitude=lst.OVRO_LON_DEG * u.deg).hour
    ok = np.array(lst.sun_ok(names))
    frame = AltAz(obstime=t, location=lst._location())
    cyg = SkyCoord("19h59m28.36s +40d44m02.1s").transform_to(frame).alt.deg
    cas = SkyCoord("23h23m24s +58d48m54s").transform_to(frame).alt.deg
    score = np.where(ok, np.minimum(cyg, cas), -90)

    def at_lst(h: float) -> Optional[int]:
        d = (lsth - h + 12) % 24 - 12
        idx = [i for i in range(len(d) - 1) if d[i] <= 0 < d[i + 1] or d[i] == 0]
        idx = [i for i in idx if ok[i] and (i + 1 >= len(ok) or ok[i + 1])]
        return idx[0] if idx else None

    out: Dict[str, Tuple[datetime, str]] = {}
    i_main = at_lst(main_lst)
    if i_main is None:
        i_main = int(np.argmax(score))
        out["main"] = (lst.frame_time(names[i_main]), f"fallback: LST {main_lst} not dark; best min(CygA,CasA)={score[i_main]:.0f} deg")
    else:
        out["main"] = (lst.frame_time(names[i_main]), f"LST {main_lst}")
    i_b = at_lst(backup_lst)
    if i_b is None or i_b == i_main:
        cand = [i for i in (i_main - 60, i_main + 60) if 0 <= i < len(ok) and ok[i]]
        i_b = max(cand, key=lambda i: score[i]) if cand else int(np.argsort(score)[-61])
        out["backup"] = (lst.frame_time(names[i_b]), f"fallback: LST {backup_lst} not dark; 1 h from main")
    else:
        out["backup"] = (lst.frame_time(names[i_b]), f"LST {backup_lst}")
    for role, i in (("main", i_main), ("backup", i_b)):
        t0, why = out[role]
        out[role] = (t0, f"{why}; LST {lsth[i]:.2f} h, Cyg A {cyg[i]:.0f} deg, Cas A {cas[i]:.0f} deg")
    return out


def select_files(slow: str, date: str, t0: datetime, bands: List[int], n: int) -> Dict[int, List[Path]]:
    """First n archive files at or after t0 per band (across the UTC hour boundary)."""
    stamp = t0.strftime("%Y%m%d_%H%M%S")
    out = {}
    for b in bands:
        files: List[Path] = []
        for h in (t0.hour, t0.hour + 1):
            d = Path(slow) / f"{b}MHz" / date / f"{h:02d}"
            if d.is_dir():
                files += sorted(p for p in d.iterdir()
                                if p.name.endswith((".ms.tar", ".ms")) and lst.timestamp(p.name) >= stamp)
        out[b] = sorted(files, key=lambda p: p.name)[:n]
    return out


def ready(sel: Dict[int, List[Path]], date_dirs: List[int], n: int, stable_s: int) -> bool:
    for b in date_dirs:
        if len(sel.get(b, [])) < n:
            return False
    newest = max((p.stat().st_mtime for v in sel.values() for p in v), default=None)
    return newest is not None and time.time() - newest >= stable_s


def stage(sel: Dict[int, List[Path]], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for files in sel.values():
        for f in files:
            if f.name.endswith(".ms.tar"):
                with tarfile.open(f) as tf:
                    tf.extractall(dest)
            else:
                shutil.copytree(f, dest / f.name, dirs_exist_ok=True)
            n += 1
    return n


def calibrate(date: str, role: str, t0: datetime, why: str, a) -> dict:
    state_dir = Path(a.fast_base) / "watch_state_lst"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{date}_{role}.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") in ("successful", "unsuccessful"):
        log(f"{date} {role}: already done")
        return json.loads(state_path.read_text())
    rec = dict(date=date, role=role, utc=t0.isoformat(), chosen=why)
    deadline = t0 + timedelta(hours=a.max_wait_hours)
    while True:
        date_dirs = [b for b in a.bands if (Path(a.slow) / f"{b}MHz" / date).is_dir()]
        sel = select_files(a.slow, date, t0, a.bands, a.n_files)
        if date_dirs and ready(sel, date_dirs, a.n_files, a.stable_seconds):
            break
        if utcnow() > deadline and any(sel.values()):
            log(f"{date} {role}: deadline passed, using what is there")
            break
        time.sleep(a.poll_seconds)
    sel = {b: v for b, v in sel.items() if v}
    dest = Path(a.fast_base) / date / t0.strftime("%H%M")
    log(f"{date} {role}: {why}; staging {sum(map(len, sel.values()))} files from "
        f"{len(sel)} bands ({t0:%H:%M:%S} UTC) -> {dest}")
    rec["bands"] = sorted(sel)
    rec["n_files"] = stage(sel, dest)
    logf = Path(a.fast_base) / "logs" / f"calib_lst_{date}_{t0:%H%M}_{role}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", a.cpus, "nice", "-n", str(a.nice), sys.executable, a.cal_py, f"{dest}/"]
    log(f"{date} {role}: running {' '.join(cmd)} (log {logf})")
    t_start = time.time()
    with open(logf, "a") as fh:
        rc = subprocess.run(cmd, cwd=a.fast_base, stdout=fh, stderr=subprocess.STDOUT).returncode
    rec.update(rc=rc, wall_s=time.time() - t_start)
    shutil.rmtree(dest, ignore_errors=True)
    finals = _FINAL.findall(logf.read_text(errors="replace"))
    if finals:
        src = Path(finals[-1])                      # <fast>/<date>/<LST>h/<status>/<stamp>
        rel = src.relative_to(a.fast_base)
        dst = Path(a.results) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.name}.{int(time.time())}")
        shutil.move(str(src), str(dst))
        rec.update(status=rel.parts[2], result=str(dst), lst_label=rel.parts[1])
        for p in (src.parent, src.parent.parent):      # remove now-empty <status>, <LST>h dirs
            try:
                p.rmdir()
            except OSError:
                pass
        tables = sorted(dst.glob("tables/*.B.flagged"))
        rec["bandpass"] = str(tables[0]) if tables else None
    else:
        rec["status"] = "failed-before-finalize"
    state_path.write_text(json.dumps(rec, indent=1))
    log(f"{date} {role}: {rec['status']} in {rec['wall_s'] / 60:.0f} min -> {rec.get('result')}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default=utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--n-days", type=int, default=365)
    ap.add_argument("--main-lst", type=float, default=22.5)
    ap.add_argument("--backup-lst", type=float, default=21.5)
    ap.add_argument("--bands", default=",".join(map(str, BANDS)))
    ap.add_argument("--n-files", type=int, default=6)
    ap.add_argument("--slow", default="/lustre/pipeline/slow")
    ap.add_argument("--fast-base", default="/fast/pipeline/calibration")
    ap.add_argument("--results", default="/lustre/pipeline/calibration/results")
    ap.add_argument("--cal-py", default=CAL_PY)
    ap.add_argument("--stable-seconds", type=int, default=600)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-wait-hours", type=float, default=3.0)
    ap.add_argument("--cpus", default="0-23")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--plan-only", action="store_true", help="print the times for the dates and exit")
    a = ap.parse_args(argv)
    a.bands = [int(b) for b in a.bands.split(",")]
    day0 = datetime.strptime(a.start_date, "%Y-%m-%d")
    for k in range(a.n_days):
        date = (day0 + timedelta(days=k)).strftime("%Y-%m-%d")
        plan = plan_times(date, a.main_lst, a.backup_lst)
        for role, (t0, why) in plan.items():
            log(f"{date} {role}: {t0:%H:%M} UTC - {why}")
        if a.plan_only:
            continue
        threads = [threading.Thread(target=calibrate, args=(date, role, t0, why, a))
                   for role, (t0, why) in plan.items()]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
