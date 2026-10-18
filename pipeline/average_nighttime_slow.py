"""Flag + frequency-average one night of slow-buffer data into night-time/averaged.

Window: whole LST hours of the given UTC date accepted by the Phase 1
Sun-altitude limits (``orca.utils.sun_cut``: evening Sun <= -18 deg, morning
Sun <= -12 deg), optionally restricted with ``--lst-hours``.  By default an LST
hour is accepted when its midpoint passes the Sun limits
(``--lst-criterion midpoint``), so the first and last hour can contain some
frames with the Sun above the limit; ``--lst-criterion full`` keeps only hours
that lie entirely inside the limits.

Every ``.ms.tar`` (or ``.ms``) integration of all 16 subbands (13-82 MHz) in
that window is run through ``run_pipeline_slow_on_one_cpu_nvme`` (AOFlagger
LWA_opt_GH1, flag metadata, chanbin 4) and stored as
``<out-base>/<freq>/<date>/<UTC hour>/<stamp>_<freq>_averaged.ms.tar`` (plus
``_flagmeta.bin``), the usual ``/lustre/pipeline/night-time/averaged`` layout.
Files already averaged there (``.ms`` or ``.ms.tar``) are skipped, so the
script can be re-run after an interruption.

Tasks are submitted in time order (oldest first, because the slow buffer is
deleted after ~24 h) and pinned to the given per-node Celery queues, weighted
by worker concurrency.

Example (as the pipeline user)::

    python pipeline/average_nighttime_slow.py --date 2026-10-17 --dry-run \\
        --queues calim10:32 calim09:20 calim06:8
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf
from celery import group

from orca.tasks.pipeline_tasks import run_pipeline_slow_on_one_cpu_nvme
from orca.utils.sun_cut import OVRO_LOC, frame_time, sun_ok_mask

iers_conf.auto_download = False
iers_conf.iers_degraded_accuracy = "warn"

FREQS = [
    "13MHz", "18MHz", "23MHz", "27MHz", "32MHz", "36MHz",
    "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz",
    "69MHz", "73MHz", "78MHz", "82MHz",
]
NAME_RE = re.compile(r"^(\d{8}_\d{6})_(\d+MHz)\.ms(\.tar)?$")


def parse_queues(specs):
    """['calim10:32', 'calim06:8'] -> [('calim10', 32), ('calim06', 8)]"""
    out = []
    for s in specs:
        q, _, w = s.partition(":")
        out.append((q, int(w or 1)))
    return out


def assign_queues(n_tasks: int, queues):
    """Weighted interleaved queue assignment (e.g. 20:8 -> 5 of every 7 go to the first)."""
    total = sum(w for _, w in queues)
    credit = {q: 0.0 for q, _ in queues}
    out = []
    for _ in range(n_tasks):
        for q, w in queues:
            credit[q] += w / total
        q = max(credit, key=credit.get)
        credit[q] -= 1.0
        out.append(q)
    return out


def lst_window(date: str, lst_hours=None, step_s: int = 10, criterion: str = "midpoint"):
    """Whole LST hours of *date* (UTC) accepted by the Sun limits.

    *criterion* "midpoint": the Sun passes at the hour's midpoint; "full": at every step.

    Returns [(lst_hour, utc_start, utc_end), ...] with utc_end exclusive.
    """
    t = Time(f"{date}T00:00:00", scale="utc", location=OVRO_LOC) + np.arange(0, 86400 + step_s, step_s) * u.s
    lst = np.floor(t.sidereal_time("apparent").hour).astype(int)
    ok, _, _ = sun_ok_mask(t)
    edges = np.concatenate([[0], np.where(lst[1:] != lst[:-1])[0] + 1, [len(t)]])
    blocks = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i0 == 0 or i1 == len(t):          # LST hour cut by the date boundary
            continue
        if criterion == "full" and not ok[i0:i1].all():
            continue
        if criterion == "midpoint" and not ok[(i0 + i1) // 2]:
            continue
        if lst_hours is not None and lst[i0] not in lst_hours:
            continue
        blocks.append((int(lst[i0]), t[i0].datetime, t[i1].datetime))
    return blocks


def averaged_path(out_base: str, freq: str, date: str, hour: str, name: str) -> str:
    stem = NAME_RE.match(name).group(1) + "_" + freq
    return os.path.join(out_base, freq, date, hour, f"{stem}_averaged.ms")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--lst-hours", nargs="+", type=int,
                   help="only these LST hours (default: every whole LST hour inside the Sun limits)")
    p.add_argument("--lst-criterion", choices=["midpoint", "full"], default="midpoint",
                   help="accept an LST hour if its midpoint passes the Sun limits (default), "
                        "or only if the whole hour does")
    p.add_argument("--queues", nargs="+", required=True, help="queue[:weight], e.g. calim10:32 calim06:8")
    p.add_argument("--root", default="/lustre/pipeline/slow")
    p.add_argument("--out-base", default="/lustre/pipeline/night-time/averaged")
    p.add_argument("--chanbin", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="list and sun-cut only, submit nothing")
    p.add_argument("--stall-min", type=float, default=30,
                   help="stop waiting after this many minutes without a task finishing ...")
    p.add_argument("--max-lost", type=int, default=100, help="... if at most this many are left")
    args = p.parse_args()

    queues = parse_queues(args.queues)

    blocks = lst_window(args.date, set(args.lst_hours) if args.lst_hours else None,
                        criterion=args.lst_criterion)
    if not blocks:
        sys.exit(f"{args.date}: no whole LST hour inside the Sun limits")
    t0, t1 = blocks[0][1], blocks[-1][2]
    print(f"{args.date}: {len(blocks)} whole LST hours accepted by the Sun limits ({args.lst_criterion})")
    for lh, a, b in blocks:
        print(f"  LST {lh:02d}h  UTC {a:%H:%M:%S} - {b:%H:%M:%S}")
    hours = sorted({f"{h:02d}" for h in range(t0.hour, t1.hour + 1)})

    def in_window(f):
        ft = frame_time(f)
        return any(a <= ft < b for _, a, b in blocks)

    todo, n_found, n_done = [], 0, 0
    for freq in FREQS:
        n_freq = 0
        for hr in hours:
            d = Path(args.root) / freq / args.date / hr
            files = sorted(str(f) for f in d.iterdir() if NAME_RE.match(f.name)) if d.is_dir() else []
            for f in files:
                if not in_window(f):
                    continue
                n_found += 1
                out = averaged_path(args.out_base, freq, args.date, hr, os.path.basename(f))
                if os.path.isdir(out) or os.path.isfile(out + ".tar"):
                    n_done += 1
                else:
                    todo.append(f)
                    n_freq += 1
        print(f"  {freq}: {n_freq} to do")

    print(f"{args.date} UTC {t0:%H:%M:%S}-{t1:%H:%M:%S}: {n_found} files in window, "
          f"{n_done} already averaged, {len(todo)} to submit")
    if not todo:
        print("Outputs missing after run: 0")
        return

    # Oldest first, subbands interleaved: the slow buffer is deleted ~24 h after recording.
    todo.sort(key=lambda f: (os.path.basename(f).split("_", 2)[:2], f))
    qs = assign_queues(len(todo), queues)
    for q, _ in queues:
        print(f"  {q}: {qs.count(q)} tasks")
    print(f"  first {os.path.basename(todo[0])}, last {os.path.basename(todo[-1])}")
    if args.dry_run:
        print("Dry run: nothing submitted.")
        return

    sigs = [
        run_pipeline_slow_on_one_cpu_nvme.si(f, start=0, end=23, chanbin=args.chanbin,
                                             output_base=args.out_base.rstrip("/") + "/",
                                             tar_output=True).set(queue=q)
        for f, q in zip(todo, qs)
    ]
    t_start = datetime.now()
    res = group(sigs).apply_async()
    print(f"[{t_start:%F %T}] Submitted group {res.id} ({len(sigs)} tasks)", flush=True)

    # A few tasks can get lost (worker restart, redelivery); don't wait for them forever.
    # Stop when nothing has finished for --stall-min minutes and at most --max-lost are left.
    last_n, last_change = -1, time.time()
    while not res.ready():
        n = res.completed_count()
        if n != last_n:
            last_n, last_change = n, time.time()
        elif len(sigs) - n <= args.max_lost and time.time() - last_change > args.stall_min * 60:
            print(f"[{datetime.now():%F %T}] {len(sigs) - n} tasks unfinished after "
                  f"{args.stall_min} min without progress; giving up on them", flush=True)
            break
        time.sleep(60)
    raw = [r.result if r.ready() else None for r in res.results]
    failed = [(f, r) for f, r in zip(todo, raw) if isinstance(r, Exception)]
    n_ok = sum(1 for r in res.results if r.successful())
    print(f"[{datetime.now():%F %T}] Done in {datetime.now() - t_start}: "
          f"{n_ok}/{len(raw)} OK, {len(failed)} failed", flush=True)
    for f, r in failed:
        print(f"  FAILED {f}: {r!r}")

    missing = [f for f in todo if not os.path.isfile(averaged_path(
        args.out_base, Path(f).parts[-4], args.date, Path(f).parts[-2], os.path.basename(f)) + ".tar")]
    print(f"Outputs missing after run: {len(missing)}")
    sys.exit(1 if failed or missing else 0)


if __name__ == "__main__":
    main()
