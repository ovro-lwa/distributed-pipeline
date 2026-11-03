#!/usr/bin/env python
"""Summarise one layout=lst night: products, QA and timing per band and LST hour.

usage: night_report.py OUTPUT_ROOT DATE RUN_LABEL [--json out.json]

Reads <output_root>/<LST>h/<date>/<run_label>/<band>MHz/ (snapshot counts, QA,
hour timing) and the worker logs under <output_root>/_logs/<date>/<run_label>/
("timing ... phase=<step> elapsed=<s>" lines), and prints one row per band-hour
plus per-step medians.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

_T = re.compile(r"^(?P<ts>\S+ \S+) .*timing (?P<band>\d+)MHz batch(?P<batch>\d+) stamp=\S+ "
                r"phase=(?P<phase>[a-z_]+)(?: file=\S+)? elapsed=(?P<s>[0-9.]+)")
_B = re.compile(r"batch (?P<band>\d+)MHz/(?P<batch>\d+): total elapsed (?P<s>[0-9.]+) s for (?P<n>\d+) files")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("date")
    ap.add_argument("run_label")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    root = Path(a.output_root)

    rows = []
    for bdir in sorted(root.glob(f"[0-2][0-9]h/{a.date}/{a.run_label}/*MHz")):
        lst_label, band = bdir.parts[-4], bdir.name
        timing = {}
        tf = bdir / "QA" / f"{band}-hour-timing.json"
        if tf.exists():
            timing = json.loads(tf.read_text())
        n_clean = len(list((bdir / "snapshots_clean").glob("*.fits.fz")))
        n_pilot = len(list((bdir / "snapshots").glob("*.fits.fz")))
        products = {p: len(list((bdir / p).glob("*.fits"))) for p in ("I/deep", "I/10min", "V/deep", "V/10min")}
        rows.append(dict(lst=lst_label, band=band, snapshots_clean=n_clean, snapshots_pilot=n_pilot,
                         qa_rejected=timing.get("qa", {}).get("rejected"),
                         hour_status=timing.get("status", "missing"),
                         hour_image_s=timing.get("timing_s", {}).get("hour_image"),
                         hour_total_s=timing.get("timing_s", {}).get("total"),
                         movie=(bdir / "Movies" / f"{band}_I_Clean.mp4").exists(), products=products))

    steps = defaultdict(list)
    batches = []
    for log in sorted((root / "_logs" / a.date / a.run_label).glob("*MHz/worker-*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = _T.search(line)
            if m:
                steps[m["phase"]].append(float(m["s"]))
            m = _B.search(line)
            if m:
                batches.append(dict(band=int(m["band"]), batch=int(m["batch"]), wall_s=float(m["s"]),
                                    n=int(m["n"]), log=log.name))
    step_summary = {k: dict(n=len(v), median_s=statistics.median(v), total_h=sum(v) / 3600)
                    for k, v in sorted(steps.items())}

    print(f"{'LST':4} {'band':6} {'clean':>5} {'pilot':>5} {'rej':>4} {'hour':8} {'img_s':>6} "
          f"{'tot_s':>6} movie  I/deep I/10min V/deep V/10min")
    for r in rows:
        p = r["products"]
        print(f"{r['lst']:4} {r['band']:6} {r['snapshots_clean']:5d} {r['snapshots_pilot']:5d} "
              f"{str(r['qa_rejected']):>4} {r['hour_status']:8} {r['hour_image_s'] or 0:6.0f} "
              f"{r['hour_total_s'] or 0:6.0f} {'yes' if r['movie'] else 'no ':5}  "
              f"{p['I/deep']:6d} {p['I/10min']:7d} {p['V/deep']:6d} {p['V/10min']:7d}")
    print("\nper-snapshot steps (median s, n, total CPU/GPU-hours):")
    for k, v in step_summary.items():
        print(f"  {k:18} {v['median_s']:7.2f}  n={v['n']:6d}  {v['total_h']:7.2f} h")
    if batches:
        per_snap = [b["wall_s"] / b["n"] for b in batches if b["n"]]
        print(f"\nbatches: {len(batches)}, median wall per snapshot per GPU slot "
              f"{statistics.median(per_snap):.1f} s")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(rows=rows, steps=step_summary, batches=batches), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
