"""Compare what different peeling runs removed from the same input MS.

    python compare_vis.py PREP_MS NAME=MS [NAME=MS ...] [--json OUT]

For every run, removed = DATA(prep) - DATA(run) over unflagged cross
correlations.  Reports, per run, the removed power and residual power, and
for every pair of runs the relative difference of what they removed:
    ||removed_a - removed_b|| / ||removed_a||
overall, per polarization, and per channel, plus the complex correlation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
from casacore.tables import table

POLS = ("XX", "XY", "YX", "YY")


def load(ms: str):
    with table(ms, ack=False) as t:
        data = t.getcol("DATA")
        flag = t.getcol("FLAG")
        a1 = t.getcol("ANTENNA1")
        a2 = t.getcol("ANTENNA2")
    return data, flag, a1 != a2


def rel(a, b):
    d = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prep")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()

    prep, pflag, cross = load(a.prep)
    runs = {}
    flag = pflag.copy()
    for spec in a.runs:
        name, ms = spec.split("=", 1)
        d, f, _ = load(ms)
        runs[name] = d
        flag |= f
    good = (~flag) & cross[:, None, None]
    out = {"n_good": int(good.sum()), "runs": {}, "pairs": {}}

    p0 = np.where(good, prep, 0)
    for name, d in runs.items():
        rem = np.where(good, prep - d, 0)
        res = np.where(good, d, 0)
        out["runs"][name] = {
            "input_power": float(np.sum(np.abs(p0) ** 2)),
            "removed_power": float(np.sum(np.abs(rem) ** 2)),
            "residual_power": float(np.sum(np.abs(res) ** 2)),
            "residual_over_input": float(np.sum(np.abs(res) ** 2) / np.sum(np.abs(p0) ** 2)),
            "removed_power_per_pol": {p: float(np.sum(np.abs(rem[:, :, i]) ** 2))
                                      for i, p in enumerate(POLS)},
        }
    for (na, da), (nb, db) in itertools.combinations(runs.items(), 2):
        ra = np.where(good, prep - da, 0)
        rb = np.where(good, prep - db, 0)
        corr = np.vdot(ra.ravel(), rb.ravel()) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-300)
        out["pairs"][f"{na}|{nb}"] = {
            "rel_diff_removed": rel(ra, rb),
            "rel_diff_removed_per_pol": {p: rel(ra[:, :, i], rb[:, :, i])
                                         for i, p in enumerate(POLS)},
            "rel_diff_removed_per_chan": [rel(ra[:, c, :], rb[:, c, :])
                                          for c in range(ra.shape[1])],
            "corr_removed_abs": float(abs(corr)),
            "rel_diff_residual": rel(np.where(good, da, 0), np.where(good, db, 0)),
        }

    for name, r in out["runs"].items():
        print(f"RUN {name:24s} removed/input={r['removed_power']/r['input_power']:.4f} "
              f"residual/input={r['residual_over_input']:.4f}")
    for k, r in out["pairs"].items():
        ch = np.array(r["rel_diff_removed_per_chan"])
        print(f"PAIR {k:40s} rel_diff_removed={r['rel_diff_removed']:.4f} "
              f"corr={r['corr_removed_abs']:.4f} chan[min/med/max]="
              f"{np.nanmin(ch):.3f}/{np.nanmedian(ch):.3f}/{np.nanmax(ch):.3f} "
              f"pol={ {p: round(v, 3) for p, v in r['rel_diff_removed_per_pol'].items()} }")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
