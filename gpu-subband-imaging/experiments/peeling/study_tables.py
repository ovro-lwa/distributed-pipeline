"""Aggregate summary.json from study_summary.py into compact tables (JSON + text).

    python study_tables.py STUDY_DIR

For each variant, over the snapshots where both it and the reference exist:
  rms_ratio      median of rms / rms(reference)
  <tgt>_ratio    median of target peak / peak(reference)   (CasA, CygA, VirA, RFI)
  corr           median correlation of removed signal with the reference's
  noop           number of snapshots where nothing was removed (removed < 1e-6)
References: cpu_sky for the sky-only variants, cpu_sky_rfi for RFI variants.
"""
import json
import os
import sys

import numpy as np

SKY = ["gpu_prod", "gpu_prodfix", "gpu_prodfix_guard", "gpu_exp", "gpu_geom", "gpu_cpumatch", "gpu_cm_exact"]
RFI = ["gpu_cm_after", "gpu_cm_after_guard", "gpu_cm_aftercpu", "gpu_cm_joint", "gpu_cm_joint_guard",
       "gpu_cm_exact_joint", "gpu_prodfix_rfi", "gpu_cm_jreg0.25", "gpu_cm_jreg3", "gpu_geom_rfi", "gpu_joint",
       "gpu_cm_jreg0.5", "gpu_cm_jreg1", "gpu_cm_jreg2", "gpu_cm_jreg5",
       "gpu_cm_areg0.5", "gpu_cm_areg1", "gpu_cm_areg2", "gpu_cm_areg5"]


def table(summary, variants, ref, bands=None):
    rows = {}
    for v in variants:
        acc = {"rms": [], "corr": [], "noop": 0, "n": 0}
        for snap, r in summary.items():
            if v not in r or ref not in r:
                continue
            acc["n"] += 1
            acc["rms"].append(r[v]["rms"] / r[ref]["rms"])
            if r[v].get("removed", 1) < 1e-6:
                acc["noop"] += 1
            if f"corr_{ref}" in r[v]:
                acc["corr"].append(r[v][f"corr_{ref}"])
            for t, m in r[v]["targets"].items():
                rt = r[ref]["targets"].get(t)
                pr = r.get("prep", {}).get("targets", {}).get(t)
                # only targets the peel actually reduced in the reference
                if rt and pr and rt["peak"] < 0.5 * pr["peak"]:
                    acc.setdefault(t, []).append(m["peak"] / rt["peak"])
        if acc["n"]:
            rows[v] = {"n": acc["n"], "noop": acc["noop"],
                       "rms_ratio": float(np.median(acc["rms"])),
                       "corr": float(np.median(acc["corr"])) if acc["corr"] else None,
                       **{f"{t}_ratio": float(np.median(acc[t]))
                          for t in ("CasA", "CygA", "VirA", "RFI") if acc.get(t)}}
    return rows


def main():
    study = sys.argv[1]
    summary = {}
    for st in sys.argv[1:]:  # several studies (dates) are pooled; snapshot keys get the date
        for k, v in json.load(open(f"{st}/summary.json")).items():
            summary[f"{os.path.basename(st)}/{k}"] = v
    band = lambda s: int(s.split("/")[-1].split("MHz")[0])
    sub = lambda lo, hi: {k: v for k, v in summary.items() if lo <= band(k) < hi}
    out = {"sky_all": table(summary, SKY, "cpu_sky"),
           "sky_low": table(sub(0, 50), SKY, "cpu_sky"),
           "sky_high": table(sub(50, 100), SKY, "cpu_sky"),
           "rfi_all": table(summary, RFI, "cpu_sky_rfi"),
           "rfi_low": table(sub(0, 50), RFI, "cpu_sky_rfi"),
           "rfi_high": table(sub(50, 100), RFI, "cpu_sky_rfi")}
    for name, rows in out.items():
        print(f"== {name}")
        for v, r in rows.items():
            print(f"  {v:16s} " + " ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}"
                                           for k, x in r.items()))
    json.dump(out, open(os.environ.get("TABLES_OUT", f"{sys.argv[1]}/tables.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
