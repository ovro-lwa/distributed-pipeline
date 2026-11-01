#!/usr/bin/env python
"""Hour products (I/V deep and 10 min) of several runs of one band-hour, pairwise (e2e_compare_v2.py
measures): noise (MAD sigma in the 0.40 horizon mask), diff rms / max in sigma, check-source peaks,
peeled-source residuals, Jupiter.

    hour_compare.py --band 23 --root M1=<band dir> --root M2=... --root A=... --root B=... \
        --pairs M1:M2,A:B,A:M1,B:M2 --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

E2E = "/lustre/nkosogor/peel_exp/e2e"
DEEP = ("I-Deep-Taper-Robust-0", "I-Deep-Taper-Robust-0.75", "I-Deep-NoTaper-Robust-0",
        "I-Deep-NoTaper-Robust-0.75", "V-Taper-Deep")


def files(root, band):
    out = {}
    for p in DEEP:
        f = sorted(glob.glob(f"{root}/{p[0]}/deep/{band}MHz-{p}-image-*.fits"))
        if f:
            out[p] = f[0]
    for pol in "IV":
        for k, f in enumerate(sorted(glob.glob(f"{root}/{pol}/10min/{band}MHz-{pol}-Taper-10min-*-image.fits"))):
            out[f"{pol}-Taper-10min-{k}"] = f
    return out


def main(argv=None) -> int:
    sys.path.insert(0, E2E)
    import e2e_compare_v2 as ec
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--root", action="append", required=True, help="label=band dir")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    roots = dict(r.split("=", 1) for r in a.root)
    fl = {k: files(v, a.band) for k, v in roots.items()}
    pairs = [p.split(":") for p in a.pairs.split(",")]
    res = {}
    tmp = tempfile.mkdtemp(prefix="nk_hcmp_", dir="/tmp")
    for prod in sorted(set.intersection(*(set(f) for f in fl.values()))):
        im = {k: ec.read_image(fl[k][prod], tmp) for k in roots}
        data = {k: v[0] for k, v in im.items()}
        k0 = next(iter(roots))
        mask = ec.horizon_mask(data[k0].shape)
        beam_px = 2 if data[k0].shape[0] <= 2048 else 3
        r = {f"{x}:{y}": ec.compare_pair(data[x], data[y], mask) for x, y in pairs}
        src = {k: ec.src_table(data[k], im[k][1], ec.mad_std(data[k][mask]), beam_px) for k in roots}
        r["src"] = src
        rat = {}
        for x, y in pairs:
            v = [src[y][n] / src[x][n] for n in ec.CHECK if np.isfinite(src[x][n]) and src[x][n] / src[x]["sigma"] > 10]
            rat[f"{x}:{y}"] = dict(n=len(v), min=min(v) if v else None, max=max(v) if v else None)
        r["src_ratio_snr>10"] = rat
        res[prod] = r
        print(prod, json.dumps({p: {k: round(v, 5) for k, v in r[p].items()} for p in r if ":" in p}),
              json.dumps(rat), flush=True)
    os.system(f"rm -rf {tmp}")
    summ = {}
    for x, y in pairs:
        p = f"{x}:{y}"
        for grp, sel in (("deep", lambda k: k in DEEP), ("10min", lambda k: "10min" in k)):
            ks = [k for k in res if sel(k)]
            if not ks:
                continue
            summ[f"{p} {grp}"] = dict(
                n=len(ks), diff_rms_sigma_max=max(res[k][p]["diff_rms_sigma"] for k in ks),
                diff_max_sigma_max=max(res[k][p]["diff_max_sigma"] for k in ks),
                noise_ratio_dev_max=max(abs(res[k][p]["noise_ratio"] - 1) for k in ks),
                src_ratio_min=min((res[k]["src_ratio_snr>10"][p]["min"] for k in ks
                                   if res[k]["src_ratio_snr>10"][p]["min"] is not None), default=None),
                src_ratio_max=max((res[k]["src_ratio_snr>10"][p]["max"] for k in ks
                                   if res[k]["src_ratio_snr>10"][p]["max"] is not None), default=None))
    print(json.dumps(summ, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(summary=summ, products=res, roots=roots), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
