"""Stokes V comparison of peeling variants (images only read).

    python vpol_summary.py STUDY [STUDY ...] > vpol.json
Per snapshot and variant: robust V rms inside the horizon, and in a 4° box at each
target (Cas A, Cyg A, Vir A, RFI emitter): peak |V|, V rms, and peak I (same box).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load, robust_rms, target_pixels, horizon_mask  # noqa: E402

VARS = ["prep", "cpu_sky", "cpu_sky_rfi", "gpu_prod", "gpu_cm_exact", "gpu_cm_exact_joint",
        "fast_sky", "fast_after", "fast_joint"]
out = {}
for study in sys.argv[1:]:
    for snapdir in sorted(glob.glob(f"{study}/*MHz_*")):
        snap = os.path.basename(snapdir)
        key = f"{os.path.basename(study)}/{snap}"
        res = {}
        for v in VARS:
            p = f"{snapdir}/img/{v}"
            if not os.path.exists(p + "-V-image.fits"):
                continue
            V, wcs, t = load(p, "V")
            I, _, _ = load(p, "I")
            hz = horizon_mask(V.shape)
            half = int(2.0 / abs(wcs.wcs.cdelt[0]))
            tg = {}
            for name, (x, y) in target_pixels(wcs, t, V.shape).items():
                x, y = int(round(x)), int(round(y))
                bv = V[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
                bi = I[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
                tg[name] = dict(Vpeak=float(np.nanmax(np.abs(bv))), Vrms=robust_rms(bv), Ipeak=float(np.nanmax(np.abs(bi))))
            res[v] = dict(V_rms=robust_rms(V[hz]), targets=tg)
        out[key] = res
        print(key, file=sys.stderr)
print(json.dumps(out))
