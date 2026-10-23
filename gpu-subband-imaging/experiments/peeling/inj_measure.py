"""How much of each injected source survives peeling (source suppression).

    python inj_measure.py INJ_STUDY REF_STUDY SNAP INJ_JSON [--png OUT]

For every variant imaged in both studies, D = I(injected run) - I(reference
run, same settings, no injection).  Dirty imaging is linear with fixed weights,
so D is what the peel left of the injected sources.  The recovered fraction
of source j is D(variant) / D(prep) at its position (peak in a 3-pixel box).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inj"); ap.add_argument("ref"); ap.add_argument("snap"); ap.add_argument("json")
    ap.add_argument("--png")
    a = ap.parse_args()
    meta = json.load(open(a.json))
    vs = sorted(os.path.basename(p)[:-len("-I-image.fits")]
                for p in glob.glob(f"{a.inj}/{a.snap}/img/*-I-image.fits"))
    vs = [v for v in vs if os.path.exists(f"{a.ref}/{a.snap}/img/{v}-I-image.fits")]
    if "prep" not in vs:
        sys.exit("need prep in both studies")
    diff = {}
    for v in vs:
        i1, wcs, _ = load(f"{a.inj}/{a.snap}/img/{v}", "I")
        i0, _, _ = load(f"{a.ref}/{a.snap}/img/{v}", "I")
        diff[v] = i1 - i0
    out = {}
    for s in meta["sources"]:
        x, y = wcs.world_to_pixel(SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg))
        x, y = int(round(float(x))), int(round(float(y)))
        val = {v: float(np.nanmax(d[y - 1:y + 2, x - 1:x + 2])) for v, d in diff.items()}
        out[s["name"]] = {v: val[v] / val["prep"] for v in vs}
        out[s["name"]]["_injected_peak"] = val["prep"]
    order = ["prep"] + [v for v in vs if v != "prep"]
    print(f"{a.snap}  recovered fraction of injected flux (1 = untouched by peeling)")
    print("  " + " ".join(f"{n:>9s}" for n in out) + "   variant")
    for v in order:
        print("  " + " ".join(f"{out[n][v]:9.3f}" for n in out) + f"   {v}")
    json.dump(out, open(f"{a.inj}/{a.snap}/recovered.json", "w"), indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [v for v in order if v in ("prep", "cpu_sky", "gpu_cm_after", "gpu_cm_areg0.25", "gpu_cm_areg1",
                                         "gpu_cm_joint", "gpu_cm_jreg0.25", "gpu_cm_jreg1")]
        half = int(1.5 / abs(wcs.wcs.cdelt[0]))
        fig, axes = plt.subplots(len(meta["sources"]), len(sel), figsize=(2.0 * len(sel), 2.1 * len(meta["sources"])),
                                 squeeze=False)
        for i, s in enumerate(meta["sources"]):
            x, y = (int(round(float(c))) for c in wcs.world_to_pixel(SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg)))
            vmax = np.nanmax(diff["prep"][y - 2:y + 3, x - 2:x + 3])
            for j, v in enumerate(sel):
                ax = axes[i, j]
                ax.imshow(diff[v][y - half:y + half, x - half:x + half], origin="lower", cmap="inferno",
                          vmin=-0.3 * vmax, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.04, 0.05, f"{out[s['name']][v]:.2f}", color="w", fontsize=8, transform=ax.transAxes)
                if i == 0:
                    ax.set_title(v.replace("gpu_cm_", ""), fontsize=7.5)
                if j == 0:
                    ax.set_ylabel(f"{s['offset_deg']:g}° from {meta['near']}", fontsize=8)
        fig.suptitle(f"{a.snap}: injected {meta['sources'][0]['flux']:g} Jy sources after peeling "
                     "(injected − not injected; number = recovered fraction)", fontsize=9, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(a.png, dpi=140)


if __name__ == "__main__":
    main()
