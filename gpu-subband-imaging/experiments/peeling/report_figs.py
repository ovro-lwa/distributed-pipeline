"""Report figures: selected variants side by side, full sky + zooms on targets.

    python report_figs.py STUDY_DIR OUT_DIR SNAP VARIANT[,VARIANT...] [--zoom RFI,CasA]

All panels of one snapshot share one colour scale (from the best rms among
the variants shown), so residuals compare directly.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load, robust_rms, target_pixels  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {"prep": "before peeling", "cpu_sky": "CPU sky (main)", "cpu_sky_rfi": "CPU sky+RFI (main)",
         "gpu_prod": "GPU production", "gpu_cpumatch": "GPU CPU-matched",
         "gpu_cm_after": "GPU sky+RFI (after)", "gpu_cm_joint": "GPU sky+RFI (joint)",
         "gpu_cm_aftercpu": "GPU sky+RFI, CPU RFI pos.", "gpu_cm_areg1": "GPU after + reg 1°",
         "gpu_cm_areg5": "GPU after + reg 5°", "gpu_cm_jreg1": "GPU joint + reg 1°",
         "gpu_cm_jreg0.25": "GPU joint + reg 0.25°", "gpu_cm_jreg3": "GPU joint + reg 3°",
         "gpu_prodfix": "GPU prod. + 3 fixes", "gpu_prodfix_rfi": "GPU prod. + fixes, joint RFI",
         "gpu_cm_exact": "GPU CPU-matched, exact model", "gpu_cm_exact_joint": "GPU exact model, joint RFI",
         "gpu_prodfix_guard": "GPU prod. + fixes + guard", "gpu_cm_joint_guard": "GPU joint + guard",
         "gpu_cm_after_guard": "GPU after + guard"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study"); ap.add_argument("out"); ap.add_argument("snap"); ap.add_argument("variants")
    ap.add_argument("--zoom", default="RFI,CasA")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",")
                if os.path.exists(f"{a.study}/{a.snap}/img/{v}-I-image.fits")]
    imgs = {v: load(f"{a.study}/{a.snap}/img/{v}", "I") for v in variants}
    img0, wcs, t = imgs[variants[0]]
    shape = img0.shape
    yy, xx = np.indices(shape)
    r = np.hypot(xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2)
    horizon = r < 0.495 * shape[0]
    rms = {v: robust_rms(im[r < 0.47 * shape[0]]) for v, (im, _, _) in imgs.items()}
    vmax = 10 * min(rms.values())
    pix = target_pixels(wcs, t, shape)
    zooms = [z for z in a.zoom.split(",") if z in pix]
    half = int(10.0 / abs(wcs.wcs.cdelt[0]) / 2)
    ncol = len(variants)
    nrow = 1 + len(zooms)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.75 * nrow), squeeze=False)
    for j, v in enumerate(variants):
        im = np.where(horizon, imgs[v][0], np.nan)
        ax = axes[0, j]
        ax.imshow(im, origin="lower", cmap="inferno", vmin=-vmax / 4, vmax=vmax)
        ax.set_title(f"{LABEL.get(v, v)}\nrms {rms[v]:.2f} Jy/b", fontsize=7.5)
        for i, z in enumerate(zooms):
            x, y = (int(round(c)) for c in pix[z])
            sl = (slice(max(0, y - half), y + half), slice(max(0, x - half), x + half))
            ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False,
                                       ec="cyan", lw=0.5))
            zi = im[sl]
            zx = axes[1 + i, j]
            zx.imshow(zi, origin="lower", cmap="inferno", vmin=-vmax / 2, vmax=2 * vmax)
            zx.set_title(f"{z} (20°) max |I| {np.nanmax(np.abs(zi)):.0f}", fontsize=7)
        for axx in axes[:, j]:
            axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle(f"{a.snap.replace('_', ' ')}  dirty Stokes I  {t.isot[:19]} UTC", fontsize=9)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/{a.snap}.png", dpi=150)


if __name__ == "__main__":
    main()
