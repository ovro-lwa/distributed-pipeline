"""Stokes V figures: full sky (before, CPU, GPU, GPU−CPU) and on-source zooms.

    python vpol_figs.py STUDY SNAP OUTDIR CPU_VARIANT GPU_VARIANT
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load, robust_rms, target_pixels, horizon_mask  # noqa: E402

study, snap, outdir, cpu, gpu = sys.argv[1:6]
LAB = {"prep": "before peeling", "cpu_sky": "CPU TTCal, sky", "cpu_sky_rfi": "CPU TTCal, sky then RFI",
       "fast_sky": "GPU TTCalX (optimized), sky", "fast_after": "GPU TTCalX (optimized), sky then RFI",
       "fast_joint": "GPU TTCalX (optimized), joint", "gpu_prod": "GPU production"}
im = {v: load(f"{study}/{snap}/img/{v}", "V") for v in ("prep", cpu, gpu)}
V0, wcs, t = im["prep"]
hz = horizon_mask(V0.shape, 0.495)
rms = {v: robust_rms(im[v][0][horizon_mask(V0.shape)]) for v in im}
vmax = 8 * rms[cpu]
pix = target_pixels(wcs, t, V0.shape)
panels = [("prep", im["prep"][0]), (cpu, im[cpu][0]), (gpu, im[gpu][0]), ("diff", im[gpu][0] - im[cpu][0])]
fig, axes = plt.subplots(1, 4, figsize=(19, 5.3))
for ax, (v, a) in zip(axes, panels):
    ax.imshow(np.where(hz, a, np.nan), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for name, (x, y) in pix.items():
        c = V0.shape[0] / 2
        ax.annotate(name, (x, y), xytext=(x + 0.25 * (c - x), y + 0.25 * (c - y)), color="k", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    title = (f"{LAB.get(gpu, gpu)} − {LAB.get(cpu, cpu)}\nrobust rms {robust_rms(a[horizon_mask(V0.shape)]):.3f}"
             if v == "diff" else f"{LAB.get(v, v)}\nrobust V rms {rms[v]:.3f} Jy/beam")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{os.path.basename(study)} {snap.replace('_', ' ')} UT — Stokes V (dirty), same colour scale ±8× CPU V rms",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(outdir, exist_ok=True)
fig.savefig(f"{outdir}/V_fullsky_{os.path.basename(study)}_{snap}.png", dpi=110)
plt.close(fig)
targets = [k for k in ("CasA", "CygA", "VirA", "RFI") if k in pix]
half = int(4.0 / abs(wcs.wcs.cdelt[0]))
fig, axes = plt.subplots(len(targets), 3, figsize=(8.4, 2.8 * len(targets)), squeeze=False)
for i, k in enumerate(targets):
    x, y = map(int, map(round, pix[k]))
    for j, v in enumerate(("prep", cpu, gpu)):
        cut = im[v][0][max(0, y - half):y + half, max(0, x - half):x + half]
        axes[i, j].imshow(cut, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[i, j].text(2, 2, f"max|V| {np.nanmax(np.abs(cut)):.1f}", fontsize=7)
        if i == 0:
            axes[i, j].set_title(LAB.get(v, v), fontsize=8)
    axes[i, 0].set_ylabel(k)
fig.suptitle(f"{snap.replace('_', ' ')}: Stokes V ±4° around each target", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{outdir}/V_onsource_{os.path.basename(study)}_{snap}.png", dpi=110)
plt.close(fig)
print("VFIG", snap, rms)
