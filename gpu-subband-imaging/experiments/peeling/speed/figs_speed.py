"""Images and numbers for the speed-up section: CPU TTCal vs GPU/CPU TTCalX outputs."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imaging as im

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
SNAPS = ["23MHz_0300", "41MHz_1200", "69MHz_1200"]
LAB = {"prep": "before peeling", "cpu_ttcal": "CPU TTCal (main pipeline)",
       "gpu_kernel": "GPU TTCalX, kernel model", "gpu_exact": "GPU TTCalX, exact model (new kernel)",
       "gpu_fast": "GPU TTCalX, optimized", "gpufast": "GPU TTCalX, optimized (in memory)",
       "cpu48": "TTCalX on CPU, 48 threads"}


def robust_rms(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def inside(N, r=0.9):
    yy, xx = np.indices((N, N))
    return np.hypot(xx - N / 2, yy - N / 2) < r * N / 2


def removed_stats(snap, tags):
    P, fp = im.load("prep", snap)
    rem, out = {}, {}
    for t in tags:
        I, f = im.load(t, snap)
        ok = ~(f | fp)
        rem[t] = np.where(ok, P - I, 0)
    ref = rem["cpu_ttcal"]
    for t in tags:
        d = rem[t] - ref
        out[t] = dict(rel_diff_vs_cpu=float(np.linalg.norm(d) / np.linalg.norm(ref)),
                      corr_vs_cpu=float(np.abs(np.vdot(rem[t], ref)) / (np.linalg.norm(rem[t]) * np.linalg.norm(ref))))
    if "gpu_fast" in rem and "cpu48" in rem:
        out["cpu48_vs_gpu_fast"] = float(np.linalg.norm(rem["cpu48"] - rem["gpu_fast"]) / np.linalg.norm(rem["gpu_fast"]))
    if "gpu_fast" in rem and "gpu_exact" in rem:
        out["gpu_fast_vs_gpu_exact"] = float(np.linalg.norm(rem["gpu_fast"] - rem["gpu_exact"]) / np.linalg.norm(rem["gpu_exact"]))
    return out


def main():
    stats = {}
    for snap in SNAPS:
        uvw, freq, t, pd = im.geometry(snap)
        tags = [t for t in ("prep", "cpu_ttcal", "gpu_kernel", "gpu_exact", "gpu_fast", "cpu48")
                if os.path.exists(f"{HERE}/npz/{t}_{snap}.npz") or os.path.exists(f"{HERE}/mem/{t}_{snap}.bin")]
        imgs = {t: im.dirty(*im.load(t, snap), uvw, freq) for t in tags}
        N = imgs["prep"].shape[0]; hz = inside(N)
        rms = {t: robust_rms(imgs[t][hz]) for t in tags}
        srcpix = {}
        for name, (ra, dec) in im.SRC.items():
            l, m, n = im.lm_of(ra, dec, pd)
            if n > 0.05:
                srcpix[name] = im.pix(l, m, N)
        peaks = {t: {s: float(np.abs(imgs[t][int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5]).max())
                     for s, (x, y) in srcpix.items()} for t in tags}
        stats[snap] = dict(rms=rms, peaks=peaks, vis=removed_stats(snap, [t for t in tags if t != "prep"]))
        # full-sky panels
        show = [t for t in ("prep", "cpu_ttcal", "gpu_fast", "cpu48") if t in tags]
        vmax = 12 * min(rms[t] for t in show if t != "prep")
        fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 5.0))
        for ax, t in zip(axes, show):
            ax.imshow(np.where(hz, imgs[t], np.nan),
                      origin="lower", cmap="inferno", vmin=-vmax / 4, vmax=vmax)
            for s, (x, y) in srcpix.items():
                tx, ty = x + 0.25 * (N / 2 - x), y + 0.25 * (N / 2 - y)
                ax.annotate(s, (x, y), xytext=(tx, ty), color="cyan", fontsize=8,
                            arrowprops=dict(arrowstyle="-", color="cyan", lw=0.6))
            ax.set_title(f"{LAB[t]}\nrobust rms {rms[t]:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{snap.replace('_', ' ')} UT, sky then RFI zest: full-sky dirty Stokes I (uniform weighting), same colour scale",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/fullsky_{snap}.png", dpi=120); plt.close(fig)
        # difference images
        pairs = [(a, b) for a, b in (("gpu_fast", "cpu_ttcal"), ("gpu_fast", "gpu_exact"), ("cpu48", "gpu_fast"))
                 if a in tags and b in tags]
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 5.0))
        for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
            dimg = np.where(hz, imgs[a] - imgs[b], np.nan)
            ax.imshow(dimg, origin="lower", cmap="RdBu_r", vmin=-0.2 * rms["cpu_ttcal"], vmax=0.2 * rms["cpu_ttcal"])
            ax.set_title(f"{LAB[a]}\nminus {LAB[b]}\nrms of difference {robust_rms(dimg[hz]):.2e}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{snap.replace('_', ' ')}: difference images, colour scale ±0.2× the CPU image rms ({rms['cpu_ttcal']:.2f})",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/diff_{snap}.png", dpi=120); plt.close(fig)
        # on-source zooms
        rows = list(srcpix)
        show2 = [t for t in ("prep", "cpu_ttcal", "gpu_kernel", "gpu_fast", "cpu48") if t in tags]
        fig, axes = plt.subplots(len(rows), len(show2), figsize=(2.6 * len(show2), 2.6 * len(rows)), squeeze=False)
        h = 60
        for i, s in enumerate(rows):
            x, y = map(int, srcpix[s])
            for j, t in enumerate(show2):
                cut = imgs[t][y - h:y + h, x - h:x + h]
                vm = 10 * rms["cpu_ttcal"]
                axes[i, j].imshow(cut, origin="lower", cmap="inferno", vmin=-vm / 4, vmax=vm)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(LAB[t], fontsize=7.5)
                axes[i, j].text(3, 3, f"peak {peaks[t][s]:.0f}", color="w", fontsize=7)
            axes[i, 0].set_ylabel(s)
        fig.suptitle(f"{snap.replace('_', ' ')}: on-source zooms (±{h * 2 / N * 57.3:.0f}° around each source)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/onsource_{snap}.png", dpi=120); plt.close(fig)
        print(snap, json.dumps(stats[snap]["vis"]), {t: round(v, 3) for t, v in rms.items()})
    json.dump(stats, open(f"{HERE}/speed_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
