"""Charts for the all-subband, two-date study, injection study and benchmark."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})
DATES = {"2024-12-20": json.load(open(os.path.join(HERE, "full_1220.json"))),
         "2024-12-27": json.load(open(os.path.join(HERE, "full_1227.json")))}


def band(s):
    return int(s.split("MHz")[0])


def per_band(summary, v, ref, key="rms"):
    out = {}
    for s, r in summary.items():
        if v in r and ref in r:
            if key == "rms":
                x = r[v]["rms"] / r[ref]["rms"]
            else:
                a, b = r[v]["targets"].get(key), r[ref]["targets"].get(key)
                if not a or not b:
                    continue
                x = a["peak"] / b["peak"]
            out.setdefault(band(s), []).append(x)
    bands = sorted(out)
    return bands, [np.median(out[b]) for b in bands], [out[b] for b in bands]


def band_chart(fname, title, series, ref, ylabel, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (date, summ) in zip(axes, DATES.items()):
        for label, v, c, m in series:
            b, med, allv = per_band(summ, v, ref)
            for bi, vals in zip(b, allv):
                ax.plot([bi] * len(vals), vals, m, color=c, ms=3, alpha=0.35)
            ax.plot(b, med, "-" + m, color=c, label=label, lw=1.6, ms=5, markeredgecolor=SURF)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_title(date, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("subband (MHz)")
        ax.set_xticks([13, 23, 32, 41, 50, 59, 69, 78])
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close(fig)


band_chart("band_sky.png", "Sky peeling vs CPU main pipeline, all 16 subbands (line = median of 03/08/12h, dots = snapshots)",
           [("GPU production", "gpu_prod", S2, "o"), ("GPU production + 3 fixes", "gpu_prodfix", S3, "D"),
            ("GPU CPU-matched + exact model", "gpu_cm_exact", S1, "s")], "cpu_sky", "image rms / CPU", ylim=(0.8, 3.0))
band_chart("band_exact.png", "CPU-matched solver: kernel model without the model-flag fix vs exact model with it",
           [("kernel model, no flag fix", "gpu_cpumatch", S2, "o"), ("exact model + flag fix", "gpu_cm_exact", S1, "s")],
           "cpu_sky", "image rms / CPU", ylim=(0.9, 1.15))
band_chart("band_rfi.png", "Sky + RFI peeling vs CPU main pipeline (sky zest, then RFI zest)",
           [("GPU sky, then RFI", "gpu_cm_after", S1, "s"), ("GPU joint", "gpu_cm_joint", S3, "D"),
            ("GPU joint + guard", "gpu_cm_joint_guard", S2, "o")],
           "cpu_sky_rfi", "image rms / CPU sky+RFI", ylim=(0.8, 1.6))
band_chart("band_reg.png", "Regularized joint zest vs unregularized joint zest (taper = offset)",
           [("prior 0.25°", "gpu_cm_jreg0.25", S2, "o"), ("prior 1°", "gpu_cm_jreg1", S3, "D"),
            ("prior 3°", "gpu_cm_jreg3", S1, "s")], "gpu_cm_joint", "image rms / unregularized", ylim=(0.9, 1.5))

# RFI residual: peak at emitter after / before
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (date, summ) in zip(axes, DATES.items()):
    for label, v, c, m in (("CPU sky only", "cpu_sky", INK2, "x"), ("CPU sky+RFI", "cpu_sky_rfi", S2, "o"),
                           ("GPU sky, then RFI", "gpu_cm_after", S1, "s"), ("GPU joint", "gpu_cm_joint", S3, "D")):
        b, med, _ = per_band(summ, v, "prep", key="RFI")
        ax.plot(b, med, "-" + m, color=c, label=label, lw=1.4, ms=5)
    ax.set_yscale("log"); ax.set_title(date, fontsize=9, loc="left")
    ax.set_xlabel("subband (MHz)"); ax.set_xticks([13, 23, 32, 41, 50, 59, 69, 78])
axes[0].set_ylabel("RFI peak after / before")
axes[0].legend(fontsize=8)
fig.suptitle("Residual at the RFI emitter relative to before peeling (median over hours)", fontsize=10,
             x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/band_rfi_peak.png", dpi=160); plt.close(fig)

# injection
inj = {}
for p in glob.glob(os.path.join(HERE, "inj", "*_recovered.json")):
    inj[os.path.basename(p).split("_recovered")[0]] = json.load(open(p))
if inj:
    snaps = sorted(inj, key=lambda s: (band(s), s))
    fig, axes = plt.subplots(1, len(snaps), figsize=(2.3 * len(snaps), 3.2), sharey=True)
    offs = [1, 2, 4, 8]
    for ax, s in zip(np.atleast_1d(axes), snaps):
        r = inj[s]
        for label, v, c, m in (("CPU", "cpu_sky", INK2, "x"), ("GPU", "gpu_cm_after", S1, "s"),
                               ("GPU + prior 0.1°", "gpu_cm_areg0.1", S2, "o"), ("GPU + prior 1°", "gpu_cm_areg1", S3, "D")):
            ys = [r.get(f"inj{o}", {}).get(v, np.nan) for o in offs]
            ax.plot(offs, ys, "-" + m, color=c, label=label, ms=4, lw=1.2)
        ax.axhline(1, color=INK2, lw=1); ax.set_xscale("log"); ax.set_xticks(offs, [f"{o}°" for o in offs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(s.replace("_", " "), fontsize=8.5); ax.set_ylim(0, 1.15)
        ax.set_xlabel("offset from Cas A")
    np.atleast_1d(axes)[0].set_ylabel("recovered fraction")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Injected 300 Jy sources near Cas A: fraction surviving peeling (prior taper 1°)",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f"{OUT}/injection.png", dpi=160); plt.close(fig)

# benchmark
bp = os.path.join(HERE, "bench.txt")
if os.path.exists(bp):
    cpu, gpu, walls = [], {}, {}
    for line in open(bp):
        m = re.match(r"BENCH cpu (\S+) sky=([\d.]+) rfi=([\d.]+) total=([\d.]+)", line)
        if m:
            cpu.append((float(m.group(2)), float(m.group(3))))
        m = re.match(r"BENCH gpu (\S+) (\S+)\.ms ([\d.]+)", line)
        if m:
            gpu.setdefault(m.group(1), []).append(float(m.group(3)))
        m = re.match(r"BENCH gpu (\S+) wall_total=([\d.]+)", line)
        if m:
            walls[m.group(1)] = float(m.group(2))
    labels = [("CPU TTCal sky + RFI (main pipeline, per-MS process)", np.median([a + b for a, b in cpu]), S2)]
    names = {"prod_sky": "GPU production (sky only)", "prodfix_rfi_joint": "GPU production + fixes, joint RFI",
             "cm_after": "GPU CPU-matched, sky then RFI", "cm_joint": "GPU CPU-matched, joint RFI",
             "cm_exact_joint": "GPU CPU-matched, joint RFI, exact model"}
    for k, lab in names.items():
        if k in gpu:
            labels.append((lab, np.median(gpu[k]), S1))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, (lab, v, c) in enumerate(labels):
        ax.barh(i, v, color=c, height=0.6)
        ax.text(v * 1.05, i, f"{v:.1f} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), [l for l, _, _ in labels]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("median seconds per snapshot (log scale; calim01, RTX A4000)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Peeling time per 48-channel snapshot, same 8 snapshots (13–82 MHz)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/bench.png", dpi=160); plt.close(fig)
    json.dump({"cpu": cpu, "gpu": gpu, "walls": walls}, open(os.path.join(HERE, "bench.json"), "w"), indent=1)
print(sorted(os.listdir(OUT)))
