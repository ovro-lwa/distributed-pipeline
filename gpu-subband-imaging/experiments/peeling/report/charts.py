"""Summary charts for the report (static PNG, light surface)."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

summary = json.load(open(os.path.join(HERE, "summary.json")))
ablation = json.load(open(os.path.join(HERE, "ablation_summary.json")))
snaps = sorted(summary, key=lambda s: (int(s.split("MHz")[0]), s))


def ratio(src, snap, v, ref):
    r = src.get(snap, {})
    if v in r and ref in r:
        return r[v]["rms"] / r[ref]["rms"]
    return np.nan


def dots(ax, xs, series, ylabel):
    for (label, ys, color, marker) in series:
        ax.plot(xs, ys, linestyle="none", marker=marker, ms=6, color=color, label=label,
                markeredgecolor=SURF, markeredgewidth=0.8)
    ax.axhline(1.0, color=INK2, lw=1)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8, ncol=len(series))


# 1. sky consistency per snapshot
fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(len(snaps))
dots(ax, x, [("GPU production settings", [ratio(summary, s, "gpu_exp", "cpu_sky") for s in snaps], S2, "o"),
             ("GPU CPU-matched", [ratio(summary, s, "gpu_cpumatch", "cpu_sky") for s in snaps], S1, "s")],
     "image rms / CPU (main pipeline)")
ax.set_xticks(x, [s.replace("_", "\n") for s in snaps], fontsize=7)
ax.set_title("Sky peeling: residual image rms relative to the main-pipeline CPU zest (1 = identical)",
             fontsize=10, loc="left", color=INK)
for i, s in enumerate(snaps):
    if summary[s].get("gpu_exp", {}).get("removed", 1) < 1e-6:
        ax.annotate("nothing\npeeled", (i, ratio(summary, s, "gpu_exp", "cpu_sky")), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
ax.set_ylim(0.8, ax.get_ylim()[1] * 1.12)
fig.tight_layout(); fig.savefig(f"{OUT}/sky_rms.png", dpi=160); plt.close(fig)

# 2. RFI per snapshot
low = [s for s in snaps if int(s.split("MHz")[0]) < 50]
fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(len(low))
dots(ax, x, [("GPU sky, then RFI", [ratio(summary, s, "gpu_cm_after", "cpu_sky_rfi") for s in low], S1, "s"),
             ("GPU sky + RFI jointly", [ratio(summary, s, "gpu_cm_joint", "cpu_sky_rfi") for s in low], S3, "D"),
             ("CPU sky only (no RFI pass)", [ratio(summary, s, "cpu_sky", "cpu_sky_rfi") for s in low], S2, "o")],
     "image rms / CPU sky+RFI")
ax.set_xticks(x, [s.replace("_", "\n") for s in low], fontsize=7)
ax.set_title("RFI peeling: residual rms relative to the main pipeline's CPU sky + RFI zest",
             fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/rfi_rms.png", dpi=160); plt.close(fig)

# 3. RFI residual peak per snapshot
fig, ax = plt.subplots(figsize=(11, 3.6))
def pk(s, v):
    t = summary[s].get(v, {}).get("targets", {}).get("RFI")
    return t["peak"] if t else np.nan
for label, v, c, m in (("before peeling", "prep", INK2, "x"), ("CPU sky+RFI", "cpu_sky_rfi", S2, "o"),
                       ("GPU sky, then RFI", "gpu_cm_after", S1, "s"), ("GPU joint", "gpu_cm_joint", S3, "D")):
    ax.plot(x, [pk(s, v) for s in low], linestyle="none", marker=m, ms=6, color=c, label=label,
            markeredgecolor=SURF if m != "x" else c, markeredgewidth=0.8)
ax.set_yscale("log"); ax.set_ylabel("peak |I| near the RFI emitter (Jy/beam)")
ax.set_xticks(x, [s.replace("_", "\n") for s in low], fontsize=7)
ax.legend(loc="upper right", fontsize=8, ncol=4)
ax.set_title("RFI emitter residual (4° box at the emitter's apparent position, inside the horizon)",
             fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/rfi_peak.png", dpi=160); plt.close(fig)

# 4. ablation
abl_snaps = sorted(ablation, key=lambda s: (int(s.split("MHz")[0]), s))
variants = [("gpu_prod", "production"), ("abl_nogate", "no conv. gate"), ("abl_el0", "elev. cut 0°"),
            ("abl_rk4", "rk4 solver, no gate"), ("abl_geom_rk4", "+ geometric UVW"),
            ("abl_nogate_el5", "no gate + elev. 5°"), ("abl_nogate_el0", "no gate + elev. 0°"),
            ("gpu_cpumatch", "all CPU settings")]
fig, ax = plt.subplots(figsize=(11, 3.8))
for j, (v, label) in enumerate(variants):
    ys = [ratio(ablation, s, v, "cpu_sky") for s in abl_snaps]
    ax.plot(np.full(len(ys), j) + np.linspace(-0.2, 0.2, len(ys)), ys, "o", ms=5,
            color=S1 if v in ("abl_nogate_el0", "gpu_cpumatch") else S2, markeredgecolor=SURF)
    ax.plot([j - 0.3, j + 0.3], [np.nanmedian(ys)] * 2, color=INK, lw=2)
ax.axhline(1.0, color=INK2, lw=1)
ax.set_xticks(range(len(variants)), [l for _, l in variants], fontsize=8)
ax.set_ylabel("image rms / CPU")
ax.set_title("Which production setting matters? One change at a time on 8 snapshots (bar = median)",
             fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/ablation.png", dpi=160); plt.close(fig)

# 5. regularization: rms vs max offset
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
offs = ["0.5", "1", "2", "5"]
for ax, key, title in ((axes[0], "rms", "image rms / unregularized"),
                       (axes[1], "CygA", "Cyg A residual peak / unregularized")):
    for s in low:
        r = summary[s]
        base = r.get("gpu_cm_after")
        if not base:
            continue
        ys = []
        for o in offs:
            v = r.get(f"gpu_cm_areg{o}")
            if not v:
                ys.append(np.nan); continue
            if key == "rms":
                ys.append(v["rms"] / base["rms"])
            else:
                a, b = v["targets"].get("CygA"), base["targets"].get("CygA")
                ys.append(a["peak"] / b["peak"] if a and b else np.nan)
        ax.plot([float(o) for o in offs], ys, color=S1, alpha=0.35, lw=1, marker="o", ms=3)
    ax.axhline(1.0, color=INK2, lw=1)
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5], ["0.5°", "1°", "2°", "5°"])
    ax.set_xlabel("prior max offset (taper +1°)"); ax.set_ylabel(title)
axes[0].set_title("Regularized sky pass (then RFI pass), 18 snapshots 18–41 MHz",
                  fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/reg.png", dpi=160); plt.close(fig)

# 6. timing
timing = {}
for line in open(os.path.join(HERE, "timing.txt")):
    m = re.match(r"TIMING (\S+)\s+n=\d+ mean=[\d.]+s median=([\d.]+)s", line)
    if m:
        timing[m.group(1)] = float(m.group(2))
labels = [("gpu_prod", "production (deployed)"), ("gpu_exp", "exp code, production settings"),
          ("gpu_geom", "+ geometric UVW"), ("gpu_cpumatch", "CPU-matched (rk4)"),
          ("gpu_cm_joint", "+ joint RFI"), ("gpu_cm_areg1", "+ RFI pass, reg. sky"),
          ("gpu_cm_jreg1", "+ joint RFI, regularized")]
fig, ax = plt.subplots(figsize=(11, 3.4))
vals = [timing.get(k, np.nan) for k, _ in labels]
bars = ax.barh(range(len(labels)), vals, color=S1, height=0.6)
ax.barh([len(labels)], [80 + 50], color=S2, height=0.6)
for i, v in enumerate(vals + [130]):
    ax.text(v + 1, i, f"{v:.1f} s", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(labels) + 1), [l for _, l in labels] + ["CPU TTCal sky + RFI (main)"])
ax.invert_yaxis(); ax.set_xlabel("median seconds per 48-channel snapshot (RTX A4000; CPU: calim01)")
ax.set_title("Peeling time per MS", fontsize=10, loc="left", color=INK)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/timing.png", dpi=160); plt.close(fig)

# 7. model geometry correlations
fig, ax = plt.subplots(figsize=(11, 3.4))
rows = [("CPU TTCal (julia060)", 0.809, 0.244), ("TTCalX production", 0.003, 0.001),
        ("TTCalX production, conjugated", 0.594, 0.180), ("TTCalX + UVW sign fix", 0.597, 0.160),
        ("TTCalX + geometric UVW (exp)", 0.807, None)]
y = np.arange(len(rows))
ax.barh(y - 0.18, [r[1] for r in rows], height=0.34, color=S1, label="Cas A+Cyg A, 73 MHz 03h")
ax.barh(y + 0.18, [r[2] if r[2] is not None else 0 for r in rows], height=0.34, color=S3,
        label="Cas A, 23 MHz 08h")
ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("correlation of the raw sky model (no gains) with the data, XX")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Is the model on the sky in the right place?", fontsize=10, loc="left", color=INK)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/geometry.png", dpi=160); plt.close(fig)

# 8. Tukey prior
fig, ax = plt.subplots(figsize=(5.5, 3.0))
phi = np.linspace(0, np.pi, 400)
start, width = 0.6, 0.5
P = np.where(phi <= start, 1, np.where(phi >= start + width, 0,
             0.5 * np.cos(np.pi * (phi - start) / width) + 0.5))
ax.plot(np.degrees(phi), P, color=S1, lw=2)
ax.axvline(np.degrees(start), color=INK2, lw=0.8, ls="--")
ax.axvline(np.degrees(start + width), color=INK2, lw=0.8, ls="--")
ax.text(np.degrees(start) / 2, 0.5, "flat:\n2π ν ρ sin(Δθ)/c", ha="center", fontsize=7.5, color=INK)
ax.text(np.degrees(start + width / 2), 1.05, "cosine taper", ha="center", fontsize=7.5, color=INK)
ax.set_xlabel("|gain phase| (deg)"); ax.set_ylabel("prior P(φ)")
ax.set_ylim(-0.05, 1.15)
ax.set_title("Tukey phase prior per antenna/channel", fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/tukey.png", dpi=160); plt.close(fig)
print(sorted(os.listdir(OUT)))
