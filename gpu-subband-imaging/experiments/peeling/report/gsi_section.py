"""Section 11 (GPU pipeline integration test on 2026-10-23 data): chart + tables -> gsi_section.json."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
W = json.load(open(f"{HERE}/gsi_worker.json"))["timing"]
R = json.load(open(f"{HERE}/gsi_ref.json"))

LAB = {"production": "last night's production run", "prod": "production settings (test run)",
       "prod_branch": "production settings, TTCalX branch", "cpumatch": "CPU-matched + RFI (new)",
       "cpumatch_cpu": "CPU-matched + RFI, TTCalX on CPU"}
fig, ax = plt.subplots(figsize=(11, 3.6))
variants = ["production", "prod", "cpumatch", "cpumatch_cpu"]
colors = [INK2, S2, S1, S3]
for i, band in enumerate((32, 41, 69)):
    for j, (v, c) in enumerate(zip(variants, colors)):
        k = f"{v}/{band}"
        if k not in W:
            continue
        x = W[k]["phases"]["zest"][0]
        y = i * 5 + j
        ax.barh(y, x, color=c, height=0.8, label=LAB[v] if i == 0 or (v == "cpumatch_cpu" and band == 32) else None)
        ax.text(x * 1.08, y, f"{x:.1f} s", va="center", fontsize=7.5, color=INK)
ax.set_xscale("log")
ax.axvline(123, color=S4, lw=1.4)
ax.text(123 * 1.06, 0.5, "CPU TTCal sky+RFI\n(main pipeline): 123 s", color=INK, fontsize=7.5, va="center")
ax.set_xlim(1.2, 900)
ax.set_yticks([i * 5 + 1.5 for i in range(3)], ["32 MHz", "41 MHz", "69 MHz"])
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("worker 'zest' phase, median seconds per snapshot (log scale; MS read + peel + write)")
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.0))
ax.set_title("GPU worker on last night's data (hour 08, 10 snapshots per band)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{HERE}/charts/gsi_zest.png", dpi=160); plt.close(fig)

acc = [["band", "GPU variant", "CPU TTCal reference", "snapshots", "image rms / CPU (median, range)",
        "Cas A resid. / CPU", "Cyg A resid. / CPU", "RFI resid. / CPU", "Stokes V rms / CPU", "removed-signal corr."]]
name = {"gpu_prod": "production settings", "gpu_cpumatch": "CPU-matched (+ RFI at 32/41 MHz)",
        "gpu_cpumatch_rfi": "CPU-matched + RFI"}
refname = {"cpu_sky_rfi": "sky + RFI (main pipeline)", "cpu_sky": "sky only"}
f = lambda x: "–" if x is None else f"{x:.3f}"
for r in R["rows"]:
    acc.append([f"{r['band']} MHz", name[r["variant"]], refname[r["ref"]], r["n"],
                f"{r['rms']:.4f} ({r['rms_min']:.4f}–{r['rms_max']:.4f})", f(r["CasA"]), f(r["CygA"]),
                f(r["RFI"]), f(r["V_rms"]), f"{r['corr']:.4f}"])

perf = [["band", "configuration", "files", "batch wall-clock", "zest phase (median)", "daemon read / peel / write",
         "flag_cal_avg (CPU)", "image"]]
for band in (32, 41, 69):
    for v in ("production", "prod", "prod_branch", "cpumatch", "cpumatch_cpu"):
        k = f"{v}/{band}"
        if k not in W:
            continue
        ph = W[k]["phases"]
        zt = W[k].get("zest_timing")
        b = W[k].get("batch")
        perf.append([f"{band} MHz", LAB[v], b[0][1] if b else ph["zest"][1],
                     f"{b[0][0]:.0f} s" if b else "–", f"{ph['zest'][0]:.1f} s",
                     "/".join(f"{x:.2f}" for x in zt) + " s" if zt else "–",
                     f"{ph.get('flag_cal_avg', [float('nan')])[0]:.1f} s", f"{ph.get('image', [float('nan')])[0]:.1f} s"])
t = R["timing"]
json.dump({"accuracy": acc, "perf": perf,
           "ref_timing": f"Reference set, per snapshot: CPU TTCal {t['cpu']['cpu_sky_rfi:sky']:.0f} s (sky) + "
                         f"{t['cpu']['cpu_sky_rfi:rfi']:.0f} s (RFI); GPU daemon production settings "
                         f"{t['gpu_prod']['read']:.2f} + {t['gpu_prod']['peel']:.2f} + {t['gpu_prod']['write']:.2f} s "
                         f"(read + peel + write), CPU-matched + RFI {t['gpu_cpumatch']['read']:.2f} + "
                         f"{t['gpu_cpumatch']['peel']:.2f} + {t['gpu_cpumatch']['write']:.2f} s."},
          open(f"{HERE}/gsi_section.json", "w"), indent=1)
print("ok")
