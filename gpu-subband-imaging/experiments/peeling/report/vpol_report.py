"""Stokes V comparison over all 96 snapshots -> vpol_section.json + charts/vpol_band.png."""
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
D = {}
for tag, date in (("1220", "2024-12-20"), ("1227", "2024-12-27")):
    D[date] = {k.split("/")[1]: v for k, v in json.load(open(f"{HERE}/vpol/vpol_{tag}.json")).items()}
MODES = [("fast_sky", "cpu_sky", "GPU optimized, sky only"), ("fast_after", "cpu_sky_rfi", "GPU optimized, sky then RFI"),
         ("fast_joint", "cpu_sky_rfi", "GPU optimized, sky + RFI joint"), ("gpu_prod", "cpu_sky", "GPU production (deployed)")]


def ratios(v, ref, tgt=None, key="Vpeak"):
    out = []
    for d in D.values():
        for snap, r in d.items():
            if v in r and ref in r and "prep" in r:
                if tgt is None:
                    out.append(r[v]["V_rms"] / r[ref]["V_rms"])
                else:
                    a, b, p = r[v]["targets"].get(tgt), r[ref]["targets"].get(tgt), r["prep"]["targets"].get(tgt)
                    if a and b and p and p[key] > 2 * b[key]:     # only where CPU peeling removed most of the V there
                        out.append(a[key] / b[key])
    return np.array(out)


rows = [["mode", "snapshots", "V rms / CPU: median", "min–max", "|V| peak / CPU at Cas A", "Cyg A", "Vir A", "RFI emitter",
         "V rms before peeling / CPU after (median)"]]
for v, ref, lab in MODES:
    r = ratios(v, ref)
    t = {k: ratios(v, ref, k) for k in ("CasA", "CygA", "VirA", "RFI")}
    pre = ratios("prep", ref)
    f = lambda a: f"{np.median(a):.3f} (n={len(a)})" if len(a) else "–"
    rows.append([lab, len(r), f"{np.median(r):.3f}", f"{r.min():.3f}–{r.max():.3f}", f(t["CasA"]), f(t["CygA"]), f(t["VirA"]),
                 f(t["RFI"]) if "rfi" in ref else "–", f"{np.median(pre):.2f}"])

def band(s): return int(s.split("MHz")[0])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (date, d) in zip(axes, D.items()):
    for (v, ref, lab), c, m in zip(MODES, (S1, S2, S3, S4), ("s", "o", "D", "x")):
        per = {}
        for snap, r in d.items():
            if v in r and ref in r:
                per.setdefault(band(snap), []).append(r[v]["V_rms"] / r[ref]["V_rms"])
        b = sorted(per)
        for bi in b:
            ax.plot([bi] * len(per[bi]), per[bi], m, color=c, ms=3, alpha=0.35)
        ax.plot(b, [np.median(per[x]) for x in b], "-" + m, color=c, lw=1.5, ms=5, label=lab)
    ax.axhline(1, color=INK2, lw=1); ax.set_title(date, fontsize=9, loc="left"); ax.set_ylim(0.85, 1.6)
    ax.set_xlabel("subband (MHz)"); ax.set_xticks([13, 23, 32, 41, 50, 59, 69, 78])
axes[0].set_ylabel("Stokes V rms / CPU TTCal"); axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Stokes V image rms after peeling relative to CPU TTCal (sky-only vs CPU sky; RFI modes vs CPU sky+RFI)",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{HERE}/charts/vpol_band.png", dpi=160); plt.close(fig)
for r in rows:
    print(r)
json.dump({"table": rows}, open(f"{HERE}/vpol_section.json", "w"), indent=1)
