"""Optimized TTCalX on all 96 snapshots: merge summaries, tables, charts -> fast96.json (read by build_report3.py)."""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
V = {"fast_sky": ("sky only", "cpu_sky", "gpu_cm_exact"),
     "fast_after": ("sky, then RFI (main pipeline's order)", "cpu_sky_rfi", None),
     "fast_joint": ("sky + RFI jointly", "cpu_sky_rfi", "gpu_cm_exact_joint")}
DATES = {"2024-12-20": "full_1220", "2024-12-27": "full_1227"}

summ = {}
for date, tag in DATES.items():
    d = json.load(open(f"{HERE}/{tag}.json"))
    for p in sorted(glob.glob(f"{HERE}/fast96/{tag}/fast_part*.json")):
        for snap, r in json.load(open(p)).items():
            for v in V:
                if v in r:
                    d.setdefault(snap, {})[v] = r[v]
    summ[date] = d
cmp_ = json.load(open(f"{HERE}/fast96/compare_fast.json"))
tim = {}
for date, tag in DATES.items():
    for v in V:
        ms, rows = None, []
        for line in open(f"{HERE}/fast96/{tag}/{v}.log"):
            m = re.match(r"ZESTED (\S+) ([\d.]+)", line)
            if m:
                ms = m.group(1)
            m = re.match(r"ZEST_TIMING read=([\d.]+) write=([\d.]+) peel=([\d.]+)", line)
            if m:
                rows.append(tuple(map(float, m.groups())))
        tim[(date, v)] = rows[1:]          # first snapshot of each daemon includes JIT compilation


def stats(v, ref, key=None):
    out = []
    for date, d in summ.items():
        for snap, r in d.items():
            if v in r and ref in r:
                if key is None:
                    out.append(r[v]["rms"] / r[ref]["rms"])
                else:
                    a, b, p = r[v]["targets"].get(key), r[ref]["targets"].get(key), r["prep"]["targets"].get(key)
                    if a and b and p and p["peak"] > 2 * b["peak"]:
                        out.append(a["peak"] / b["peak"])
    return np.array(out)


rows = [["mode (optimized GPU code)", "snapshots", "median rms / CPU", "min–max rms / CPU", "removed-signal corr. w/ CPU (median)",
         "Cas A resid. / CPU", "Cyg A resid. / CPU", "Vir A resid. / CPU", "RFI resid. / CPU", "diverged",
         "max |Δvis|/rms vs earlier run", "s per snapshot (median)"]]
for v, (lab, ref, old) in V.items():
    r = stats(v, ref)
    corr = [x[v].get(f"corr_{ref}") for d in summ.values() for x in d.values() if v in x and x[v].get(f"corr_{ref}") is not None]
    div = sum(1 for d in summ.values() for x in d.values() if v in x and x[v].get("removed", 0) > 1.05)
    t = [sum(x) for date in DATES for x in tim[(date, v)]]
    cm = [c["max_rel"] for k, c in cmp_.items() if k.endswith("/" + v)]
    tg = lambda k: (f"{np.median(stats(v, ref, k)):.3f}" if len(stats(v, ref, k)) else "–")
    rows.append([lab, len(r), f"{np.median(r):.3f}", f"{r.min():.3f}–{r.max():.3f}",
                 f"{np.median(corr):.4f}" if corr else "–", tg("CasA"), tg("CygA"), tg("VirA"),
                 tg("RFI") if "rfi" in ref else "–", div,
                 (f"{max(cm):.1e} ({len(cm)} snaps)" if cm else "– (no earlier exact run)"), f"{np.median(t):.2f}"])

# per-band chart
def band(s): return int(s.split("MHz")[0])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (date, d) in zip(axes, summ.items()):
    for (v, (lab, ref, _)), c, m in zip(V.items(), (S1, S2, S3), ("s", "o", "D")):
        per = {}
        for snap, r in d.items():
            if v in r and ref in r:
                per.setdefault(band(snap), []).append(r[v]["rms"] / r[ref]["rms"])
        b = sorted(per)
        for bi in b:
            ax.plot([bi] * len(per[bi]), per[bi], m, color=c, ms=3, alpha=0.35)
        ax.plot(b, [np.median(per[x]) for x in b], "-" + m, color=c, lw=1.5, ms=5, label=lab, markeredgecolor=SURF)
    ax.axhline(1, color=INK2, lw=1); ax.set_title(date, fontsize=9, loc="left")
    ax.set_xlabel("subband (MHz)"); ax.set_xticks([13, 23, 32, 41, 50, 59, 69, 78]); ax.set_ylim(0.9, 1.15)
axes[0].set_ylabel("image rms / CPU TTCal"); axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Optimized GPU TTCalX vs CPU TTCal (main pipeline), every snapshot (sky-only vs CPU sky; RFI modes vs CPU sky+RFI)",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{HERE}/charts/fast96_band.png", dpi=160); plt.close(fig)

# timing per snapshot
fig, ax = plt.subplots(figsize=(11, 3.4))
for i, (v, (lab, _, _)) in enumerate(V.items()):
    allt = [x for date in DATES for x in tim[(date, v)]]
    for j, (k, c) in enumerate((("read", S1), ("peel", S2), ("write", S3))):
        vals = [x[j] for x in allt]
        y = i * 4 + j
        ax.plot(vals, np.full(len(vals), y) + np.random.default_rng(0).uniform(-0.25, 0.25, len(vals)), "o", color=c, ms=2.5, alpha=0.5)
        ax.plot([np.median(vals)] * 2, [y - 0.4, y + 0.4], color=INK, lw=2)
    tot = [sum(x) for x in allt]
    ax.text(9.2, i * 4 + 1, f"total median {np.median(tot):.2f} s", va="center", fontsize=8)
ax.set_yticks([i * 4 + j for i in range(3) for j in range(3)],
              [f"{V[v][0][:24]}: {k}" for v in V for k in ("read", "peel", "write")], fontsize=7)
ax.invert_yaxis(); ax.set_xlim(0, 12); ax.set_xlabel("seconds per snapshot (lustre, one daemon per GPU, RTX A4000; bar = median)")
ax.set_title("Optimized GPU TTCalX: per-snapshot times over all 96 snapshots", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{HERE}/charts/fast96_timing.png", dpi=160); plt.close(fig)

nsky = rows[1][1]
bullets = [
    "The optimized code (all six changes, CPU-exact model, CPU-matched solver) was run on every snapshot of both dates in three "
    "modes: 288 runs, 0 failures. Each output was imaged with the GPU pipeline's WSClean settings and summarised like the other "
    "variants. CPU TTCal (main pipeline) is the reference.",
    "<b>Sky:</b> it reproduces the earlier CPU-exact GPU run on all 96 snapshots: max |Δvis| ≤ 3e-5 of the rms, rms difference "
    "~1e-8, flags identical. Image rms and every source residual equal CPU TTCal (ratio 1.000); the removed signal correlates "
    "with CPU's at 0.9999.",
    "<b>Sky then RFI</b> (the main pipeline's order): image rms 1.000× CPU (median), RFI residual 0.93× CPU, no divergence. "
    "It is not bit-for-bit CPU's, by design: ttcal_dev misreads the emitter position. With CPU's (wrong) position forced "
    "(ZEST_RFI_POSITION_ITRF), 23 MHz 03h matches CPU at rms 1.000 and correlation 0.9998.",
    "<b>The one outlier (13 MHz 03h, both dates; rms 1.13× on 12-20)</b> is the daemon's production safety limit "
    "ZEST_MAX_GAIN_AMPLITUDE=100, which CPU TTCal does not have. 13 MHz lies below the RFI model's frequency range "
    "(14.8–40.9 MHz), so the RFI gains go to ~270 and every RFI solve (and one Cyg A solve) is rejected. Rerun without the limit: "
    "rms 0.998× and V rms 1.004× CPU. No other snapshot in sky or sky-then-RFI mode triggered it.",
    "<b>Joint</b> sky+RFI: the same two unstable cases as before (12-20 27 MHz 12h, 12-27 23 MHz 03h); there the result depends on "
    "rounding, so it differs from the earlier run. Use sky-then-RFI, or the guard/prior from section 6.",
    "<b>Speed on lustre</b> (one daemon per GPU, every snapshot except each daemon's first, which includes ~25–30 s of "
    "compilation): median 2.5 s per snapshot for sky only and 3.2 s for sky then RFI, MS read/write included.",
]
json.dump({"bullets": bullets, "table": rows,
           "figs": [["charts/fast96_band.png", "Per subband: image rms relative to CPU TTCal (line = median over hours, dots = snapshots)."],
                    ["charts/fast96_timing.png", "Per-snapshot read / peel / write times of the optimized daemon (all snapshots except each daemon's first)."]]},
          open(f"{HERE}/fast96.json", "w"), indent=1)
for r in rows:
    print(r)
