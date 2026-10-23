"""Speed charts and table for v3 (numbers from bench.txt, bench2/fasttest/writetest/e2e logs, mem_zest runs)."""
import json
import os

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
med = lambda x: float(np.median(x))

# per-snapshot medians, warm-up (first) snapshot excluded
bench_cpu = 130.0                                                         # bench.txt: sky 78 + RFI 52
bench_kernel = med([11.47, 13.33, 10.79, 10.58, 10.56, 10.54, 10.86])     # cm_after (lustre)
bench_exact_old = med([24.57, 23.04, 17.48, 25.8, 22.57, 21.95, 20.2])    # cm_exact_joint (lustre)
b2 = dict(read=med([3.93, 1.67, 3.00, 2.57, 1.48, 2.03, 2.30]), peel=med([5.61, 5.60, 5.33, 5.08, 5.06, 5.08, 5.09]),
          write=med([2.75, 6.29, 6.14, 7.46, 4.18, 3.58, 5.74]))           # bench2 gpu_exact_after (lustre)
wt_slow = dict(read=med([1.44, 1.41]), peel=med([1.90, 1.87]), write=med([1.87, 1.63]))   # /tmp, original I/O
e2e_gpu = dict(read=med([0.40, 0.42]), peel=med([1.87, 1.71]), write=med([0.22, 0.25]))  # /tmp, fast I/O
e2e_cpu = dict(read=med([0.43, 0.40]), peel=med([6.60, 6.54]), write=med([0.21, 0.28]))  # /tmp, 24 threads
rows = [("GPU, CPU-exact model, earlier code (lustre)", None, bench_exact_old),
        ("GPU, + exact-model kernel (lustre)", b2, None),
        ("GPU, + GPU flags + coalesced step, original MS I/O (/tmp)", wt_slow, None),
        ("GPU, + numpy MS read/write (/tmp)", e2e_gpu, None),
        ("TTCalX on CPU, 24 threads, numpy MS I/O (/tmp)", e2e_cpu, None)]

fig, ax = plt.subplots(figsize=(11, 3.6))
for i, (lab, d, tot) in enumerate(rows):
    if d is None:
        ax.barh(i, tot, color=INK2, height=0.6, alpha=0.45)
        ax.text(tot + 0.3, i, f"{tot:.1f} s (not split)", va="center", fontsize=8, color=INK)
        continue
    left = 0
    for k, c in (("read", S1), ("peel", S2), ("write", S3)):
        ax.barh(i, d[k], left=left, color=c, height=0.6, label=k if i == 1 else None, edgecolor=SURF, linewidth=1.5)
        left += d[k]
    ax.text(left + 0.3, i, f"{left:.1f} s", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("median seconds per snapshot (sky then RFI zest, CPU-matched settings, exact model; first snapshot excluded)")
ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=8, ncol=3)
ax.set_title(f"Where the time goes, step by step (CPU TTCal main pipeline: {bench_cpu:.0f} s; GPU kernel model: {bench_kernel:.1f} s)",
             fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{HERE}/charts/speed_breakdown.png", dpi=160); plt.close(fig)

th = [1, 4, 12, 24, 48]
p41 = [19.63, 7.14, 6.54, 6.51, 7.63]; p69 = [None, None, 5.58, 6.51, 7.18]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(th, p41, "-o", color=S1, label="41 MHz 12h", lw=1.6, ms=5)
ax.plot([t for t, p in zip(th, p69) if p], [p for p in p69 if p], "-s", color=S3, label="69 MHz 12h", lw=1.6, ms=5)
ax.axhline(1.8, color=S2, lw=1.4); ax.text(30, 2.2, "GPU (RTX A4000): 1.8 s", color=INK, fontsize=8)
ax.set_xscale("log", base=2); ax.set_xticks(th, [str(t) for t in th]); ax.set_ylim(0, 21)
ax.set_xlabel("Julia threads (calim03: 24 cores / 48 hardware threads)"); ax.set_ylabel("peel time per snapshot (s)")
ax.legend(fontsize=8); ax.set_title("TTCalX on the CPU (ZEST_DEVICE=cpu): sky+RFI peel time vs threads", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{HERE}/charts/speed_threads.png", dpi=160); plt.close(fig)

st = json.load(open(os.path.join(HERE, "..", "speed", "speed_stats.json")))
table = [
    ["CPU TTCal, main pipeline (sky zest + RFI zest, new Julia process each; lustre)", "–", "–", "–", f"{bench_cpu:.0f} s", "reference (RFI pass differs, see below)"],
    ["GPU TTCalX, kernel model (lustre)", "–", "–", "–", f"{bench_kernel:.1f} s", "model differs by ~2–4% on long baselines"],
    ["GPU TTCalX, CPU-exact model, earlier code (lustre, joint RFI)", "–", "~10.7 s (est.)", "–", f"{bench_exact_old:.1f} s", "reference for the rows below"],
    ["  + exact model as GPU kernel (lustre)", f"{b2['read']:.1f} s", f"{b2['peel']:.1f} s", f"{b2['write']:.1f} s", f"{sum(b2.values()):.1f} s", "≤2e-9 (model), images identical"],
    ["  + GPU short-baseline flags + coalesced step (/tmp)", f"{wt_slow['read']:.1f} s", f"{wt_slow['peel']:.1f} s", f"{wt_slow['write']:.1f} s", f"{sum(wt_slow.values()):.1f} s", "identical at complex64"],
    ["  + numpy MS read and write (/tmp)", f"{e2e_gpu['read']:.2f} s", f"{e2e_gpu['peel']:.1f} s", f"{e2e_gpu['write']:.2f} s", f"{sum(e2e_gpu.values()):.1f} s", "byte-identical DATA and FLAG"],
    ["TTCalX on CPU, 24 threads, numpy MS I/O (/tmp)", f"{e2e_cpu['read']:.2f} s", f"{e2e_cpu['peel']:.1f} s", f"{e2e_cpu['write']:.2f} s", f"{sum(e2e_cpu.values()):.1f} s", "≤3e-8 of rms (last complex64 bit)"],
    ["TTCalX on CPU, 1 thread (in memory)", "", "19.6 s", "", "", "same"],
]
eq = []
for snap, d in st.items():
    v = d["vis"]
    eq.append([snap.replace("_", " "), f"{v['gpu_fast']['rel_diff_vs_cpu']:.3f} / {v['gpu_fast']['corr_vs_cpu']:.4f}",
               f"{d['rms']['cpu_ttcal']:.2f}", f"{d['rms']['gpu_fast']:.2f}", f"{d['rms'].get('cpu48', float('nan')):.2f}",
               f"{v.get('gpu_fast_vs_gpu_exact', float('nan')):.1e}", f"{v.get('cpu48_vs_gpu_fast', float('nan')):.1e}"])
json.dump({"table": table, "equality": eq}, open(f"{HERE}/speed_timing.json", "w"), indent=1)
print("ok")
