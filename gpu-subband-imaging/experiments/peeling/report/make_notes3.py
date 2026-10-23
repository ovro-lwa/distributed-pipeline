"""v3 notes: v2 + speed-up of the CPU-exact model, TTCalX on CPU, math/code map."""
import json

N = json.load(open("notes2.json"))
N["title"] = ("GPU vs CPU peeling, GPU RFI peeling and Regularized Peeling — OVRO-LWA "
              "(v3: two dates, all subbands, speed-ups, TTCalX on CPU, math)")
N["subtitle"] = N["subtitle"].replace("Generated 2026-10-23.", "Generated 2026-10-23 (v3).")
N["answers"][3] = (
    "<b>Speed:</b> CPU TTCal sky+RFI takes ~130 s per snapshot (sky 78 s + RFI 52 s). The GPU daemon took ~11 s with the kernel "
    "model and ~22 s with the CPU-exact model. The exact model was slow only because it was evaluated in a single-threaded, "
    "type-unstable host loop (~5.5 s per snapshot). It now runs as a GPU kernel (0.01 s, identical to 1e-9). With the other "
    "speed-ups (short-baseline flags on the GPU, coalesced StefCal step, numpy MS read and write-back), a whole CPU-exact sky+RFI "
    "snapshot takes <b>2.4 s</b> on the GPU (read 0.4 + peel 1.8 + write 0.2 s) instead of ~22 s. That is ~50× faster than CPU "
    "TTCal's ~130 s, and the DATA and FLAG written are byte-identical to the unoptimized run.")
N["answers"].insert(4,
    "<b>TTCalX on the CPU</b> (same code on host arrays, one channel per thread) gives the same output as the GPU to ≤3e-8 of the "
    "rms (the last bit of the stored complex64). A snapshot takes 7.2 s end-to-end with 24 threads (peel 6.5 s); the peel takes "
    "19.6 s with a single thread. Compare 2.4 s on the GPU and ~130 s for CPU TTCal.")
N["speed2_intro"] = [
    "<b>Why the CPU-exact model was ~2× slower.</b> genvis_exact! (the model that makes GPU = CPU TTCal to 1.000) was written as a "
    "host loop over every component, channel and baseline. The components come from a Vector{GPUSource} (abstract type), so "
    "flux, shape and every inner-loop operation were dynamically dispatched and boxed: ~4 s for Cas A and ~1.3 s for Vir A per "
    "snapshot, single-threaded, then a host→GPU copy. That is 5–6 s of the 11 s difference. The rest was the daemon "
    "regenerating the model for each pass and the extra JIT compilation in the first snapshot.",
    "<b>Fix 1 — exact model as a GPU kernel</b> (experimental.jl _exact_tables / _exact_vis / genvis_exact_kernel!). The "
    "per-component tables (N<sub>ant</sub> delays, N<sub>ν</sub> fluxes, Gaussian axes) are built on the host, then one thread per "
    "(baseline, channel) evaluates the same expression. 3.8 s → 0.008 s for Cas A, 1.3 s → 0.002 s for Vir A (~500×); on host "
    "threads 0.02 s. Max relative difference from the reference loop: 0 for points, ≤2e-9 for Gaussians (operation order).",
    "<b>Fix 2 — short-baseline flags on the GPU</b> (flag_short_baselines_gpu!). TTCalX's flag_short_baselines! copies the four "
    "N<sub>ant</sub>×N<sub>ant</sub>×N<sub>ν</sub> square matrices to the host, loops, and copies them back: 85–90 ms, twice per solve, "
    "~40% of a zest solve. A broadcast mask on the GPU takes 0.2 ms. The result is bit-identical.",
    "<b>Fix 3 — coalesced StefCal step</b> (stefcal_step_full_herm_kernel!). Thread j summed M[i,j] over i, so neighbouring "
    "threads read addresses N<sub>ant</sub> apart. The square matrices are Hermitian, so M[i,j] = M[j,i]<super>H</super>: reading "
    "the row instead makes the loads coalesced. 10.4 → 8.5 ms per step; same result to 2e-16. Solve: 0.48 s → 0.15 s.",
    "<b>Fix 4 — MS read and write in numpy</b> (read_ms_fast, write_ms_fast!). read_ms_to_gpu / write_gpu_to_ms! convert whole "
    "columns between numpy and Julia through PyCall and gather/scatter rows in Julia. The write also re-reads DATA and FLAG and "
    "goes back through complex128. The new paths gather/scatter in numpy on the arrays casacore returns. Read: 1.3 → 0.4 s, with "
    "all returned arrays, metadata and baseline map identical. Write: 1.6–1.9 → 0.23 s on /tmp (2–9 s on lustre), with DATA and "
    "FLAG byte-identical.",
    "<b>TTCalX on the CPU</b> (ZEST_DEVICE=cpu). TTCalX's peel operations launch CUDA kernels unconditionally, and its existing "
    "cpu_* fallbacks are single-threaded (and cpu_makesquare! differs from the GPU kernel: it keeps autos and accumulates). The new "
    "host methods run the GPU kernels' arithmetic per channel on Julia threads: gpu_corrupt!, gpu_subsrc!/putsrc!, host_makesquare! "
    "(as makesquare_kernel!), _host_stefcal_steps! (cpu_stefcal_step_full! per channel) and genvis_exact! on host threads. "
    "No phase prior on the host yet.",
]
N["speed2_notes"] = [
    "Peel times are in-memory runs of the daemon's own safe_zest_gpu! (sky pass then RFI pass, CPU-matched settings, exact "
    "model) on calim03, after a warm-up snapshot. The MS read/write times are from daemon runs on lustre; they vary a lot with "
    "lustre load (1.3–4.4 s read, 1.9–8.8 s write).",
    "Per-node throughput is a different question from per-snapshot time. CPU TTCal is single-threaded, so a 48-core node can run "
    "many snapshots at once, if memory allows. TTCalX on the CPU uses all cores for one snapshot. The 24-thread run is faster than 48: "
    "the node has 24 physical cores.",
    "The production GPU daemon (kernel model, damped solver) is unaffected; all of this is opt-in in the experimental branch "
    "(ZEST_GENVIS=exact, ZEST_DEVICE=cpu, ZEST_FAST_READ=1, ZEST_FAST_WRITE=1). The first snapshot of each daemon pays ~25–30 s "
    "of JIT compilation.",
    "Storage: the user's lustre file-count quota (2 M files) filled up during the benchmark. The remaining runs therefore used MS "
    "copies on calim03's local /tmp and in-memory runs, and nothing on lustre was deleted. Rows marked lustre and /tmp are not "
    "directly comparable for I/O: lustre read/write is slower and much more variable.",
]
N["speed2_figs"] = [
    ["charts/speed_breakdown.png", "Per-snapshot time split into MS read, peel (sky+RFI) and MS write."],
    ["charts/speed_threads.png", "TTCalX on the CPU: peel time vs threads (41 and 69 MHz, 12h)."],
    ["speed/fullsky_41MHz_1200.png", "41 MHz 12h: before peeling, CPU TTCal, optimized GPU TTCalX, TTCalX on CPU (sky then RFI)."],
    ["speed/diff_41MHz_1200.png", "41 MHz 12h difference images (±0.2× CPU rms). GPU−CPU TTCal differs mostly through the RFI pass "
     "(different emitter position); the optimized GPU, the earlier exact-model run and the CPU TTCalX are identical."],
    ["speed/onsource_23MHz_0300.png", "23 MHz 03h on-source zooms (peak |I| in each box)."],
    ["speed/fullsky_69MHz_1200.png", "69 MHz 12h full sky."],
    ["speed/onsource_69MHz_1200.png", "69 MHz 12h on-source zooms."],
]
N["math"] = [
    ["Measurement equation and peeling/zesting", ["measurement", "zest"],
     "Each source d gets its own full-Jones gains J<sub>p,d</sub>(ν) per antenna and channel (zest = full Jones, per channel). "
     "Peeling adds the source back with its current gains, re-solves, and subtracts again, peeliter = 3 times.",
     "CPU TTCal: peel.jl:34 peel!. GPU: zest_daemon.jl:137 safe_zest_gpu!; TTCalX peel_gpu.jl:842 gpu_corrupt! (J M J<super>H</super>), "
     "kernels/genvis.jl subsrc/putsrc."],
    ["Far-field model (CPU-exact)", ["farfield", "stokes", "gauss"],
     "Apparent ITRF directions via R(t) (precession, nutation, GAST). Each component is skipped below the horizon. "
     "Gaussian axes come from the component's local north/east, with the 3-D baseline projected on them. TTCalX's kernels "
     "used (u,v), a PA from east and no foreshortening.",
     "experimental.jl:759 _exact_tables, :790 _exact_vis, :812 genvis_exact_kernel!, :832 genvis_exact!, :684 _gaussian_axes, "
     ":713 genvis_exact_reference!. CPU TTCal: instrument/genvis.jl:107 genvis_onesource_onechannel!, :263 delays_to_fringes, "
     ":290–311 Gaussian."],
    ["Near-field RFI model", ["rfi"],
     "Spherical wavefront from emitter position P (ITRF), referenced to the phase centre ŝ<sub>0</sub>. The old GPU branch dropped "
     "ŝ<sub>0</sub>·r<sub>p</sub>; ttcal_dev reads P from WGS84 (el, long, lat) as Cartesian.",
     "experimental.jl:231 gpu_genvis!(…::GPURFISource), :109 uvw_geometry (ŝ<sub>0</sub> from the MS w column). "
     "CPU TTCal: instrument/genvis.jl:245 geometric_delays(…, ::Position)."],
    ["StefCal step", ["stefcal"],
     "Per antenna j and channel: the least-squares Jones matrix given all other gains. Baselines below minuvw and flagged "
     "data are excluded (model flags). The model's squared matrix is Hermitian, which the coalesced kernel uses.",
     "TTCalX kernels/stefcal.jl:159 stefcal_step_full_kernel!, experimental.jl:951 stefcal_step_full_herm_kernel!, peel_gpu.jl:524 "
     "cpu_stefcal_step_full!. CPU TTCal: calibration.jl:488 stefcal_step!, _stefcal_step."],
    ["Solvers", ["rk4", "damped"],
     "CPU TTCal integrates the StefCal step with RK4 (maxiter 5, tol 1e-4) and stops each channel separately. Production "
     "TTCalX takes damped half-steps with one global test (maxiter 30, tol 1e-2) and a gate that rejects the whole source "
     "unless every channel converges.",
     "experimental.jl:564 gpu_stefcal_exp! (solver=:rk4 / :damped). TTCalX peel_gpu.jl:129 gpu_stefcal!. CPU TTCal: "
     "calibration.jl:169 solve!, math/rungekutta.jl:38 rk4!."],
    ["Short baselines and flags", ["minuvw", "herm"],
     "Both codes drop baselines shorter than minuvw wavelengths from the solve (but subtract on all). The model-flag fix "
     "makes the model see the data's flags, so flagged baselines drop out of both the numerator and the denominator.",
     "peel_gpu.jl:48 flag_short_baselines!, experimental.jl:929 flag_short_baselines_gpu!, zest_daemon.jl:188 model flags. "
     "CPU TTCal: peel.jl:35 flag_short_baselines!."],
    ["Geometric UVW", ["uvw"],
     "TTCalX's kernels take UVW from the MS (OVRO-LWA: r<sub>1</sub>−r<sub>2</sub>, unprecessed). Recomputing from ITRF "
     "baselines with R(t) gives the apparent geometry CPU TTCal uses.",
     "experimental.jl:150 j2000_to_itrf_matrix, :195 geometric_uvw!."],
    ["Regularized Peeling (Tukey phase prior)", ["tukey", "regcost"],
     "A source offset Δθ appears as an antenna phase 2πνρ<sub>j</sub>sinΔθ/c (ρ<sub>j</sub> = planar distance from the array "
     "centre). Each diagonal Jones term is replaced by the exact 1-D minimiser of its quadratic StefCal cost minus λ log P "
     "(golden-section search along the phase). In full Jones, the off-diagonal term is first minimised out (Schur complement).",
     "experimental.jl:297 PhaseRegularization, :317 phase_regularization, :335 tukey_prior, :353 regularize_element, "
     ":399/:438 regularized step kernels."],
    ["Divergence guard", ["guard"],
     "After a source is re-subtracted with new gains, the residual power is checked. If it went up, the old gains are restored "
     "and the source is re-subtracted with them.",
     "zest_daemon.jl:90 residual_power, :232 guard; :251 re-subtract on rejection (ZEST_RESUBTRACT)."],
    ["Cost of the pieces", ["cost"],
     "The model evaluation and one StefCal step have similar arithmetic. The step runs 4× per RK4 iteration, 5 iterations, "
     "4 sources and 3 peel iterations (≤240 steps per snapshot), which is why the step and the per-solve flag copies dominated.",
     "Measured: prof_solve.jl, bench_exact.jl (experiments/peeling)."],
]
N["recommendations"].insert(1,
    "If the CPU-exact model is adopted, use the GPU kernel version (genvis_exact!), GPU short-baseline flags, the coalesced step and "
    "read_ms_fast/write_ms_fast!: 2.4 s per snapshot. The StefCal step is now most of the peel time. Next: "
    "parallelise its inner sum over antennas (only N<sub>ant</sub>N<sub>ν</sub> ≈ 17k threads today) and keep the measured square "
    "matrix between the RK4 sub-steps.")
N["recommendations"].insert(2,
    "TTCalX on the CPU is a usable fallback for nodes without a GPU: same output, ~7 s per snapshot with 24 threads, or ~20 s "
    "single-threaded (vs ~130 s for CPU TTCal), so even one core per snapshot is ~6× faster.")
N["where"].append(
    "v3 additions: TTCalX experimental.jl (genvis_exact! kernel/host threads, host backend, flag_short_baselines_gpu!, "
    "stefcal_step_full_herm_kernel!, write_ms_fast!, read_ms_fast), zest_daemon.jl (ZEST_DEVICE, ZEST_FAST_READ, ZEST_FAST_WRITE, ZEST_TIMING), "
    "experiments/peeling: bench_exact.jl, prof_solve.jl, bench2.sh, speed/ (mem_zest.jl, run_mem.sh, writetest.sh, extract.py, "
    "imaging.py, figs_speed.py, readtest.jl, e2e.sh). TTCalX commits edbd26e, a72f4e5 (local).")
json.dump(N, open("notes3.json", "w"), indent=1)
print("ok", len(N["math"]))
