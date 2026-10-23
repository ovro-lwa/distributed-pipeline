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
    "TTCal's ~130 s, and the DATA and FLAG written are byte-identical to the unoptimized run. Over all 96 snapshots on lustre: median "
    "2.5 s (sky) and 3.2 s (sky then RFI) per snapshot, with results equal to CPU TTCal.")
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
    "To match CPU TTCal everywhere, drop (or raise) the daemon's ZEST_MAX_GAIN_AMPLITUDE=100 rejection. It is the only remaining "
    "GPU/CPU difference in sky and sky-then-RFI mode (13 MHz 03h). Better still, skip the RFI zest outside the RFI model's "
    "14.8–40.9 MHz range, where it can only fit noise.")
N["recommendations"].insert(3,
    "TTCalX on the CPU is a usable fallback for nodes without a GPU: same output, ~7 s per snapshot with 24 threads, or ~20 s "
    "single-threaded (vs ~130 s for CPU TTCal), so even one core per snapshot is ~6× faster.")
N["where"].append(
    "v3 additions: TTCalX experimental.jl (genvis_exact! kernel/host threads, host backend, flag_short_baselines_gpu!, "
    "stefcal_step_full_herm_kernel!, write_ms_fast!, read_ms_fast), zest_daemon.jl (ZEST_DEVICE, ZEST_FAST_READ, ZEST_FAST_WRITE, ZEST_TIMING), "
    "experiments/peeling: bench_exact.jl, prof_solve.jl, bench2.sh, speed/ (mem_zest.jl, run_mem.sh, writetest.sh, extract.py, "
    "imaging.py, figs_speed.py, readtest.jl, e2e.sh). TTCalX commits edbd26e, a72f4e5 (local).")
N["glance_intro"] = (
    "The optimized code uses the same configuration (CPU-matched solver, CPU-exact model, sky then RFI or joint) and gives the "
    "same output as before. Each change was checked against the code it replaces; rows below are in the order they were applied.")
N["glance_rows"] = [
    ["step", "what was slow → what it does now", "code (TTCalX src/gpu/experimental.jl unless noted)", "time before → after", "check"],
    ["1. CPU-exact sky model", "single-threaded host loop over an abstract-typed component list (every operation dynamically "
     "dispatched), then a host→GPU copy → per-component tables on the host, one GPU thread per (baseline, channel)",
     "genvis_exact! :832, genvis_exact_kernel! :812, _exact_vis :790, _exact_tables :759 (old: genvis_exact_reference! :713)",
     "5.5 s → 0.01 s per snapshot", "≤2e-9 relative"],
    ["2. Short-baseline flags", "TTCalX's flag_short_baselines! copies the four N<sub>ant</sub>×N<sub>ant</sub>×N<sub>chan</sub> "
     "matrices to the host and back, twice per solve → a broadcast mask on the GPU",
     "flag_short_baselines_gpu! :929, used in gpu_stefcal_exp! :564", "87 ms → 0.2 ms per call", "bit-identical"],
    ["3. StefCal step", "thread j read M[i, j] (addresses N<sub>ant</sub> apart) → reads M[j, i]<super>H</super> (Hermitian), so "
     "neighbouring threads read neighbouring addresses", "stefcal_step_full_herm_kernel! :951, called from _stefcal_steps! :514",
     "10.4 → 8.5 ms per step; one solve 0.48 → 0.15 s (with 2)", "2e-16"],
    ["4. MS read", "whole columns converted numpy→Julia through PyCall, rows gathered in Julia → rows gathered in numpy",
     "read_ms_fast (end of file), _ms_subtables", "1.3 → 0.4 s", "all arrays identical"],
    ["5. MS write", "DATA and FLAG re-read, converted to Julia, scattered, converted back as complex128 → numpy scatter on "
     "casacore's own arrays", "write_ms_fast!", "1.6–1.9 → 0.23 s (/tmp)", "byte-identical DATA, FLAG"],
    ["6. TTCalX on CPU (new)", "CUDA kernels were launched unconditionally; the cpu_* fallbacks were single-threaded → the same "
     "per-element arithmetic on host arrays, one channel per Julia thread", "gpu_corrupt!/gpu_subsrc! host methods :874, "
     "host_makesquare! :886, _host_stefcal_steps! :907; daemon ZEST_DEVICE=cpu", "– → 7.2 s per snapshot (24 threads)", "≤3e-8 of rms"],
]
N["code_used"] = [
    "<b>Code:</b> TTCalX branch exp/peeling-experiments @ a72f4e5 (src/gpu/experimental.jl; the original TTCalX files are unchanged) "
    "and distributed-pipeline branch exp/gpu-peeling-experiments @ 3054c20 (gpu-subband-imaging/workers/zest_daemon.jl). Both are "
    "local only.",
    "<b>Settings</b> (environment of the daemon): ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 "
    "ZEST_PEELITER=3 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_UVW_CONVENTION=geom ZEST_FLAG_MODEL=1 "
    "ZEST_RESUBTRACT=1 ZEST_GENVIS=exact ZEST_FAST_READ=1 ZEST_FAST_WRITE=1; RFI: ZEST_RFI_SOURCES=rfi_43.2_ver20251101.json with "
    "ZEST_RFI_MODE=after (main pipeline's order) or joint; CPU: ZEST_DEVICE=cpu JULIA_NUM_THREADS=24. Sky model: the main "
    "pipeline's sources.json.",
    "<b>Tests and benchmarks</b> (experiments/peeling): bench_exact.jl (model and solve: timing and equality), prof_solve.jl "
    "(solve breakdown), speed/readtest.jl (reader equality), speed/writetest.sh and speed/e2e.sh (end-to-end, byte "
    "comparison), speed/mem_zest.jl + run_mem.sh (in-memory GPU/CPU peel, thread scaling), run_variant.sh fast_sky / fast_after / "
    "fast_joint + compare_fast.py (all 96 snapshots).",
]
N["vpol"] = [
    "<b>Polarization calibration as in the pipeline.</b> Every snapshot was prepared like prepare_one_ms_task "
    "(orca/tasks/subband_tasks.py): bad antennas flagged, then CASA applycal with the night's bandpass and the regular XY-phase "
    "table /lustre/gh/polcal/xyphase_delay_pos_3.8643ns.Xf (the only table in polcal), then zest: sky, then RFI. CPU = the main "
    "pipeline's ttcal.jl zest commands (julia060 sky, ttcal_dev RFI). GPU = the optimized TTCalX daemon. Stokes V comes from the same "
    "WSClean runs as I (-pol IV).",
    "<b>Result:</b> Stokes V after peeling equals CPU's. V rms ratio median 1.000 in all three modes. |V| at Cas A, Cyg A and "
    "Vir A is 1.000–1.013× CPU (boxes where peeling removed most of the V). At the RFI emitter, whose model carries Stokes V up to "
    "10 kJy, the GPU leaves 0.98× CPU's |V|. That also confirms the V sign convention (xy = U − iV) in the GPU RFI and sky models: "
    "the opposite sign would add V instead of removing it.",
    "The V outliers are the same snapshots as in Stokes I: 13 MHz 03h (gain limit, see above; 1.11× CPU, 1.004× without the "
    "limit) and the unstable joint cases. The deployed production GPU settings leave 3× CPU's |V| at Vir A (Vir A skipped by "
    "the gate/elevation cut) and up to 2.8× the V rms.",
]
N["vpol_figs"] = [
    ["charts/vpol_band.png", "Per subband: Stokes V rms after peeling relative to CPU TTCal."],
    ["vpol/V_fullsky_full_1220_23MHz_0300.png", "12-20 23 MHz 03h Stokes V: before, CPU sky→RFI, GPU sky→RFI, GPU − CPU (difference rms 0.45 vs 4.35 image rms)."],
    ["vpol/V_onsource_full_1220_23MHz_0300.png", "12-20 23 MHz 03h Stokes V ±4° around Cas A, Cyg A and the RFI emitter."],
    ["vpol/V_fullsky_full_1227_23MHz_0800.png", "12-27 23 MHz 08h Stokes V."],
    ["vpol/V_onsource_full_1227_23MHz_0800.png", "12-27 23 MHz 08h Stokes V around the targets."],
    ["vpol/V_fullsky_full_1220_41MHz_1200.png", "12-20 41 MHz 12h Stokes V."],
    ["vpol/V_fullsky_full_1227_69MHz_1200.png", "12-27 69 MHz 12h Stokes V."],
    ["vpol/V_fullsky_full_1220_13MHz_0300.png", "12-20 13 MHz 03h Stokes V: the gain-limit outlier (Cyg A and RFI solves rejected by ZEST_MAX_GAIN_AMPLITUDE=100)."],
]
N["answers"].insert(5,
    "<b>Polarization:</b> with the pipeline's own bandpass + XY-phase calibration applied before zest, Stokes V after GPU sky and "
    "sky+RFI zest equals CPU TTCal's on all 96 snapshots (V rms ratio 1.000; |V| at the sources 1.00–1.01×; RFI emitter 0.98×).")
N["review_intro"] = [
    "<b>What was submitted:</b> TTCalX branch <b>feature/cpu-matched-zest</b> on GitHub (nkosogor/TTCalX), based on the "
    "current main (98b6561). Three commits: 200c5c6 (sources.jl: base-10 power law, RFI entries), 9fdd308 (seven new "
    "files in src/gpu/, about 1,040 lines), 94f062e (unit tests, a CI step, README). No pull request was opened; CI runs "
    "when one is opened against main. The distributed-pipeline daemon changes are not in this branch.",
    "<b>How it was checked:</b>"
    "<br/>• Unit tests (test/runtests.jl): 1016 pass on calim03 with the GPU, 982 with the GPU hidden (as in CI), 956 on "
    "Julia 1.11 (macOS, fresh dependency resolve). quick_test.jl passes.<br/>"
    "• An independent code review; its findings were fixed and tested.<br/>"
    "• Real data: on three OVRO-LWA snapshots, sky-then-RFI zest with the branch writes DATA and FLAG <b>byte-identical</b> "
    "to the experimental code validated on all 96 snapshots, and the CPU mode and the prior agree to ≤2.6e-7 of the rms.",
    "<b>Where to spend review time:</b> about 450 lines carry the risk (table below, highest first). Everything else is "
    "plumbing, ports of standard formulas, or tests, and can be skimmed.",
]
N["review_high"] = [
    ["file:lines (branch)", "what it is", "why it is risky / what to check", "how it is verified"],
    ["src/gpu/sources.jl:24–35", "GPUPowerLaw: natural log → log10",
     "<b>Changes every existing caller</b>, including the deployed GPU pipeline once deployed: curved spectra (Cyg A) get "
     "brighter, single-index spectra are unchanged. Check it matches TTCal's PowerLaw (sky/spectra.jl).",
     "unit test with a curved spectrum; GPU = CPU TTCal on 96 snapshots"],
    ["src/gpu/solver.jl:44–113", "_full_normal_equations, _full_step",
     "Core StefCal math. M[i,j] is read as M[j,i]ᴴ (xy ↔ yx swapped and conjugated), which is valid only because the "
     "square matrices are Hermitian. The unregularized branch (84–92) must stay the 2×2 solve of the GPU kernel. The "
     "prior branch (94–112) does a Schur-complement update.",
     "equals TTCalX's cpu_stefcal_step_full! to 1e-12; GPU = host; real data byte-identical"],
    ["src/gpu/solver.jl:225–303", "solve_gains!",
     "RK4 with per-channel freezing (275–300) must match CPU TTCal's solve! (calibration.jl:169): the step that converges "
     "is still applied, and frozen channels keep their gains. Also the damped branch (257–273) and the prior/device "
     "checks (234–242).",
     "noiseless/noisy recovery, diagonal, minuvw, flags, invalid-argument tests"],
    ["src/gpu/genvis_exact.jl:43–113", "exact_model_tables, _exact_vis, kernel",
     "Determines GPU = CPU equality. Check the signs (delay τ = r·(ŝ - ŝ₀), cis(-k(τ₂ - τ₁)) at 92), Stokes→linear "
     "(xy = U - iV, 62), Gaussian widths and axes, and the geocentric horizon test (50), which is ~0.2° from CPU TTCal's.",
     "matches a CPU-TTCal-style reference loop to 1e-8; GPU = host; 96-snapshot study"],
    ["src/gpu/geometry.jl:29–104", "j2000_to_itrf_matrix, uvw_geometry, geometric_uvw!",
     "Precession/nutation/GAST formulas and the UVW sign (b = r₂ - r₁, E rows at 89, 99). An error here mirrors or "
     "rotates the sky model; free gains would partly hide it.",
     "rotation, precession and sidereal-day tests; recovers the synthetic MS phase centre and sign"],
    ["src/gpu/rfi.jl:105–133", "near-field RFI model",
     "Phase sign (124), and ŝ₀ fitted from meta.uvw (114): it must be called after geometric_uvw! (documented; the s0 "
     "keyword overrides it).",
     "tends to the analytic plane wave far away (incl. V sign); RFI residual 0.93× CPU"],
    ["src/gpu/ms_io.jl:22–84, 133–175", "read_ms_fast / write_ms_fast! (numpy)",
     "Writes the MS: a row-mapping error would corrupt data. Check the fallback conditions (several rows per baseline, "
     "not 4 polarizations), the 0/1-based index conversion (PyReverseDims) and the padding at 64–67.",
     "same arrays as read_ms_to_gpu; byte-identical DATA/FLAG vs write_gpu_to_ms! (also with a partial baseline map)"],
]
N["review_low"] = [
    ["file", "why lower risk"],
    ["src/gpu/regularization.jl", "Opt-in only (no prior unless reg is passed). regularize_element (93) is a 1-D search, unit-tested for its limits."],
    ["src/gpu/host_backend.jl", "Opt-in CPU mode. Thin per-channel wrappers around TTCalX's own cpu_corrupt!/cpu_subsrc!, plus a makesquare tested equal to the GPU kernel."],
    ["src/gpu/solver.jl:20–35, 116–200", "mask_short_baselines! (bit-identical to flag_short_baselines!), the diagonal step (the existing kernel's arithmetic, tested equal), and the kernel/thread dispatch."],
    ["src/gpu/rfi.jl:1–100", "Spectrum lookup ported from ttcal_dev, standard WGS84 → ITRF, JSON parsing with validation; all unit-tested."],
    ["src/TTCalX.jl", "Includes and exports only."],
    ["test/*, .github/workflows/ci.yml, README.md", "Tests (skim helpers.jl to trust the fixtures), one CI step, documentation. quick_test.jl: a one-line fix of a check that errored on GPU hosts."],
]
N["review_notes"] = [
    "Not in this branch: the daemon changes (ZEST_* switches, guard, re-subtract on rejection, model flags, fast I/O "
    "switches) live in distributed-pipeline exp/gpu-peeling-experiments and still call the experimental name "
    "gpu_stefcal_exp!; they become solve_gains! when that branch is prepared.",
    "Known differences from CPU TTCal kept on purpose: the horizon test is geocentric (≤0.2°), and autocorrelations are "
    "not peeled (unchanged TTCalX behaviour).",
]
for m in N["math"]:
    m[3] = (m[3].replace("experimental.jl:759 _exact_tables, :790 _exact_vis, :812 genvis_exact_kernel!, :832 genvis_exact!, :684 _gaussian_axes, :713 genvis_exact_reference!",
                         "genvis_exact.jl:43 exact_model_tables, :82 _exact_vis, :104 genvis_exact_kernel!, :128 genvis_exact!, :19 _gaussian_axes")
                .replace("experimental.jl:231 gpu_genvis!(…::GPURFISource), :109 uvw_geometry", "rfi.jl:105 gpu_genvis!(…::GPURFISource), geometry.jl:58 uvw_geometry")
                .replace("experimental.jl:951 stefcal_step_full_herm_kernel!", "solver.jl:44 _full_normal_equations, :76 _full_step, :169 stefcal_steps!")
                .replace("experimental.jl:564 gpu_stefcal_exp! (solver=:rk4 / :damped)", "solver.jl:225 solve_gains! (solver=:rk4 / :damped)")
                .replace("experimental.jl:929 flag_short_baselines_gpu!", "solver.jl:20 mask_short_baselines!")
                .replace("experimental.jl:150 j2000_to_itrf_matrix, :195 geometric_uvw!", "geometry.jl:29 j2000_to_itrf_matrix, :83 geometric_uvw!")
                .replace("experimental.jl:297 PhaseRegularization, :317 phase_regularization, :335 tukey_prior, :353 regularize_element, :399/:438 regularized step kernels",
                         "regularization.jl:26 PhaseRegularization, :53 phase_regularization, :71 tukey_prior, :93 regularize_element; solver.jl:94–112 prior update"))
N["where"].append("<b>Submitted:</b> TTCalX branch feature/cpu-matched-zest (GitHub, commits 200c5c6, 9fdd308, 94f062e); "
                  "see section 10 for the review guide. The distributed-pipeline branch is not yet prepared.")
N["answers"].insert(6, "<b>Code submitted for review:</b> TTCalX branch feature/cpu-matched-zest, with unit tests. Its real-data "
                    "output is byte-identical to the validated code. Section 10 lists the ~450 high-risk lines to review "
                    "and what can be skimmed.")
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "ₙ": "n"}


def _subs(x):
    """Unicode subscripts → <sub> (the PDF font has no subscript glyphs)."""
    if isinstance(x, str):
        for k, v in SUB.items():
            x = x.replace(k, f"<sub>{v}</sub>")
        return x.replace("ᴴ", "<super>H</super>")
    if isinstance(x, list):
        return [_subs(y) for y in x]
    if isinstance(x, dict):
        return {k: _subs(v) for k, v in x.items()}
    return x


N = _subs(N)
json.dump(N, open("notes3.json", "w"), indent=1)
print("ok", len(N["math"]))
