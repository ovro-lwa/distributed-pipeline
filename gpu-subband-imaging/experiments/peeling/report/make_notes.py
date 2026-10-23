import json

notes = {
 "subtitle": "OVRO-LWA 2024-12-20, 24 snapshots (18/23/27/32/36/41/55/73 MHz at 03/08/12 h UT), 48 channels each. "
             "Experiments on copies only; production code and data untouched; nothing pushed. Generated 2026-10-23.",
 "summary": [
  "<b>CPU vs GPU are consistent when run with the same settings.</b> GPU TTCalX with CPU-matched settings removes the same "
  "signal as the main-pipeline CPU TTCal (median correlation of removed visibilities 1.000, image rms ratio 0.999 over 24 snapshots).",
  "<b>Production GPU settings leave more behind.</b> Median image rms is 1.38x the CPU result; in 5/24 snapshots "
  "(27 MHz 03h and the 12h snapshots at 36-73 MHz) nothing is peeled at all. An ablation shows two settings are responsible: "
  "the all-channels convergence gate (ZEST_REQUIRE_CONVERGENCE) and the 15° elevation cut. With both relaxed "
  "(gate off, elevation 0°) production matches CPU (rms ratio 1.00, removed-signal corr. 0.98-1.00).",
  "<b>TTCalX's sky model is mirrored and unprecessed.</b> OVRO-LWA MSes store UVW as r1-r2; TTCalX assumes r2-r1 and uses J2000 "
  "positions without precession. The raw model correlates with the data only after conjugation (0.59 vs CPU 0.81 at 73 MHz). Free "
  "per-channel full-Jones gains hide this in plain peeling. Fixed on the experimental branch (geometric UVW): 0.807, same as CPU.",
  "<b>GPU RFI peeling now works.</b> The new GPU RFI pass reproduces the main pipeline's CPU RFI zest "
  "(removed-signal corr. 0.999-1.000, rms ratio 0.998, RFI residual 0.82-0.90x CPU), both as a separate pass "
  "and zested jointly with the sky. The old feature/rfi-peeling-gpu branch lacked the phase-centre term, and the production "
  "solver's convergence gate rejects the RFI solve when it runs after the sky pass.",
  "<b>The main pipeline's RFI emitter position is wrong.</b> ttcal_dev reads the JSON (el, long, lat) as a Cartesian vector "
  "(lon -3.3°, lat +1.0°). Its raw RFI model correlates 0.005 with the data, versus 0.2-0.3 for the correct position. "
  "Peeling still works because gains absorb it.",
  "<b>Regularized Peeling (Tukey phase prior) is implemented and works as designed, but gives no net gain here.</b> "
  "Gains are confined to the prior. Images are equal or slightly worse (rms +1-4%), and Cyg A residuals grow at tight offsets. "
  "At ≤40 MHz the prior only reaches the core (37% of gains regularized at 5° at 23 MHz), and with correct models nothing needs constraining.",
  "<b>Speed is unchanged.</b> The experimental code at production settings takes 4.3 s per MS (median) vs 4.5 s deployed. "
  "CPU-matched takes 6.2 s; joint RFI 8.9 s; regularized 8.4-10.6 s. The main-pipeline CPU takes about 130 s per MS for sky+RFI.",
 ],
 "setup": [
  "Data: 24 night snapshots from the 2024-12-20 backlog (already 48-channel). Each is copied, bad antennas flagged (MNC, 84 antennas), "
  "and bandpass + XY phase applied as the main pipeline does, giving a prepped DATA column. Every variant peels its own copy.",
  "CPU reference: exactly the main pipeline's commands. Sky: julia060 ttcal.jl zest --beam constant --minuvw 5 --maxiter 5 --tolerance 1e-4. "
  "RFI: ttcal_dev with rfi_43.2_ver20251101.json and the same flags.",
  "GPU: deployed TTCalX (main @62f8c54) with the production daemon and settings (damped StefCal, maxiter 30, tol 1e-2, minuvw 10, "
  "3 peel iterations, elevation ≥ 15°, convergence gate, |g| ≤ 100). The experimental branch adds opt-in env switches: "
  "ZEST_UVW_CONVENTION=geom, ZEST_SOLVER=rk4, ZEST_RFI_SOURCES / ZEST_RFI_MODE=after|joint, ZEST_REG_*, ZEST_SKIP_SHORT, ZEST_SAVE_GAINS_DIR.",
  "Images: WSClean dirty Stokes I/V with the GPU pipeline's tiers (briggs 0, -taper-inner-tukey 30). Metrics: robust rms inside the "
  "horizon; peak |I| in 4° boxes at Cyg A / Cas A / Vir A and at the emitter's apparent position; fraction of power removed and "
  "the correlation of removed visibilities with the CPU's.",
 ],
 "geometry": [
  "Each model is written into a copy of the MS and correlated with the data, with no gains. This isolates model geometry from "
  "calibration freedom.",
  "The MS w column fits w = -s0·(r2-r1) to 0.1 m rms, i.e. UVW = r1-r2. The data follow exp(-ik (r2-r1)·(s-s0)).",
  "TTCalX also leaves J2000 positions unprecessed (~0.35° since 2000). The experimental geometric_uvw! rotates ITRF baselines into "
  "J2000 (IAU-76 precession, leading nutation terms, GAST). It reproduces the CPU model per source to 1-2% "
  "(Cas A: amplitude ratio 0.997, phase rms 0.01 rad).",
  "RFI: with ttcal_dev's misread position the GPU RFI model reproduces CPU's (rel. diff 0.06). Both correlate 0.005 with the data; "
  "the correct position gives 0.31 (23 MHz). Dropping the phase-centre term, as the old branch did, gives 0.28.",
 ],
 "geometry_table": [
  ["model", "73 MHz 03h XX corr", "conj. model", "23 MHz 08h XX corr", "note"],
  ["CPU TTCal sky (julia060)", "0.809", "0.001", "0.244", "reference"],
  ["TTCalX production sky", "0.003", "0.594", "0.001 (conj 0.180)", "mirror image"],
  ["TTCalX + UVW sign fix", "0.597", "0.002", "0.160", "still unprecessed"],
  ["TTCalX + geometric UVW", "0.807", "0.001", "—", "= CPU"],
  ["CPU RFI (ttcal_dev)", "0.004", "0.002", "0.002", "emitter misplaced"],
  ["GPU RFI, correct position", "0.105", "0.001", "0.203 (chan. median 0.31)", "best"],
  ["GPU RFI, no s0 term (old branch)", "0.068", "0.001", "0.180", ""],
 ],
 "sky": [
  "gpu_exp (experimental code at production defaults) gives the same images as deployed production: identical in 22/24 snapshots, "
  "<0.4% different in 2 (log10 power-law fix).",
  "With all CPU settings (rk4 per-channel convergence, maxiter 5, tol 1e-4, minuvw 5, all sources above the horizon, no gate) the "
  "GPU result equals the CPU one. Exception: the 23 MHz snapshots, where the GPU leaves ~10-30% higher rms and a compact residual "
  "at Cas A. That remains open: per-source models agree to 1%, flags are honoured, and leaving short baselines unpeeled "
  "(as CPU does) changes nothing.",
  "The ablation shows the production differences come from source selection and acceptance, not the solver. "
  "1) The convergence gate rejects every solve in some snapshots (27 MHz 03h: Cyg A and Cas A never reach tol 1e-2 in 30 iterations). "
  "2) The 15° elevation cut drops Vir A at 08h (el 2°) and Cas A at 12h (el 7°). Unpeeled, those leave large residuals: "
  "Cas A ≈3000 Jy/beam at 18 MHz 08h vs ≈100 for CPU. Gate off plus elevation ≥ 0° fixes both. Solver choice, tolerance, minuvw and "
  "maxiter barely matter. A tighter tolerance with the gate on is worse: everything is rejected.",
 ],
 "ablation": [
  "'no gate + elev. 0°' is the minimal production change that reaches CPU quality. 'elev. 5°' is not enough at 08h (Vir A at 2°).",
 ],
 "rfi": [
  "gpu_cm_after = the main pipeline's scheme on the GPU (sky zest, then a separate RFI zest), with CPU-like solver settings and the "
  "correct emitter position. It matches the CPU's sky+RFI result (removed-signal corr. 0.999, rms ratio 0.998).",
  "gpu_cm_joint zests the emitter together with Cyg A/Cas A/Vir A. Each source gets its own share instead of the sky pass absorbing "
  "the RFI first (at 23 MHz 08h, Cas A's gain amplitude drops from 2.3 to 0.6-0.9). The RFI residual is the lowest "
  "(0.82x CPU); Cyg A is slightly worse at 03h.",
  "The emitter position barely matters for the images (gpu_cm_aftercpu ≈ gpu_cm_after). Full-Jones per-channel gains can fit any "
  "dominant rank-1 source. A correct position matters only when the gains are constrained (regularization).",
  "With the production solver (damped, tol 1e-2, gate), the RFI solve run after the sky pass is always rejected. The sky pass has "
  "already soaked up most of the RFI, so the residual converges slowly. Run alone, it converges in about 23 iterations.",
  "The RFI pass also clears Cyg A at low frequency (23 MHz 03h: 5715 → 324-460 Jy/beam), CPU and GPU alike: the emitter's "
  "sidelobes corrupted the Cyg A solution.",
 ],
 "reg": [
  "Implemented as in Byrne's note. Each direction-dependent gain element gets a Tukey prior on its phase: flat up to "
  "2πν ρ_j sin(Δθ)/c and cosine-tapered over +1°. Here ρ_j is the antenna's distance from the array centre. Gains with a flat "
  "range ≥ π are left free. The prior enters StefCal as an exact per-antenna minimisation (Schur complement + 1-D phase search). "
  "Unit tests match brute force exactly.",
  "It works mechanically: with a 1° offset the fraction of gain phases beyond the allowed range drops to ≈0. It also tames the "
  "noisy gains of faint, low sources (Vir A at el 2°: amplitude scatter 0.30 → 0.09).",
  "But it does not improve the images. Applied to the sky pass before the RFI pass (the case where Cas A absorbs the RFI), "
  "rms rises 1-4% and Cyg A residuals roughly double at 0.5-2°. At 5° it converges to the unregularized result. "
  "Applied to joint peeling it is similar. Only at 55/73 MHz is joint+reg better, and that gain comes from the joint RFI term, "
  "not the prior.",
  "Why: at ≤40 MHz the flat region reaches π within ~170-340 m, so only core antennas are constrained, and the core "
  "carries the extended emission. Cyg A's gains need ~1 rad of non-geometric phase on the core (structure/ionosphere), which the prior "
  "removes. With correct model geometry (geometric UVW) the gains show no coherent position shift for the prior to stop.",
 ],
 "gains_table": [
  ["23 MHz 03h, gains per source", "phase scatter (rad)", "outside 0.5° prior", "outside 1°", "outside 2°", "outside 5°", "amp. MAD"],
  ["Cyg A, no prior", "1.15", "0.52", "0.45", "0.29", "0.16", "0.455"],
  ["Cyg A, prior 1°", "0.82", "0.49", "0.42", "0.13", "0.00", "0.479"],
  ["Cas A, no prior", "0.22", "0.07", "0.04", "0.02", "0.01", "0.070"],
  ["Cas A, prior 1°", "0.22", "0.12", "0.07", "0.02", "0.00", "0.094"],
  ["RFI, no prior", "0.74", "0.50", "0.38", "0.21", "0.11", "0.455"],
  ["RFI, prior 1°", "0.70", "0.52", "0.39", "0.11", "0.00", "0.592"],
 ],
 "timing_labels": [
  ["gpu_prod", "deployed production TTCalX + daemon, production settings"],
  ["gpu_exp", "experimental code, production settings"],
  ["gpu_geom", "+ geometric UVW"],
  ["gpu_cpumatch", "CPU-matched settings (rk4, maxiter 5, tol 1e-4)"],
  ["gpu_cm_joint", "CPU-matched + joint RFI source"],
  ["gpu_cm_areg1", "CPU-matched + RFI pass, sky regularized 1°"],
  ["gpu_cm_jreg1", "CPU-matched + joint RFI, regularized 1°"],
 ],
 "timing": [
  "Same 8 snapshots, the same idle RTX A4000 (calim01), one daemon per mode. The time is per MS from read to write-back, as "
  "printed by the daemon (ZESTED). The maximum (~30 s) is the first MS, which includes Julia compilation.",
  "The experimental code at production settings is as fast as deployed production. Additions cost only the extra source (RFI) "
  "and the regularized kernels (≈+2 s). CPU TTCal is about 80 s (sky) + 50 s (RFI) per MS.",
  "On calim03 (load ≈50 from main-pipeline workers) the same GPU daemon took ~150-200 s per MS, because the daemon is CPU-bound "
  "outside the kernels. Run GPU peeling on lightly loaded nodes.",
 ],
 "bugs": [
  ["issue", "detail", "impact / fix (experimental branch)"],
  ["TTCalX far-field model mirrored", "UVW sign: the MS stores r1-r2; the kernels assume r2-r1", "hidden by gains; ZEST_UVW_CONVENTION=geom"],
  ["TTCalX ignores precession/nutation", "J2000 l,m about PHASE_DIR + MS UVW", "geometric_uvw! (IAU-76 + nutation + GAST)"],
  ["TTCalX power law uses ln", "CPU TTCal uses log10; curved spectra (Cyg A) 2.7-6x too faint", "fixed (sources.jl)"],
  ["TTCalX Stokes V sign", "xy = U+iV vs CPU U-iV", "RFI source uses the CPU convention; sky unchanged"],
  ["ttcal_dev RFI position misread", "Position(pos\"WGS84\", el, long, lat) with plain numbers → Cartesian", "emitter at lon -3.3°, lat 1.0°; gains absorb it"],
  ["RFI model JSON", "frequency entry 225862 at index 81 (should be ~22.586 MHz)", "fix the JSON"],
  ["old feature/rfi-peeling-gpu", "missing the s0·r phase-centre term", "why it failed; fixed in GPURFISource"],
  ["production convergence gate", "all channels must reach tol 1e-2 in 30 its", "5/24 snapshots peel nothing; the RFI pass is always rejected"],
  ["production elevation cut 15°", "drops Vir A at 2° (08h), Cas A at 7° (12h)", "large residuals; use 0°"],
  ["CaliCo tukey_taper", "uses 2π and signed phase (non-monotonic, asymmetric)", "π and |φ| used here"],
 ],
 "recommendations": [
  "Production GPU pipeline (minimal change, no code): set ZEST_REQUIRE_CONVERGENCE=false and ZEST_MIN_SOURCE_ELEVATION_DEG=0. "
  "This reaches CPU quality on all tested snapshots at the same speed. Keep the max-gain / finite checks as the safety net.",
  "Adopt the geometric UVW fix and the log10 fix in TTCalX. Images barely change, but they are needed for anything that constrains "
  "gains (regularization, fewer gain degrees of freedom, using the model for flux or position).",
  "RFI on the GPU: use the new RFI source with the correct emitter position, either as a pass after the sky (the main-pipeline "
  "scheme) or jointly (lowest RFI residual), with the rk4 / no-gate settings for that pass. Fix ttcal_dev's position parsing "
  "and the corrupt JSON entry in the main pipeline.",
  "Regularized Peeling: keep it as an opt-in. It is not a net win for OVRO-LWA at these settings. It may be worth revisiting with "
  "a scale-aware prior (core-relative phases, or constraining the gradient only), or for faint/low sources only (it does tame Vir A at 2°).",
  "Open: the extra 23 MHz residual of GPU vs CPU (not models, flags or short baselines). Next step: a channel-by-channel gain "
  "comparison of CPU vs GPU at 23 MHz.",
 ],
 "where": [
  "TTCalX branch exp/peeling-experiments (local, not pushed): src/gpu/experimental.jl, sources.jl, TTCalX.jl. "
  "distributed-pipeline branch exp/gpu-peeling-experiments (local): gpu-subband-imaging/workers/zest_daemon.jl and "
  "experiments/peeling/*.",
  "Server: /lustre/nkosogor/peel_exp. The study is in runs/study/<snap>/ (MS + img/), with summary.json and summary.txt. The ablation is "
  "in runs/ablation, timing in runs/timing, figures in runs/report.",
 ],
 "sky_figs": [
  ["charts/sky_rms.png", "Per snapshot: image rms of GPU peeling relative to the CPU main-pipeline zest. Production settings (orange) "
   "vs CPU-matched settings (blue)."],
  ["charts/ablation.png", "Ablation: one production setting changed at a time. Only 'no gate + elevation 0°' and the full CPU "
   "setting set reach CPU quality."],
  ["charts/geometry.png", "Raw model–data correlation (no gains): the production TTCalX model is the mirror image of the sky."],
  ["more/fullsky_41MHz_1200.png", "41 MHz 12h: production peels nothing (Vir A fails the gate, Cas A is below 15°); CPU-matched GPU = CPU."],
  ["more/fullsky_73MHz_0300.png", "73 MHz 03h: Cyg A and Cas A high; all methods are similar."],
  ["figs2/sky/18MHz_0800.png", "18 MHz 08h zooms: the production Cas A residual comes from not peeling Vir A at el 2°."],
 ],
 "rfi_figs": [
  ["charts/rfi_rms.png", "RFI study: rms relative to the CPU sky+RFI zest. GPU sky-then-RFI and joint both ≈1; CPU sky-only "
   "(no RFI pass) is shown for scale."],
  ["charts/rfi_peak.png", "Residual peak at the RFI emitter's position (log scale)."],
  ["more/fullsky_27MHz_0300.png", "27 MHz 03h full sky: production GPU peels nothing (convergence gate); GPU sky+RFI = CPU sky+RFI."],
  ["more/fullsky_23MHz_0800.png", "23 MHz 08h full sky (brightest RFI)."],
  ["figs2/rfi/23MHz_0800.png", "23 MHz 08h: full sky and zooms on the RFI emitter and Cas A for CPU and GPU RFI variants."],
  ["figs2/rfi/18MHz_0800.png", "18 MHz 08h: RFI and Cas A zooms."],
  ["figs2/rfi/32MHz_1200.png", "32 MHz 12h: RFI and Cas A zooms."],
  ["more/onsource_23MHz_0300.png", "23 MHz 03h on-source residuals: the RFI pass is what clears Cyg A. The GPU leaves a compact "
   "Cas A residual (open item)."],
  ["more/onsource_32MHz_0800.png", "32 MHz 08h on-source residuals."],
 ],
 "reg_figs": [
  ["charts/tukey.png", "Tukey phase prior."],
  ["charts/reg.png", "Regularized sky pass (then RFI pass) vs max offset: per-snapshot rms and Cyg A residual relative to the "
   "unregularized run."],
  ["more/gains_23MHz_0300_gpu_cm_jreg1.ms.sky_1.png", "Cyg A gain phase vs antenna distance at 24.9 MHz, without and with the 1° prior. "
   "Red: the flat-region edge (solid) and taper end (dashed). Only antennas within ~340 m are constrained."],
  ["more/gains_23MHz_0300_gpu_cm_jreg1.ms.sky_3.png", "The same for the RFI emitter's gains."],
  ["figs2/reg/23MHz_0300.png", "23 MHz 03h: regularized variants vs unregularized and CPU, zooms on Cyg A, Cas A and RFI."],
  ["more/onsource_18MHz_1200.png", "18 MHz 12h on-source residuals including the regularized sky pass (Vir A residual grows)."],
 ],
}
json.dump(notes, open("notes.json", "w"), indent=1)
