import json

N = {
 "title": "GPU vs CPU peeling, GPU RFI peeling and Regularized Peeling — OVRO-LWA (v2: two dates, all subbands)",
 "subtitle": "2024-12-20 and 2024-12-27, all 16 subbands (13–82 MHz) at 03/08/12 h UT = 96 snapshots, 48 channels each. "
             "Experiments on copies only; production code and data untouched; nothing pushed. Generated 2026-10-23.",
 "answers": [
  "<b>Does the GPU do the same job as the CPU main pipeline?</b> Yes, once the bugs below are fixed. With CPU-matched solver settings "
  "plus the model-flag fix, the GPU result equals CPU TTCal on both dates and all 16 subbands: median image rms ratio 0.999–1.000, "
  "removed-signal correlation 1.000. With the CPU-exact model it agrees to 1.000 in rms and in every source residual.",
  "<b>Production GPU settings</b> peel nothing at all in 24 of 96 snapshots (convergence gate, 15° elevation cut). Median rms is 1.38× CPU. "
  "Switching the gate off with the damped solver is not safe: about a tenth of low-band snapshots then diverge (the gate was hiding them). "
  "The safe fix is the rk4 per-channel solver (as CPU) + elevation 0° + model flags, optionally with the new divergence guard.",
  "<b>Sky + RFI zest works on the GPU.</b> The main pipeline's scheme (sky zest, then RFI zest) and a joint zest both reproduce "
  "the CPU result on both dates (rms ratio 1.000, removed-signal corr. 0.999). The RFI residual is 0.88–0.95× CPU's.",
  "<b>Speed:</b> CPU TTCal sky+RFI takes ~130 s per snapshot (sky 78 s + RFI 52 s). GPU sky+RFI takes ~11 s per snapshot "
  "(~17 s including the daemon's one-off start-up), about 10× faster. The CPU-exact model costs ~27 s.",
  "<b>Regularized Peeling</b> (Tukey phase prior) does not reduce source suppression. Peeling absorbs 13–68% of a source 1° from Cas A "
  "(up to 48% at 2°), and any prior leaves that untouched: the absorption needs only ~0.03 rad of gain phase. Images are otherwise "
  "~1% worse. Where it does help is stability: it prevents the divergence of joint sky+RFI solves when Cas A is near the horizon "
  "(27 MHz 12h: rms 25 → 2.5).",
  "<b>Bugs:</b> 11 issues found and fixed or worked around (table, section 1). The largest are the missing model flags, the mirrored "
  "and unprecessed sky model, the convergence gate and an elevation cut that silently skip sources, and ttcal_dev's misread RFI "
  "position.",
 ],
 "setup": [
  "Snapshots: first averaged snapshot of 03, 08 and 12 h UT for every subband, on 2024-12-20 and 2024-12-27 "
  "(82 MHz on 12-20 has no 12 h data; 11 h is used). Each is copied, bad antennas flagged (MNC), and the night's bandpass "
  "+ XY phase applied, as in the main pipeline. Every variant peels its own copy of the same prepared DATA.",
  "CPU reference = the main pipeline's exact commands: julia060 ttcal.jl zest --beam constant --minuvw 5 --maxiter 5 --tolerance 1e-4 "
  "(sky), then ttcal_dev with rfi_43.2_ver20251101.json (RFI).",
  "GPU = deployed TTCalX (main @62f8c54) with production settings, and the experimental branch with opt-in env switches "
  "(all off by default): ZEST_SOLVER=rk4, ZEST_UVW_CONVENTION=geom, ZEST_GENVIS=exact, ZEST_FLAG_MODEL, ZEST_RESUBTRACT, "
  "ZEST_GUARD, ZEST_RFI_SOURCES / ZEST_RFI_MODE=after|joint, ZEST_REG_*. 'CPU-matched' = rk4, maxiter 5, tol 1e-4, minuvw 5, "
  "sources above the horizon, no gate, geometric UVW, model flags, resubtract.",
  "Metrics: WSClean dirty Stokes I with the GPU pipeline's settings. Robust rms inside the horizon; peak |I| in 4° boxes at Cyg A / "
  "Cas A / Vir A and the RFI emitter; fraction of power removed and the correlation of removed visibilities with the CPU's.",
 ],
 "bugs": [
  ["issue", "detail", "impact", "status / fix (experimental, opt-in)"],
  ["Model visibilities never get the data flags", "makesquare of the model includes flagged baselines → StefCal denominator biased low in heavily flagged channels",
   "high at 23 MHz (bright RFI channels); was the whole 23 MHz GPU-vs-CPU gap", "ZEST_FLAG_MODEL=1 → identical to CPU"],
  ["Convergence gate rejects whole solves", "all 48 channels must reach tol 1e-2 in 30 damped iterations, else the source is not peeled",
   "24/96 snapshots peel nothing", "use rk4 per-channel solver; no gate"],
  ["Elevation cut 15°", "Vir A at 2°/Cas A at 7° left in the data", "large residuals at 08/12 h", "ZEST_MIN_SOURCE_ELEVATION_DEG=0"],
  ["Damped solver unstable without the gate", "unconverged half-steps run away", "~11/96 low-band snapshots diverge", "rk4 solver; ZEST_GUARD"],
  ["Rejected solve leaves the source unsubtracted", "daemon restores gains but never re-subtracts", "source fully left in the output", "ZEST_RESUBTRACT=1"],
  ["Far-field model mirrored", "kernels assume UVW = r2−r1; OVRO-LWA MS has r1−r2", "hidden by gains", "ZEST_UVW_CONVENTION=geom"],
  ["No precession/nutation", "J2000 l,m about PHASE_DIR", "~0.35° offset; hidden by gains", "geometric_uvw!"],
  ["Power law ln instead of log10", "curved spectra (Cyg A) several × too faint", "model flux", "fixed (sources.jl)"],
  ["Gaussian shapes", "PA measured from east, wrong rotation angle, no 3-D foreshortening", "~2–4% of model on long baselines", "ZEST_GENVIS=exact (= CPU to 1.000)"],
  ["Stokes V sign", "xy = U+iV vs CPU U−iV", "V model", "exact model / RFI source use CPU convention"],
  ["ttcal_dev RFI emitter misplaced", "WGS84 (el,long,lat) read as Cartesian → lon −3.3°, lat +1.0°", "RFI model uncorrelated with data; gains absorb it", "GPU RFI source uses the correct position"],
  ["RFI model frequency range", "model covers ≤40.9 MHz (+ corrupt entry 225862 at index 81) but RFI zest runs on all subbands",
   "above 41 MHz the 'RFI' direction is a free extra gain set", "restrict to ≤41 MHz or extend the model"],
  ["Old feature/rfi-peeling-gpu branch", "dropped the s0·r phase-centre term", "RFI peeling failed", "fixed in GPURFISource"],
  ["Autocorrelations never peeled on GPU", "MS bridge drops autos", "only matters for products using autos", "open"],
  ["Elevation from first component / PHASE_DIR", "≤0.2° vs astropy", "negligible", "open"],
  ["(my code) genvis_exact! uninitialised buffers", "similar() + accumulate gave random YY/XY/YX on busy nodes", "random divergences", "fixed 59edbbc; runs redone"],
 ],
 "bugs_note": "Correction to the v1 report: CPU TTCal does subtract on short baselines. It solves on deep copies with baselines below "
              "minuvw flagged, then subtracts with the final gains on all baselines. ZEST_SKIP_SHORT therefore does not mimic CPU "
              "and is not used. The bug review was done by reading both code bases; every high-impact item was then confirmed on data.",
 "sky": [
  "Both dates and all subbands agree. Production peels nothing in 24/96 snapshots, mostly 12 h and at 50–82 MHz, where Vir A fails the "
  "gate and Cas A is below 15°.",
  "CPU-matched settings reproduce CPU everywhere (rms 0.999, corr 1.000). The one outlier (12-20 23 MHz 08h, 1.31×) is the run "
  "without the model-flag fix, i.e. the flag bug itself. With the flag fix and the CPU-exact model, all ratios are 1.000 with no outliers "
  "in 96 snapshots.",
  "'production + 3 fixes' (damped solver, gate off, elevation 0°, model flags) matches CPU in the median but diverges in a tenth of "
  "low-band snapshots. The convergence gate was masking an unstable solver, not just being over-strict.",
 ],
 "rfi": [
  "Both GPU schemes reproduce the main pipeline's sky+RFI result. The emitter residual is somewhat lower than CPU's, even though CPU "
  "uses a misplaced emitter: full-Jones per-channel gains fit any dominant rank-1 source.",
  "Joint zest (RFI emitter together with Cyg A/Cas A/Vir A) is as good on average, but can diverge when Cas A is low near the emitter's "
  "direction (12-20 27 MHz 12h). The Tukey prior or the divergence guard prevents this.",
  "The production (damped) solver with joint RFI diverges often at low frequency: use the rk4 settings for RFI peeling.",
  "Above ~41 MHz the RFI model has no meaningful flux, yet the main pipeline still runs the RFI zest; it then acts as an extra free "
  "gain direction (it still lowers Cas A residuals at 46 MHz 12h).",
 ],
 "speed": [
  "Same 8 snapshots (13–82 MHz), same idle node (calim01) and GPU (RTX A4000). CPU = the main pipeline's two ttcal.jl calls per snapshot, "
  "each a new Julia process. GPU = one daemon for all 8 snapshots; per-snapshot times from the daemon (read → peel → write).",
  "On a heavily loaded node (calim03, load ~50) the GPU daemon slows to 100–200 s per snapshot, because reading/writing the MS and "
  "model generation run on the CPU.",
 ],
 "reg": [
  "Injection test: four 300 Jy flat-spectrum sources were added 1°, 2°, 4° and 8° from Cas A (5 snapshots, 23–73 MHz). "
  "Peeling with and without them was compared (difference of dirty images). Sources within 2° of Cas A lose 13–68% (1°) and up to 48% (2°) of their "
  "flux to Cas A's gains, on CPU and GPU alike. The phase prior changes this by at most a few percent, at any offset "
  "(0.1–5°) or taper. Absorbing the injected flux moves Cas A's gains by only ~0.03 rad and ~3% (23–41 MHz). A phase prior "
  "physically allowing even 0.1° permits more than that, so it cannot stop the suppression.",
  "Over all 96 snapshots the joint zest with a prior of 0.25°, 1° or 3° (taper = offset) changes rms by a median of +1%.",
  "Where it helps: on 12-20 at 27 MHz 12h the unregularized joint zest diverges (Cas A at el 7°, near the RFI emitter's direction; "
  "rms 25 vs 2.5). All regularized runs stay stable and give the lowest Cas A and RFI residuals of any method, CPU included. "
  "Regularization is useful as a stabiliser for nearly degenerate multi-direction solves, not as a flux-suppression cure.",
  "Higher frequencies: the prior's flat region scales as ν·ρ·sinΔθ, so at 50–82 MHz more antennas are constrained for a given offset. "
  "Results there are the same: no improvement for the sky sources, and stabilisation only.",
 ],
 "recommendations": [
  "Production GPU pipeline: adopt the CPU-matched settings (ZEST_SOLVER=rk4, maxiter 5, tol 1e-4, minuvw 5, elevation 0°, no gate) "
  "and the model-flag fix. Optionally add ZEST_GUARD as a safety net. This reproduces the main pipeline at ~10× its speed.",
  "Merge the TTCalX fixes (UVW convention / precession, log10, model flags, exact Gaussian model) and make the daemon re-subtract on "
  "rejection. Images barely change today because free gains absorb model errors, but anything that constrains gains depends on them.",
  "RFI: use the GPU RFI source (correct emitter position), as a pass after the sky or jointly, with rk4 settings. In the main pipeline, "
  "fix ttcal_dev's position parsing and the corrupt JSON entry, and decide whether RFI peeling above 41 MHz is intended.",
  "Regularized Peeling: keep it opt-in. Enable it (≈1°) for joint sky+RFI zest when a peel target is within ~10° of the horizon.",
  "Run the GPU daemon on lightly loaded nodes. Don't run more than 3 daemons per 16 GB GPU (out-of-memory failures leave the MS unpeeled).",
 ],
 "where": [
  "Code (local branches, not pushed): TTCalX exp/peeling-experiments (src/gpu/experimental.jl); distributed-pipeline "
  "exp/gpu-peeling-experiments (gpu-subband-imaging/workers/zest_daemon.jl, experiments/peeling/*: run_variant.sh, prep_date.sh, "
  "inject.py, inj_measure.py, study_summary.py, study_tables.py, report scripts).",
  "Server: /lustre/nkosogor/peel_exp — data/ and data_1227/ (prepared inputs), runs/full_1220 and runs/full_1227 "
  "(MS + img/ + summary.json), runs/inj* (injection), runs/bench_* (timing), runs/report (figures).",
 ],
 "sky_figs": [
  ["charts/band_sky.png", "Per subband: image rms relative to CPU (line = median over hours, dots = snapshots)."],
  ["charts/band_exact.png", "Kernel vs CPU-exact model with the CPU-matched solver."],
  ["v2/d1227/fullsky_82MHz_0300.png", "12-27 82 MHz 03h full sky (same colour scale)."],
  ["v2/d1227/onsource_82MHz_0300.png", "12-27 82 MHz 03h on-source residuals."],
  ["v2/d1227/fullsky_59MHz_1200.png", "12-27 59 MHz 12h: production peels nothing (Vir A fails the gate, Cas A below 15°)."],
  ["v2/d1220/fullsky_13MHz_0300.png", "12-20 13 MHz 03h (lowest subband)."],
 ],
 "rfi_figs": [
  ["charts/band_rfi.png", "Per subband: sky+RFI rms relative to the CPU sky+RFI zest."],
  ["charts/band_rfi_peak.png", "Residual at the RFI emitter relative to before peeling."],
  ["v2/d1227/fullsky_23MHz_0800.png", "12-27 23 MHz 08h full sky."],
  ["v2/d1227/onsource_23MHz_0800.png", "12-27 23 MHz 08h on-source residuals for all main variants."],
 ],
 "reg_figs": [
  ["charts/injection.png", "Injection test: fraction of injected flux surviving peeling vs distance from Cas A."],
  ["charts/band_reg.png", "Regularized vs unregularized joint zest, all snapshots (the 27 MHz 12h point on 12-20 is ~0.1: divergence prevented)."],
  ["inj/23MHz_0800_recovered.png", "23 MHz 08h: injected-source residual images (injected − not injected) after peeling."],
  ["inj/41MHz_0800_recovered.png", "41 MHz 08h: same."],
  ["v2/d1220/onsource_27MHz_1200.png", "12-20 27 MHz 12h: unregularized joint zest diverges; every regularized run is clean."],
  ["v2/d1220/fullsky_27MHz_1200.png", "12-20 27 MHz 12h full sky."],
  ["more/gains_23MHz_0300_gpu_cm_jreg1.ms.sky_1.png", "Cyg A gain phases vs antenna distance without and with the 1° prior."],
  ["charts/tukey.png", "Tukey phase prior."],
 ],
 "appendix": [
  "v1 study, 24 snapshots of 2024-12-20 at 8 subbands, before the model-flag fix (hence the 23 MHz differences there). "
  "Kept for the geometry check and the first ablation of production settings.",
 ],
 "appendix_figs": [
  ["charts/geometry.png", "Raw model–data correlation: the production TTCalX sky model is the mirror image of the sky."],
  ["charts/ablation.png", "First ablation (8 snapshots): elevation cut and gate are what matter for production."],
  ["more/onsource_23MHz_0300.png", "12-20 23 MHz 03h on-source residuals (v1 variants)."],
 ],
}
json.dump(N, open("notes2.json", "w"), indent=1)
