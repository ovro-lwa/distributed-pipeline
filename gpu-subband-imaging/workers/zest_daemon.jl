# Keep TTCalX and CUDA loaded while peeling MS paths read from stdin.
#
# ARGS[1] is the sky model ("none" skips the sky pass).  Optional, all off by
# default (production behaviour unchanged):
#   ZEST_RFI_SOURCES          RFI model; adds a near-field RFI pass after sky
#                             (as the main pipeline: sky zest, then RFI zest)
#   ZEST_RFI_MODE             after (separate pass after sky, as the main pipeline)
#                             | joint (RFI zested together with the sky sources)
#   ZEST_RFI_{MAXITER,TOLERANCE,MINUVW,PEELITER}   RFI-pass overrides (mode=after)
#   ZEST_SOLVER               damped (TTCalX) | rk4 (CPU TTCal solve!)
#   ZEST_REG_MAX_OFFSET_DEG   >0 enables the Tukey phase prior (Regularized Peeling)
#   ZEST_REG_TAPER_DEG        taper width beyond the max offset (deg)
#   ZEST_REG_LAMBDA           prior weight (1 = unit visibility weights)
#   ZEST_REG_TARGETS          all | sky | rfi
#   ZEST_RFI_GEOMETRY         uvw (phase centre from the MS w column) | none (old-branch bug)
#   ZEST_RFI_POSITION_ITRF    "x,y,z" override of the emitter position (m)
#   ZEST_UVW_CONVENTION       as_is (production) | auto (negate UVW when the MS
#                             stores r1-r2, so the far-field model is not mirrored)
#                             | geom (UVW from ITRF baselines rotated to J2000:
#                             apparent geometry with precession/nutation, as CPU TTCal)
#   ZEST_SKIP_SHORT           1: leave baselines < minuvw unpeeled (as CPU TTCal,
#                             which flags them before peeling)
#   ZEST_FLAG_MODEL           1: give each model the data's flags, so flagged
#                             baselines drop out of both sides of StefCal (as CPU
#                             TTCal; otherwise the model's power on flagged baselines
#                             biases gains low in heavily flagged channels)
#   ZEST_GENVIS               kernels (TTCalX) | exact (host model as CPU TTCal:
#                             3-D Gaussian projection, per-component horizon,
#                             U - iV; implies apparent geometry, any UVW convention)
#   ZEST_RESUBTRACT           1: when a solve is rejected, subtract the source again
#                             with its restored (last accepted) gains instead of
#                             leaving it in the data
#   ZEST_GUARD                1: divergence guard; after subtracting a source with
#                             new gains, revert to the previous state if the residual
#                             power (unflagged XX+YY) went up
#   ZEST_SAVE_GAINS_DIR       write per-source gains (<ms>.<pass>.<i>.gains.bin)
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf

envf(k, d) = parse(Float64, get(ENV, k, string(d)))
envi(k, d) = parse(Int, get(ENV, k, string(d)))
envb(k, d) = lowercase(get(ENV, k, string(d))) in ("1", "true", "yes", "on")

sources_path = ARGS[1]
col = get(ENV, "ZEST_COLUMN", "DATA")
mi = envi("ZEST_MAXITER", 30)
tol = envf("ZEST_TOLERANCE", 1e-4)
muvw = envf("ZEST_MINUVW", 10.0)
piter = envi("ZEST_PEELITER", 3)
min_elevation = envf("ZEST_MIN_SOURCE_ELEVATION_DEG", 15.0)
require_convergence = envb("ZEST_REQUIRE_CONVERGENCE", true)
max_gain = envf("ZEST_MAX_GAIN_AMPLITUDE", 100.0)

rfi_path = get(ENV, "ZEST_RFI_SOURCES", "")
rfi_mode = get(ENV, "ZEST_RFI_MODE", "after")
rfi_mi = envi("ZEST_RFI_MAXITER", mi)
rfi_tol = envf("ZEST_RFI_TOLERANCE", tol)
rfi_muvw = envf("ZEST_RFI_MINUVW", muvw)
rfi_piter = envi("ZEST_RFI_PEELITER", piter)
rfi_geometry = Symbol(get(ENV, "ZEST_RFI_GEOMETRY", "uvw"))
rfi_position = let p = get(ENV, "ZEST_RFI_POSITION_ITRF", "")
    isempty(p) ? nothing : parse.(Float64, split(p, ","))
end
uvw_convention = get(ENV, "ZEST_UVW_CONVENTION", "as_is")

solver = Symbol(get(ENV, "ZEST_SOLVER", "damped"))
reg_offset = envf("ZEST_REG_MAX_OFFSET_DEG", 0.0)
reg_taper = envf("ZEST_REG_TAPER_DEG", 1.0)
reg_lambda = envf("ZEST_REG_LAMBDA", 1.0)
reg_targets = get(ENV, "ZEST_REG_TARGETS", "all")
save_gains_dir = get(ENV, "ZEST_SAVE_GAINS_DIR", "")
skip_short = envb("ZEST_SKIP_SHORT", false)
flag_model = envb("ZEST_FLAG_MODEL", false)
genvis_mode = get(ENV, "ZEST_GENVIS", "kernels")
resubtract = envb("ZEST_RESUBTRACT", false)
guard = envb("ZEST_GUARD", false)

"""Residual power of the unflagged XX and YY visibilities."""
residual_power(vis) = sum(abs2, vis.xx .* .!vis.flags) + sum(abs2, vis.yy .* .!vis.flags)
experimental = solver != :damped || reg_offset > 0

function elevation_deg(source, meta)
    is_rfi(source) && return NaN
    lmn = source_direction_lmn(
        source, meta.phase_center_ra, meta.phase_center_dec, 0.0
    )
    rad2deg(asin(clamp(lmn[3], -1.0, 1.0)))
end

function save_calibration(cal)
    (copy(cal.xx), copy(cal.xy), copy(cal.yx), copy(cal.yy), copy(cal.flags))
end

function restore_calibration!(cal, saved)
    cal.xx .= saved[1]
    cal.xy .= saved[2]
    cal.yx .= saved[3]
    cal.yy .= saved[4]
    cal.flags .= saved[5]
end

function gain_quality(cal)
    finite = true
    max_amplitude = 0.0
    for values in (cal.xx, cal.xy, cal.yx, cal.yy)
        host = Array(values)
        finite &= all(isfinite, host)
        max_amplitude = max(max_amplitude, maximum(abs, host))
    end
    finite, max_amplitude
end

function solve!(calibration, vis, coherency, meta, maxiter, tolerance, minuvw, reg)
    if !experimental
        converged, niters = gpu_stefcal!(
            calibration, vis, coherency, meta;
            maxiter=maxiter, tolerance=tolerance, minuvw=minuvw
        )
        return converged, niters, converged ? 1.0 : 0.0
    end
    gpu_stefcal_exp!(calibration, vis, coherency, meta;
                     maxiter=maxiter, tolerance=tolerance, minuvw=minuvw,
                     solver=solver, reg=reg)
end

function safe_zest_gpu!(vis, meta, catalog_sources; pass="sky", maxiter=mi,
                        tolerance=tol, minuvw=muvw, peeliter=piter,
                        ms="", mjd=NaN)
    selected_sources = GPUSource[]
    elevations = Float64[]
    for source in catalog_sources
        elevation = elevation_deg(source, meta)
        selected = is_rfi(source) || elevation >= min_elevation
        @printf(
            "PEEL_SOURCE pass=%s name=\"%s\" elevation_deg=%.2f selected=%s threshold_deg=%.2f\n",
            pass, get_name(source), elevation, selected, min_elevation
        )
        if selected
            push!(selected_sources, source)
            push!(elevations, elevation)
        end
    end

    if isempty(selected_sources)
        println("PEEL_SUMMARY pass=$pass selected=0 accepted=0 action=unchanged")
        return
    end

    reg = nothing
    if reg_offset > 0 && (reg_targets == "all" || reg_targets == pass)
        reg = phase_regularization(meta, reg_offset, reg_taper; lambda=reg_lambda)
        @printf("PEEL_REG pass=%s max_offset_deg=%.3f taper_deg=%.3f lambda=%.3g regularized_fraction=%.3f\n",
                pass, reg_offset, reg_taper, reg_lambda, regularized_fraction(reg))
    end

    peel_sources = [GPUZestingSource(source) for source in selected_sources]
    calibrations = [
        calibration_type(source, meta.Nant, meta.Nfreq; gpu=true)
        for source in peel_sources
    ]
    coherencies = GPUVisibilities[]
    for source in peel_sources
        coherency = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=true)
        if genvis_mode == "exact" && !is_rfi(source)
            genvis_exact!(coherency, meta, unwrap(source), mjd)
        elseif is_rfi(source)
            gpu_genvis!(coherency, meta, unwrap(source), meta.phase_center_ra,
                        meta.phase_center_dec, 0.0; geometry=rfi_geometry,
                        position=rfi_position)
        else
            gpu_genvis!(
                coherency, meta, unwrap(source),
                meta.phase_center_ra, meta.phase_center_dec, 0.0
            )
        end
        skip_short && zero_short_baselines!(coherency, meta, minuvw)
        flag_model && (coherency.flags .= vis.flags)
        push!(coherencies, coherency)
    end

    subtracted = trues(length(peel_sources))
    accepted_final = falses(length(peel_sources))
    for (coherency, calibration) in zip(coherencies, calibrations)
        corrupted = deepcopy_gpu(coherency)
        gpu_corrupt!(corrupted, calibration, meta)
        gpu_subsrc!(vis, corrupted)
    end

    for iteration in 1:peeliter
        for index in eachindex(peel_sources)
            source = peel_sources[index]
            coherency = coherencies[index]
            calibration = calibrations[index]
            power_before = guard ? residual_power(vis) : 0.0
            was_subtracted = subtracted[index]
            if subtracted[index]
                corrupted = deepcopy_gpu(coherency)
                gpu_corrupt!(corrupted, calibration, meta)
                gpu_putsrc!(vis, corrupted)
                subtracted[index] = false
            end

            saved = save_calibration(calibration)
            converged, niters, frac = solve!(calibration, vis, coherency, meta,
                                             maxiter, tolerance, minuvw, reg)
            finite, max_amplitude = gain_quality(calibration)
            accepted = finite && max_amplitude <= max_gain &&
                       (!require_convergence || converged)
            @printf(
                "PEEL_SOLVE pass=%s name=\"%s\" elevation_deg=%.2f iteration=%d/%d converged=%s niters=%d chan_converged=%.3f finite=%s max_gain=%.6g accepted=%s\n",
                pass, get_name(source), elevations[index], iteration, peeliter,
                converged, niters, frac, finite, max_amplitude, accepted
            )

            if accepted
                corrupted = deepcopy_gpu(coherency)
                gpu_corrupt!(corrupted, calibration, meta)
                gpu_subsrc!(vis, corrupted)
                subtracted[index] = true
                accepted_final[index] = true
                if guard
                    power_after = residual_power(vis)
                    if power_after > power_before
                        # undo: put the new model back, restore and re-subtract the old one
                        gpu_putsrc!(vis, corrupted)
                        restore_calibration!(calibration, saved)
                        if was_subtracted
                            corrupted = deepcopy_gpu(coherency)
                            gpu_corrupt!(corrupted, calibration, meta)
                            gpu_subsrc!(vis, corrupted)
                        end
                        subtracted[index] = was_subtracted
                        accepted_final[index] = was_subtracted
                        @printf("PEEL_GUARD pass=%s name=\"%s\" iteration=%d power_ratio=%.4g reverted=true\n",
                                pass, get_name(source), iteration, power_after / power_before)
                    end
                end
            else
                restore_calibration!(calibration, saved)
                if resubtract && accepted_final[index]
                    corrupted = deepcopy_gpu(coherency)
                    gpu_corrupt!(corrupted, calibration, meta)
                    gpu_subsrc!(vis, corrupted)
                    subtracted[index] = true
                else
                    accepted_final[index] = false
                end
            end
        end
    end
    if !isempty(save_gains_dir)
        for (index, calibration) in enumerate(calibrations)
            tag = basename(dirname(ms)) * "_" * basename(ms)
            path = joinpath(save_gains_dir, "$(tag).$(pass).$(index).gains.bin")
            dims = save_calibration_bin(path, calibration)
            open(path * ".json", "w") do io
                write(io, "{\"name\": \"$(get_name(peel_sources[index]))\", \"shape\": [$(join(dims, ", "))], " *
                          "\"layout\": \"complex128 (Nant, Nfreq, [xx,xy,yx,yy]) column-major\"}")
            end
        end
    end
    @printf(
        "PEEL_SUMMARY pass=%s selected=%d accepted=%d action=safe_zest\n",
        pass, length(peel_sources), count(accepted_final)
    )
end

init_pycasacore() || error("python-casacore not available")
sources = sources_path == "none" ? GPUSource[] : read_gpu_sources(sources_path)
rfi_sources = isempty(rfi_path) ? GPUSource[] : read_gpu_sources(rfi_path)
set_verbosity(:quiet)
device_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "unavailable"
println(
    "DAEMON_READY sources=$(length(sources)) device=$device_name " *
    "min_elevation=$min_elevation require_convergence=$require_convergence " *
    "max_gain=$max_gain rfi_sources=$(length(rfi_sources)) solver=$solver " *
    "reg_max_offset_deg=$reg_offset uvw_convention=$uvw_convention rfi_mode=$rfi_mode"
)
flush(stdout)

for line in eachline(stdin)
    ms = String(strip(line))
    isempty(ms) && continue
    ms == "QUIT" && break
    t0 = time()
    try
        vis, cal, meta, bl, nrows = read_ms_to_gpu(ms; gpu=true, column=col)
        mjd = genvis_mode == "exact" ? ms_mjd(ms) : NaN
        if uvw_convention == "auto"
            println("UVW_CONVENTION sign=$(apply_uvw_convention!(meta))")
        elseif uvw_convention == "geom"
            rms, offset = geometric_uvw!(meta, ms_mjd(ms))
            @printf("UVW_GEOMETRIC rms_w_vs_ms=%.4f phase_centre_offset_deg=%.4f\n", rms, offset)
        end
        if rfi_mode == "joint"
            joint = vcat(sources, rfi_sources)
            isempty(joint) || safe_zest_gpu!(vis, meta, joint; ms=ms, mjd=mjd)
        else
            isempty(sources) || safe_zest_gpu!(vis, meta, sources; ms=ms, mjd=mjd)
        end
        if !isempty(rfi_sources) && rfi_mode != "joint"
            safe_zest_gpu!(vis, meta, rfi_sources; pass="rfi", maxiter=rfi_mi,
                           tolerance=rfi_tol, minuvw=rfi_muvw, peeliter=rfi_piter,
                           ms=ms, mjd=mjd)
        end
        write_gpu_to_ms!(ms, vis, bl, nrows; column=col)
        @printf("ZESTED %s %.2f\n", ms, time() - t0)
    catch e
        println("FAILED $ms $(sprint(showerror, e))")
    end
    flush(stdout)
    CUDA.reclaim()
end
println("DAEMON_EXIT"); flush(stdout)
