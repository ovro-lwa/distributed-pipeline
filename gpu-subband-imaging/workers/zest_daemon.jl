# Keep TTCalX and CUDA loaded while peeling MS paths read from stdin.
#
# ARGS[1] is the sky model.  Settings come from ZEST_* variables
# (PeelParams.daemon_env); the defaults are the original GPU peeling:
#   ZEST_COLUMN, ZEST_MAXITER, ZEST_TOLERANCE, ZEST_MINUVW, ZEST_PEELITER,
#   ZEST_MIN_SOURCE_ELEVATION_DEG, ZEST_REQUIRE_CONVERGENCE, ZEST_MAX_GAIN_AMPLITUDE
#   ZEST_SOLVER        damped (gpu_stefcal!) | rk4 (CPU TTCal's solve!, solve_gains!)
#   ZEST_MODEL         kernels (TTCalX genvis) | exact (CPU TTCal's model, genvis_exact!)
#   ZEST_UVW           ms | geometric (precessed ITRF baselines, geometric_uvw!)
#   ZEST_FLAG_MODEL    give each model the data's flags, so flagged visibilities
#                      drop out of both sides of the solve (as CPU TTCal)
#   ZEST_RESUBTRACT    when a solve is rejected, subtract the source again with
#                      its last accepted gains instead of leaving it in the data
#   ZEST_GUARD         undo a source update that raises the residual power
#   ZEST_RFI_SOURCES   RFI model (near-field emitters); zested after the sky
#                      (ZEST_RFI_MODE=after, the main pipeline's order) or with it (joint)
#   ZEST_FAST_IO       read_ms_fast / write_ms_fast! (same results, faster)
#   ZEST_DEVICE        gpu | cpu (TTCalX on Julia threads; needs ZEST_MODEL=exact)
#   ZEST_REG_MAX_OFFSET_DEG, ZEST_REG_TAPER_DEG, ZEST_REG_LAMBDA
#                      Tukey phase prior on the gains (off when the offset is 0)
# Anything but the defaults needs TTCalX with solve_gains!
# (branch feature/cpu-matched-zest).
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf

envf(k, d) = parse(Float64, get(ENV, k, string(d)))
envi(k, d) = parse(Int, get(ENV, k, string(d)))
envb(k, d) = lowercase(get(ENV, k, string(d))) in ("1", "true", "yes", "on")
function envs(k, d, allowed)
    v = lowercase(get(ENV, k, d))
    v in allowed || error("$k=$v; expected one of $(join(allowed, ", "))")
    v
end

sources_path = ARGS[1]
col = get(ENV, "ZEST_COLUMN", "DATA")
mi = envi("ZEST_MAXITER", 30)
tol = envf("ZEST_TOLERANCE", 1e-4)
muvw = envf("ZEST_MINUVW", 10.0)
piter = envi("ZEST_PEELITER", 3)
min_elevation = envf("ZEST_MIN_SOURCE_ELEVATION_DEG", 15.0)
require_convergence = envb("ZEST_REQUIRE_CONVERGENCE", true)
max_gain = envf("ZEST_MAX_GAIN_AMPLITUDE", 100.0)
solver = Symbol(envs("ZEST_SOLVER", "damped", ("damped", "rk4")))
exact_model = envs("ZEST_MODEL", "kernels", ("kernels", "exact")) == "exact"
geometric = envs("ZEST_UVW", "ms", ("ms", "geometric")) == "geometric"
flag_model = envb("ZEST_FLAG_MODEL", false)
resubtract = envb("ZEST_RESUBTRACT", false)
guard = envb("ZEST_GUARD", false)
rfi_path = get(ENV, "ZEST_RFI_SOURCES", "")
rfi_joint = envs("ZEST_RFI_MODE", "after", ("after", "joint")) == "joint"
fast_io = envb("ZEST_FAST_IO", false)
use_gpu = envs("ZEST_DEVICE", "gpu", ("gpu", "cpu")) == "gpu"
reg_offset = envf("ZEST_REG_MAX_OFFSET_DEG", 0.0)
reg_taper = envf("ZEST_REG_TAPER_DEG", 1.0)
reg_lambda = envf("ZEST_REG_LAMBDA", 1.0)

# The original code path (gpu_stefcal!, kernel model, MS UVW, GPU) needs no new
# TTCalX API, so the defaults also run with older TTCalX versions.
extended = solver != :damped || exact_model || geometric || fast_io || !use_gpu ||
           reg_offset > 0 || !isempty(rfi_path)
extended && !isdefined(TTCalX, :solve_gains!) &&
    error("these ZEST_* settings need TTCalX with solve_gains! (branch feature/cpu-matched-zest)")
use_gpu || exact_model || error("ZEST_DEVICE=cpu needs ZEST_MODEL=exact")

is_rfi_source(source) = isdefined(TTCalX, :is_rfi) && TTCalX.is_rfi(source)

"""Residual power of the unflagged XX and YY visibilities."""
residual_power(vis) = sum(abs2, vis.xx .* .!vis.flags) + sum(abs2, vis.yy .* .!vis.flags)

function elevation_deg(source, meta)
    is_rfi_source(source) && return NaN
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

"""Solve one direction's gains; returns (converged, niters)."""
function solve!(calibration, vis, coherency, meta, reg)
    if solver == :damped && reg === nothing
        return gpu_stefcal!(calibration, vis, coherency, meta;
                            maxiter=mi, tolerance=tol, minuvw=muvw)
    end
    converged, niters, _ = solve_gains!(calibration, vis, coherency, meta; solver=solver,
                                        maxiter=mi, tolerance=tol, minuvw=muvw, reg=reg)
    converged, niters
end

function model_visibilities!(coherency, meta, source, mjd)
    if exact_model && !is_rfi_source(source)
        genvis_exact!(coherency, meta, source, mjd)
    else
        gpu_genvis!(coherency, meta, source, meta.phase_center_ra, meta.phase_center_dec, 0.0)
    end
    coherency
end

function subtract_with_gains!(vis, coherency, calibration, meta)
    corrupted = deepcopy_gpu(coherency)
    gpu_corrupt!(corrupted, calibration, meta)
    gpu_subsrc!(vis, corrupted)
end

function add_with_gains!(vis, coherency, calibration, meta)
    corrupted = deepcopy_gpu(coherency)
    gpu_corrupt!(corrupted, calibration, meta)
    gpu_putsrc!(vis, corrupted)
end

function safe_zest!(vis, meta, catalog_sources, mjd; pass="sky")
    selected_sources = GPUSource[]
    elevations = Float64[]
    for source in catalog_sources
        elevation = elevation_deg(source, meta)
        selected = is_rfi_source(source) || elevation >= min_elevation
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

    reg = reg_offset > 0 ?
        phase_regularization(meta, reg_offset, reg_taper; lambda=reg_lambda, gpu=use_gpu) : nothing
    peel_sources = [GPUZestingSource(source) for source in selected_sources]
    calibrations = [
        calibration_type(source, meta.Nant, meta.Nfreq; gpu=use_gpu)
        for source in peel_sources
    ]
    coherencies = GPUVisibilities[]
    for source in peel_sources
        coherency = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=use_gpu)
        model_visibilities!(coherency, meta, unwrap(source), mjd)
        flag_model && (coherency.flags .= vis.flags)
        push!(coherencies, coherency)
    end

    subtracted = trues(length(peel_sources))
    accepted_final = falses(length(peel_sources))
    for (coherency, calibration) in zip(coherencies, calibrations)
        subtract_with_gains!(vis, coherency, calibration, meta)
    end

    for iteration in 1:piter
        for index in eachindex(peel_sources)
            source = peel_sources[index]
            coherency = coherencies[index]
            calibration = calibrations[index]
            power_before = guard ? residual_power(vis) : 0.0
            was_subtracted = subtracted[index]
            if subtracted[index]
                add_with_gains!(vis, coherency, calibration, meta)
                subtracted[index] = false
            end

            saved = save_calibration(calibration)
            converged, niters = solve!(calibration, vis, coherency, meta, reg)
            finite, max_amplitude = gain_quality(calibration)
            accepted = finite && max_amplitude <= max_gain &&
                       (!require_convergence || converged)
            @printf(
                "PEEL_SOLVE pass=%s name=\"%s\" elevation_deg=%.2f iteration=%d/%d converged=%s niters=%d finite=%s max_gain=%.6g accepted=%s\n",
                pass, get_name(source), elevations[index], iteration, piter,
                converged, niters, finite, max_amplitude, accepted
            )

            if accepted
                subtract_with_gains!(vis, coherency, calibration, meta)
                subtracted[index] = true
                accepted_final[index] = true
                if guard && residual_power(vis) > power_before
                    # Undo: put the new model back, restore and re-subtract the old one.
                    add_with_gains!(vis, coherency, calibration, meta)
                    restore_calibration!(calibration, saved)
                    was_subtracted && subtract_with_gains!(vis, coherency, calibration, meta)
                    subtracted[index] = was_subtracted
                    accepted_final[index] = was_subtracted
                    @printf("PEEL_GUARD pass=%s name=\"%s\" iteration=%d reverted=true\n",
                            pass, get_name(source), iteration)
                end
            else
                restore_calibration!(calibration, saved)
                if resubtract && accepted_final[index]
                    subtract_with_gains!(vis, coherency, calibration, meta)
                    subtracted[index] = true
                else
                    accepted_final[index] = false
                end
            end
        end
    end
    @printf(
        "PEEL_SUMMARY pass=%s selected=%d accepted=%d action=safe_zest\n",
        pass, length(peel_sources), count(accepted_final)
    )
end

init_pycasacore() || error("python-casacore not available")
sources = read_gpu_sources(sources_path)
rfi_sources = isempty(rfi_path) ? GPUSource[] : read_gpu_sources(rfi_path)
set_verbosity(:quiet)
device_name = !use_gpu ? "cpu threads=$(Threads.nthreads())" :
              CUDA.functional() ? CUDA.name(CUDA.device()) : "unavailable"
println(
    "DAEMON_READY sources=$(length(sources)) device=$device_name " *
    "min_elevation=$min_elevation require_convergence=$require_convergence " *
    "max_gain=$max_gain solver=$solver model=$(exact_model ? "exact" : "kernels") " *
    "uvw=$(geometric ? "geometric" : "ms") rfi_sources=$(length(rfi_sources)) " *
    "rfi_mode=$(rfi_joint ? "joint" : "after")"
)
flush(stdout)

read_ms = fast_io ? read_ms_fast : read_ms_to_gpu
write_ms! = fast_io ? write_ms_fast! : write_gpu_to_ms!

for line in eachline(stdin)
    ms = String(strip(line))
    isempty(ms) && continue
    ms == "QUIT" && break
    t0 = time()
    try
        t_read = @elapsed (vis, cal, meta, bl, nrows) = read_ms(ms; gpu=use_gpu, column=col)
        t_peel = @elapsed begin
            mjd = (exact_model || geometric) ? ms_mjd(ms) : NaN
            if geometric
                rms_w, offset = geometric_uvw!(meta, mjd)
                @printf("UVW_GEOMETRIC rms_w_vs_ms=%.4f phase_centre_offset_deg=%.4f\n", rms_w, offset)
            end
            if rfi_joint && !isempty(rfi_sources)
                safe_zest!(vis, meta, vcat(sources, rfi_sources), mjd)
            else
                safe_zest!(vis, meta, sources, mjd)
                isempty(rfi_sources) || safe_zest!(vis, meta, rfi_sources, mjd; pass="rfi")
            end
        end
        t_write = @elapsed write_ms!(ms, vis, bl, nrows; column=col)
        @printf("ZEST_TIMING read=%.2f peel=%.2f write=%.2f\n", t_read, t_peel, t_write)
        @printf("ZESTED %s %.2f\n", ms, time() - t0)
    catch e
        println("FAILED $ms $(sprint(showerror, e))")
    end
    flush(stdout)
    use_gpu && CUDA.reclaim()
end
println("DAEMON_EXIT"); flush(stdout)
