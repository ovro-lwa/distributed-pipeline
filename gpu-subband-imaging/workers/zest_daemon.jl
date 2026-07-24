# Keep TTCalX and CUDA loaded while peeling MS paths read from stdin.
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf

sources_path = ARGS[1]
col = get(ENV, "ZEST_COLUMN", "DATA")
mi = parse(Int, get(ENV, "ZEST_MAXITER", "30"))
tol = parse(Float64, get(ENV, "ZEST_TOLERANCE", "1e-4"))
muvw = parse(Float64, get(ENV, "ZEST_MINUVW", "10.0"))
piter = parse(Int, get(ENV, "ZEST_PEELITER", "3"))
min_elevation = parse(
    Float64, get(ENV, "ZEST_MIN_SOURCE_ELEVATION_DEG", "15.0")
)
require_convergence = lowercase(
    get(ENV, "ZEST_REQUIRE_CONVERGENCE", "true")
) in ("1", "true", "yes", "on")
max_gain = parse(Float64, get(ENV, "ZEST_MAX_GAIN_AMPLITUDE", "100.0"))

function elevation_deg(source, meta)
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

function safe_zest_gpu!(vis, meta, catalog_sources)
    selected_sources = GPUSource[]
    elevations = Float64[]
    for source in catalog_sources
        elevation = elevation_deg(source, meta)
        selected = elevation >= min_elevation
        @printf(
            "PEEL_SOURCE name=\"%s\" elevation_deg=%.2f selected=%s threshold_deg=%.2f\n",
            get_name(source), elevation, selected, min_elevation
        )
        if selected
            push!(selected_sources, source)
            push!(elevations, elevation)
        end
    end

    if isempty(selected_sources)
        println("PEEL_SUMMARY selected=0 accepted=0 action=unchanged")
        return
    end

    peel_sources = [GPUZestingSource(source) for source in selected_sources]
    calibrations = [
        calibration_type(source, meta.Nant, meta.Nfreq; gpu=true)
        for source in peel_sources
    ]
    coherencies = GPUVisibilities[]
    for source in peel_sources
        coherency = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=true)
        gpu_genvis!(
            coherency, meta, unwrap(source),
            meta.phase_center_ra, meta.phase_center_dec, 0.0
        )
        push!(coherencies, coherency)
    end

    subtracted = trues(length(peel_sources))
    accepted_final = falses(length(peel_sources))
    for (coherency, calibration) in zip(coherencies, calibrations)
        corrupted = deepcopy_gpu(coherency)
        gpu_corrupt!(corrupted, calibration, meta)
        gpu_subsrc!(vis, corrupted)
    end

    for iteration in 1:piter
        for index in eachindex(peel_sources)
            source = peel_sources[index]
            coherency = coherencies[index]
            calibration = calibrations[index]
            if subtracted[index]
                corrupted = deepcopy_gpu(coherency)
                gpu_corrupt!(corrupted, calibration, meta)
                gpu_putsrc!(vis, corrupted)
                subtracted[index] = false
            end

            saved = save_calibration(calibration)
            converged, niters = gpu_stefcal!(
                calibration, vis, coherency, meta;
                maxiter=mi, tolerance=tol, minuvw=muvw
            )
            finite, max_amplitude = gain_quality(calibration)
            accepted = finite && max_amplitude <= max_gain &&
                       (!require_convergence || converged)
            @printf(
                "PEEL_SOLVE name=\"%s\" elevation_deg=%.2f iteration=%d/%d converged=%s niters=%d finite=%s max_gain=%.6g accepted=%s\n",
                get_name(source), elevations[index], iteration, piter,
                converged, niters, finite, max_amplitude, accepted
            )

            if accepted
                corrupted = deepcopy_gpu(coherency)
                gpu_corrupt!(corrupted, calibration, meta)
                gpu_subsrc!(vis, corrupted)
                subtracted[index] = true
                accepted_final[index] = true
            else
                restore_calibration!(calibration, saved)
                accepted_final[index] = false
            end
        end
    end
    @printf(
        "PEEL_SUMMARY selected=%d accepted=%d action=safe_zest\n",
        length(peel_sources), count(accepted_final)
    )
end

init_pycasacore() || error("python-casacore not available")
sources = read_gpu_sources(sources_path)
set_verbosity(:quiet)
device_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "unavailable"
println(
    "DAEMON_READY sources=$(length(sources)) device=$device_name " *
    "min_elevation=$min_elevation require_convergence=$require_convergence " *
    "max_gain=$max_gain"
)
flush(stdout)

for line in eachline(stdin)
    ms = String(strip(line))
    isempty(ms) && continue
    ms == "QUIT" && break
    t0 = time()
    try
        vis, cal, meta, bl, nrows = read_ms_to_gpu(ms; gpu=true, column=col)
        safe_zest_gpu!(vis, meta, sources)
        write_gpu_to_ms!(ms, vis, bl, nrows; column=col)
        @printf("ZESTED %s %.2f\n", ms, time() - t0)
    catch e
        println("FAILED $ms $(sprint(showerror, e))")
    end
    flush(stdout)
    CUDA.reclaim()
end
println("DAEMON_EXIT"); flush(stdout)
