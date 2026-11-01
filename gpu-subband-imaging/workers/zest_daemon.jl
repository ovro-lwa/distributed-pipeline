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
#   ZEST_RESIDUAL_GUARD  TTCalX.guarded_zest!: keep a source update only if subtracting
#                      it lowers the residual (vs. source left in and vs. previous gains),
#                      else keep the previous gains or leave the source in
#   ZEST_FLAG_ZERO     flag samples with XX == YY == 0 before peeling (dropped data)
#   ZEST_BACKSTOP      undo a whole sky/RFI pass that raises the residual power
#   ZEST_RFI_SOURCES   RFI model (near-field emitters); zested after the sky
#                      (ZEST_RFI_MODE=after, the main pipeline's order) or with it (joint)
#   ZEST_FAST_IO       read_ms_fast / write_ms_fast! (same results, faster)
#   ZEST_DEVICE        gpu | cpu (TTCalX on Julia threads; needs ZEST_MODEL=exact)
#   ZEST_REG_MAX_OFFSET_DEG, ZEST_REG_TAPER_DEG, ZEST_REG_LAMBDA
#                      Tukey phase prior on the gains (off when the offset is 0)
#   ZEST_TEST_IO       tests only: "" (peel) | identity (no peel: read + write back) |
#                      perturb (no peel: scale the visibilities by 1.0001 e^{0.3i} and invert
#                      the flags of every 7th baseline), for the slot I/O equivalence test
#   ZEST_ENGINE        daemon (the loop below) | sky_rfi (TTCalX.zest_sky_rfi!, branch
#                      feature/sponge-restore-fast-zest) with ZEST_SMOOTH_FLUX, ZEST_RESTORE,
#                      ZEST_RESTORE_BELOW_MHZ, ZEST_RESTORE_OFFSET_DEG, ZEST_PRECISION,
#                      ZEST_WARM_START (the exact model, geometric UVW, flagged models and the
#                      residual guard are built in), and the smoothing switches
#                      ZEST_SMOOTH_BELOW_MHZ (> 0: smooth only subbands whose centre frequency
#                      is below it), ZEST_SMOOTH_EXEMPT_FLUX (> 0: sky sources with catalog flux
#                      >= it keep per-channel gains), ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG (> 0: sky
#                      sources below this elevation keep them) and ZEST_RFI_SMOOTH_FLUX (unset:
#                      the ZEST_SMOOTH_FLUX value), all unset by default
# Anything but the defaults needs TTCalX with solve_gains!
# (branch feature/cpu-matched-zest).
#
# Commands (one per line): an MS path (peel it in place), "SHM <slot> [<id>]" (gsi v3: peel a
# /dev/shm slot written by src/gpu_subband_imaging/v3/slots.py in place, see read_slot /
# write_slot! below; ZEST_COLUMN must be DATA), "RESET_WARM [<id>]" (drop the warm-start
# gains), or QUIT.  Replies: the PEEL_* / ZEST_TIMING lines, then "ZESTED <path> <seconds>"
# or "FAILED <path> <error on one line>"; slot commands append " id=<id>" to ZESTED and put
# "id=<id>" right after the path in FAILED; RESET_WARM answers "RESET_WARM_OK id=<id>".
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
include(joinpath(@__DIR__, "zest_smooth.jl"))
using .TTCalX
using CUDA
using Printf
using Mmap
using JSON

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
residual_guard = envb("ZEST_RESIDUAL_GUARD", false)
flag_zero = envb("ZEST_FLAG_ZERO", false)
backstop = envb("ZEST_BACKSTOP", false)
(residual_guard || flag_zero || backstop) && !isdefined(TTCalX, :guarded_zest!) &&
    error("ZEST_RESIDUAL_GUARD/FLAG_ZERO/BACKSTOP need TTCalX with guarded_zest! (branch fix/zest-divergence-guard)")
rfi_path = get(ENV, "ZEST_RFI_SOURCES", "")
rfi_joint = envs("ZEST_RFI_MODE", "after", ("after", "joint")) == "joint"
fast_io = envb("ZEST_FAST_IO", false)
use_gpu = envs("ZEST_DEVICE", "gpu", ("gpu", "cpu")) == "gpu"
reg_offset = envf("ZEST_REG_MAX_OFFSET_DEG", 0.0)
reg_taper = envf("ZEST_REG_TAPER_DEG", 1.0)
reg_lambda = envf("ZEST_REG_LAMBDA", 1.0)
sky_rfi = envs("ZEST_ENGINE", "daemon", ("daemon", "sky_rfi")) == "sky_rfi"
smooth_flux = envf("ZEST_SMOOTH_FLUX", 0.0)
smooth_below_mhz = envf("ZEST_SMOOTH_BELOW_MHZ", 0.0)       # 0: smooth every subband
smooth_exempt_flux = envf("ZEST_SMOOTH_EXEMPT_FLUX", 0.0)   # 0: no source exempt
rfi_smooth_flux = envf("ZEST_RFI_SMOOTH_FLUX", smooth_flux)
# ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG > 0: sky sources below this elevation (in this snapshot) keep
# per-channel gains under smoothing (low-elevation models need them; robust across LST, see
# docs/peel_config.md); needs TTCalX zest_sky_rfi!(...; smooth_exempt_names)
smooth_exempt_below_el = envf("ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG", 0.0)
for (k, v) in (("ZEST_SMOOTH_FLUX", smooth_flux), ("ZEST_SMOOTH_BELOW_MHZ", smooth_below_mhz),
               ("ZEST_SMOOTH_EXEMPT_FLUX", smooth_exempt_flux), ("ZEST_RFI_SMOOTH_FLUX", rfi_smooth_flux),
               ("ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG", smooth_exempt_below_el))
    v >= 0 || (println("CONFIG_ERROR $k must be >= 0, got $v"); flush(stdout); exit(2))
end
!(sky_rfi && smooth_flux > 0 && smooth_exempt_below_el > 0) ||
    (isdefined(TTCalX, :zest_sky_rfi!) &&
     any(m -> :smooth_exempt_names in Base.kwarg_decl(m), methods(TTCalX.zest_sky_rfi!))) ||
    (println("CONFIG_ERROR ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG needs TTCalX zest_sky_rfi!(...; smooth_exempt_names)");
     flush(stdout); exit(2))
restore_path = get(ENV, "ZEST_RESTORE", "")
restore_below_mhz = envf("ZEST_RESTORE_BELOW_MHZ", Inf)
restore_offset_deg = envf("ZEST_RESTORE_OFFSET_DEG", 0.0)
precision = Symbol(envs("ZEST_PRECISION", "double", ("double", "mixed")))
warm_start = envb("ZEST_WARM_START", false)
test_io = Symbol(envs("ZEST_TEST_IO", "", ("", "identity", "perturb")))
sky_rfi && !isdefined(TTCalX, :zest_sky_rfi!) &&
    (println("CONFIG_ERROR ZEST_ENGINE=sky_rfi needs TTCalX with zest_sky_rfi!"); flush(stdout); exit(2))
# zest_sky_rfi! runs on host arrays as well (TTCalX host backend), so device=cpu is allowed
sky_rfi && !(fast_io && exact_model && geometric && residual_guard && flag_model) &&
    error("ZEST_ENGINE=sky_rfi is the exact-model, geometric-UVW, guarded zest: " *
                      "set model=exact uvw=geometric residual_guard flag_model fast_io")

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

    if residual_guard
        solvefn = (cal, v, coh) -> solve!(cal, v, coh, meta, reg)
        report = (s, it; converged, niters, finite, max_gain, p_with, p_old, p_new, action) -> @printf(
            "PEEL_SOLVE pass=%s name=\"%s\" elevation_deg=%.2f iteration=%d/%d converged=%s niters=%d finite=%s max_gain=%.6g accepted=%s action=%s p_ratio_new=%.4g p_ratio_old=%.4g\n",
            pass, get_name(peel_sources[s]), elevations[s], it, piter, converged, niters, finite,
            max_gain, action == :accept, action, p_new / p_with, p_old / p_with)
        sub = guarded_zest!(vis, meta, coherencies, calibrations, solvefn; peeliter=piter,
                            max_gain=max_gain, require_convergence=require_convergence, report=report)
        @printf("PEEL_SUMMARY pass=%s selected=%d accepted=%d action=guarded_zest\n",
                pass, length(peel_sources), count(sub))
        return
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
restore_sources = isempty(restore_path) ? GPUSource[] : read_gpu_sources(restore_path)
any(s -> get_name(s) in ("Sun", "Moon", "Jupiter"), restore_sources) &&
    error("ZEST_RESTORE: a source named Sun/Moon/Jupiter is read as a placeholder; rename it")
sky_rfi && println(@sprintf("SKY_RFI smooth_flux=%g restore=%d below_mhz=%g offset_deg=%g precision=%s warm_start=%s peeliter=%d smooth_below_mhz=%g smooth_exempt_flux=%g rfi_smooth_flux=%g smooth_exempt_below_el_deg=%g",
                           smooth_flux, length(restore_sources), restore_below_mhz, restore_offset_deg,
                           precision, warm_start, piter, smooth_below_mhz, smooth_exempt_flux,
                           rfi_smooth_flux, smooth_exempt_below_el))
flush(stdout)

# zest_sky_rfi! for one snapshot; returns the gains for a warm start of the next one
function run_sky_rfi!(vis, meta, mjd, initial_gains)
    # smoothing switches: the subband-centre gate and the bright-source exemption
    sflux, rflux = smooth_flux, rfi_smooth_flux
    if smooth_below_mhz > 0
        centre_mhz = sum(Array(meta.channels)) / length(meta.channels) / 1e6
        gated = centre_mhz >= smooth_below_mhz
        gated && (sflux = rflux = 0.0)
        @printf("PEEL_SMOOTH_GATE centre_mhz=%.3f below_mhz=%g smoothing=%s\n", centre_mhz,
                smooth_below_mhz, (sflux == 0 && rflux == 0) ? "off" : "on")
    end
    elev = Dict(get_name(s) => elevation_deg(s, meta) for s in vcat(sources, rfi_sources))
    rfi_names = Set(get_name.(rfi_sources))
    exempt_kw = (;)
    if smooth_exempt_below_el > 0 && sflux > 0
        exempt = smooth_exempt_by_elevation(get_name.(sources), [elev[get_name(s)] for s in sources],
                                            min_elevation, smooth_exempt_below_el)
        exempt_kw = (; smooth_exempt_names=exempt)
        @printf("PEEL_SMOOTH_EXEMPT below_el_deg=%g names=\"%s\"\n", smooth_exempt_below_el, join(exempt, ","))
    end
    report = (name, it; converged, niters, finite, max_gain, p_with, p_old, p_new, action) -> @printf(
        "PEEL_SOLVE pass=%s name=\"%s\" elevation_deg=%.2f iteration=%d/%d converged=%s niters=%d finite=%s max_gain=%.6g accepted=%s action=%s p_ratio_new=%.4g p_ratio_old=%.4g\n",
        name in rfi_names ? "rfi" : "sky", name, elev[name], it, piter, converged, niters, finite,
        max_gain, action == :accept, action, p_new / p_with, p_old / p_with)
    r = zest_sky_rfi!(vis, meta, mjd, sources, rfi_sources;
                      rfi_mode=rfi_joint ? :joint : :after, solver=solver, maxiter=mi, tolerance=tol,
                      minuvw=muvw, peeliter=piter, max_gain=max_gain, min_elevation=min_elevation,
                      flag_zeros=flag_zero, report=report, smooth_flux=sflux, rfi_smooth_flux=rflux,
                      smooth_exempt_flux=smooth_exempt_flux > 0 ? smooth_exempt_flux : Inf,
                      restore=restore_sources, restore_below_mhz=restore_below_mhz,
                      restore_offset_deg=restore_offset_deg, precision=precision,
                      initial_gains=initial_gains, exempt_kw...)
    @printf("PEEL_SUMMARY pass=all selected=%d accepted=%d skipped=%d flagged_zeros=%d action=sky_rfi\n",
            length(sources) - length(r.skipped) + length(rfi_sources), length(r.peeled),
            length(r.skipped), r.flagged_zeros)
    for (name, (C, d)) in r.restored
        @printf("PEEL_RESTORE name=\"%s\" I_jy=%.2f V_jy=%.2f offset_l_arcmin=%.2f offset_m_arcmin=%.2f\n",
                name, real(sum(C[1]) + sum(C[4])) / length(C[1]),
                real(-1im * (sum(C[2]) - sum(C[3]))) / length(C[1]),
                rad2deg(d[1]) * 60, rad2deg(d[2]) * 60)
    end
    warm_start ? r.gains : nothing
end
prev_gains = nothing

# --- gsi v3 shared-memory slots (src/gpu_subband_imaging/v3/slots.py, design 3.2) -------------
# A slot holds one averaged snapshot (all MS main-table rows in MS order).  read_slot and
# write_slot! do exactly what read_ms_fast / write_ms_fast! (TTCalX src/gpu/ms_io.jl) do with
# the arrays casacore returns, so peeling a slot gives the bytes v2 writes into the MS.
# Channels, antenna positions and the phase centre come from the slot (prep took them from
# the MS subtables), the MJD from its TIME rows as ms_mjd.
const SLOT_MAGIC = Vector{UInt8}("GSISLOT1")
const SLOT_JSON_OFFSET = 64
const SLOT_STATE_FREE = UInt32(0)       # contents not valid
const SLOT_STATE_PREPPED = UInt32(1)
const SLOT_STATE_OFFSET = 12
const SLOT_STATE_PEELED = UInt32(2)
const SLOT_TYPES = Dict("<c8" => ComplexF32, "|b1" => Bool, "<i4" => Int32, "<f8" => Float64,
                        "<f4" => Float32)

function slot_header(io::IOStream)
    seekstart(io)
    read(io, 8) == SLOT_MAGIC || error("not a GSISLOT1 slot")
    n = ltoh(read(io, UInt32))
    state = ltoh(read(io, UInt32))
    seek(io, SLOT_JSON_OFFSET)
    h = JSON.parse(String(read(io, n)))
    h["state"] = Int(state)
    h
end

"""Mmap of one C-order slot array as Array{T,N}, dims reversed (numpy (nrow, nchan, 4) ->
(4, nchan, nrow)).  T and N are checked against the slot JSON, so the result is concretely typed."""
function slot_array(io::IOStream, h, name, ::Type{T}, ::Val{N}) where {T,N}
    e = get(h["arrays"], name, nothing)
    e === nothing && error("slot has no array $name")
    get(SLOT_TYPES, e["dtype"], nothing) === T || error("slot array $name: dtype $(e["dtype"]), expected $T")
    length(e["shape"]) == N || error("slot array $name: shape $(e["shape"]), expected $N dims")
    dims = ntuple(i -> Int(e["shape"][N + 1 - i]), N)
    Mmap.mmap(io, Array{T,N}, dims, Int64(e["offset"]); grow=false)::Array{T,N}
end

"""read_ms_fast's gather: cross rows -> baseline-major ComplexF64, flags OR'd over the pols."""
function slot_gather!(xx, xy, yx, yy, flags, data::Array{ComplexF32,3}, flag::Array{Bool,3},
                      cross::Vector{Int})
    @inbounds for f in axes(xx, 2), α in eachindex(cross)
        r = cross[α]
        xx[α, f] = data[1, f, r]; xy[α, f] = data[2, f, r]
        yx[α, f] = data[3, f, r]; yy[α, f] = data[4, f, r]
        flags[α, f] = flag[1, f, r] | flag[2, f, r] | flag[3, f, r] | flag[4, f, r]
    end
end

"""write_ms_fast!'s scatter: cross rows <- ComplexF32.(vis), FLAG of all pols <- baseline flag."""
function slot_scatter!(data::Array{ComplexF32,3}, flag::Array{Bool,3}, a1::Vector{Int32},
                       a2::Vector{Int32}, index::Matrix{Int}, xx, xy, yx, yy, fl)
    @inbounds for r in eachindex(a1)
        a1[r] == a2[r] && continue
        b = index[min(a1[r], a2[r]) + 1, max(a1[r], a2[r]) + 1]
        b == 0 && continue
        for f in axes(data, 2)
            data[1, f, r] = ComplexF32(xx[b, f]); data[2, f, r] = ComplexF32(xy[b, f])
            data[3, f, r] = ComplexF32(yx[b, f]); data[4, f, r] = ComplexF32(yy[b, f])
            g = fl[b, f]
            flag[1, f, r] = g; flag[2, f, r] = g; flag[3, f, r] = g; flag[4, f, r] = g
        end
    end
end

unmap!(arrays...) = foreach(finalize, arrays)    # munmap now (nothing refers to them any more)

"""Cross rows as read_ms_fast's _cross_rows: (cross row indices, lo, hi), 0-based antennas."""
function slot_cross_rows(a1, a2)
    cross = findall(a1 .!= a2)
    lo = Int.(min.(a1[cross], a2[cross]))
    hi = Int.(max.(a1[cross], a2[cross]))
    length(unique(lo .* 1000003 .+ hi)) == length(cross) ||
        error("slot has several rows per baseline (read_ms_fast would fall back; not supported)")
    cross, lo, hi
end

"""
    read_slot(path; gpu=true) -> (vis, cal, meta, baseline_dict, Nrows, mjd)

`read_ms_fast` on a slot: baselines in order of first appearance as (min, max) antenna
pairs, autocorrelations dropped, flags OR'd over the polarizations, the UVW of the
baseline's row, identity calibration, ComplexF32 -> ComplexF64.
"""
function read_slot(path::String; gpu::Bool=true)
    open(path, "r") do io
        h = slot_header(io)
        h["state"] == SLOT_STATE_PREPPED ||
            error("slot state is $(h["state"]), expected PREPPED ($(Int(SLOT_STATE_PREPPED)))")
        Int(h["ncorr"]) == 4 || error("slot has $(h["ncorr"]) correlations, expected 4")
        maps = Any[]           # every mapping is unmapped below, also when a later one fails
        try
            m(name, T, n) = (A = slot_array(io, h, name, T, Val(n)); push!(maps, A); A)
            a1, a2 = m("antenna1", Int32, 1), m("antenna2", Int32, 1)
            data, flag = m("data", ComplexF32, 3), m("flag", Bool, 3)        # (4, Nf, Nrows)
            uvw_rows, time_rows = m("uvw", Float64, 2), m("time", Float64, 1)
            freqs, ant_pos = m("meta.freqs", Float64, 1), m("meta.ant_pos", Float64, 2)
            pd = m("meta.phase_dir", Float64, 1)
            Nrows = length(a1)
            size(data, 1) == 4 && size(data, 3) == Nrows && length(a2) == Nrows &&
                size(flag) == size(data) && size(uvw_rows) == (3, Nrows) &&
                length(time_rows) == Nrows ||
                error("slot arrays disagree: antenna1 $(length(a1)), antenna2 $(length(a2)), " *
                      "data $(size(data)), flag $(size(flag)), uvw $(size(uvw_rows)), time $(length(time_rows))")
            size(ant_pos, 1) == 3 || error("slot ant_pos must be (nant, 3)")
            length(pd) >= 2 || error("slot phase_dir must be (2,)")
            cross, lo0, hi0 = slot_cross_rows(a1, a2)
            lo, hi = lo0 .+ 1, hi0 .+ 1
            Nbase, Nf = length(cross), size(data, 2)
            xx, xy, yx, yy = (Matrix{ComplexF64}(undef, Nbase, Nf) for _ in 1:4)
            flags = Matrix{Bool}(undef, Nbase, Nf)
            slot_gather!(xx, xy, yx, yy, flags, data, flag, cross)
            uvw = Float64.(uvw_rows[:, cross])
            channels = Float64.(vec(freqs))
            length(channels) == Nf || error("slot has $(length(channels)) freqs for $Nf channels")
            antenna_positions = Float64.(ant_pos)
            ra, dec = Float64(pd[1]), Float64(pd[2])
            mjd = (minimum(time_rows) + maximum(time_rows)) / 2 / 86400      # ms_mjd
            Na = size(antenna_positions, 2)
            maximum(hi) <= Na || error("slot antenna index $(maximum(hi)) > $Na antennas")
            baselines = Int32.(permutedims(hcat(lo, hi)))
            baseline_dict = Dict{Tuple{Int,Int}, Int}((lo[α], hi[α]) => α for α in 1:Nbase)
            up = gpu && CUDA.functional() ? CuArray : identity
            vis = GPUVisibilities(up(xx), up(xy), up(yx), up(yy), up(flags))
            meta = GPUMetadata(up(antenna_positions), up(baselines), up(channels), up([0.0, 0.0, 1.0]),
                               ra, dec, up(uvw))
            cal = GPUCalibration(up(ones(ComplexF64, Na, Nf)), up(zeros(ComplexF64, Na, Nf)),
                                 up(zeros(ComplexF64, Na, Nf)), up(ones(ComplexF64, Na, Nf)),
                                 up(falses(Na, Nf)), true)
            vis, cal, meta, baseline_dict, Nrows, mjd
        finally
            unmap!(maps...)
        end
    end
end

"""
    write_slot!(path, vis, baseline_dict, Nrows)

`write_ms_fast!` on a slot: the cross rows' DATA = ComplexF32.(vis), FLAG of all four
polarizations = the baseline flag, autocorrelations untouched.  The state is FREE (not
valid) while the rows are written and PEELED after, so a daemon killed mid-write never
leaves a slot that looks prepped or peeled.
"""
function write_slot!(path::String, vis::GPUVisibilities, baseline_dict::Dict, Nrows::Int)
    xx, xy, yx, yy = Array(vis.xx), Array(vis.xy), Array(vis.yx), Array(vis.yy)
    fl = Array(vis.flags)
    open(path, "r+") do io
        h = slot_header(io)
        state_offset = Int(h["state_offset"])
        maps = Any[]
        try
            m(name, T, n) = (A = slot_array(io, h, name, T, Val(n)); push!(maps, A); A)
            a1, a2 = m("antenna1", Int32, 1), m("antenna2", Int32, 1)
            data, flag = m("data", ComplexF32, 3), m("flag", Bool, 3)
            length(a1) == Nrows && length(a2) == Nrows && size(data, 3) == Nrows &&
                size(flag) == size(data) && size(data, 1) == 4 &&
                size(data, 2) == size(xx, 2) ||
                error("slot has $(length(a1)) rows / data $(size(data)); the read had $Nrows rows, $(size(xx, 2)) channels")
            seek(io, state_offset); write(io, htol(SLOT_STATE_FREE)); flush(io)
            Na = maximum(maximum(k) for k in keys(baseline_dict))
            n = max(Int(maximum(a1)), Int(maximum(a2))) + 1
            index = zeros(Int, max(Na, n), max(Na, n))   # [a1, a2] (1-based) -> baseline
            for ((p, q), α) in baseline_dict
                index[p, q] = α
            end
            slot_scatter!(data, flag, a1, a2, index, xx, xy, yx, yy, fl)
        finally
            unmap!(maps...)
        end
        seek(io, state_offset)
        write(io, htol(SLOT_STATE_PEELED))
        flush(io)
    end
    nothing
end

"""ZEST_TEST_IO=perturb: a deterministic change at full precision (no peeling)."""
function perturb_for_io_test!(vis)
    for f in (:xx, :xy, :yx, :yy)
        getfield(vis, f) .*= 1.0001 * cis(0.3)
    end
    vis.flags[1:7:end, :] .= .!vis.flags[1:7:end, :]
    println("PEEL_SUMMARY pass=all selected=0 accepted=0 action=test_io_perturb")
end

"""Peel one snapshot in place with the configured engine (MS and slot commands alike)."""
function peel_snapshot!(vis, meta, mjd)
    if test_io != Symbol("")
        test_io == :perturb ? perturb_for_io_test!(vis) :
            println("PEEL_SUMMARY pass=all selected=0 accepted=0 action=test_io_identity")
        return
    end
    if geometric
        rms_w, offset = geometric_uvw!(meta, mjd)
        @printf("UVW_GEOMETRIC rms_w_vs_ms=%.4f phase_centre_offset_deg=%.4f\n", rms_w, offset)
    end
    if sky_rfi
        global prev_gains = run_sky_rfi!(vis, meta, mjd, prev_gains)
    else
    flag_zero && @printf("FLAG_ZERO flagged=%d\n", flag_zero_visibilities!(vis))
    function run_pass(catalog, name)
        before = backstop ? deepcopy_gpu(vis) : nothing
        p0 = backstop ? TTCalX.residual_power(vis) : 0.0
        safe_zest!(vis, meta, catalog, mjd; pass=name)
        if backstop && TTCalX.residual_power(vis) > p0
            @printf("PEEL_BACKSTOP pass=%s reverted=true ratio=%.4g\n", name, TTCalX.residual_power(vis) / p0)
            for f in (:xx, :xy, :yx, :yy); getfield(vis, f) .= getfield(before, f); end
        end
    end
    if rfi_joint && !isempty(rfi_sources)
        run_pass(vcat(sources, rfi_sources), "sky")
    else
        run_pass(sources, "sky")
        isempty(rfi_sources) || run_pass(rfi_sources, "rfi")
    end
    end
end

read_ms = fast_io ? read_ms_fast : read_ms_to_gpu
write_ms! = fast_io ? write_ms_fast! : write_gpu_to_ms!

for line in eachline(stdin)
    cmd = String(strip(line))
    isempty(cmd) && continue
    cmd == "QUIT" && break
    if cmd == "RESET_WARM" || startswith(cmd, "RESET_WARM ")
        global prev_gains = nothing
        println("RESET_WARM_OK id=", strip(cmd[11:end])); flush(stdout)
        continue
    end
    shm = startswith(cmd, "SHM ")
    parts = split(cmd)
    ms = shm ? String(parts[2]) : cmd
    rid = shm && length(parts) >= 3 ? " id=" * parts[3] : ""
    t0 = time()
    try
        if shm
            col == "DATA" || error("SHM slots carry DATA only (ZEST_COLUMN=$col)")
            t_read = @elapsed (vis, cal, meta, bl, nrows, slot_mjd) = read_slot(ms; gpu=use_gpu)
        else
            t_read = @elapsed (vis, cal, meta, bl, nrows) = read_ms(ms; gpu=use_gpu, column=col)
        end
        t_peel = @elapsed begin
            mjd = (exact_model || geometric) ? (shm ? slot_mjd : ms_mjd(ms)) : NaN
            peel_snapshot!(vis, meta, mjd)
        end
        t_write = @elapsed (shm ? write_slot!(ms, vis, bl, nrows) :
                                  write_ms!(ms, vis, bl, nrows; column=col))
        @printf("ZEST_TIMING read=%.2f peel=%.2f write=%.2f\n", t_read, t_peel, t_write)
        @printf("ZESTED %s %.2f%s\n", ms, time() - t0, rid)
    catch e
        global prev_gains = nothing
        # one line, so a multi-line message (e.g. a PyCall traceback) cannot be read as the
        # next command's reply
        msg = replace(sprint(showerror, e), r"\s*\n\s*" => " | ")
        println("FAILED $ms$rid $msg")
    end
    flush(stdout)
    use_gpu && CUDA.reclaim()
end
println("DAEMON_EXIT"); flush(stdout)
