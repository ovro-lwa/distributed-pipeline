# Where the time of one GPU zest solve goes (MS only read).
#   julia --project=$TTCALX_EXP prof_solve.jl SOURCES MS
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf
init_pycasacore()
set_verbosity(:quiet)
sky = read_gpu_sources(ARGS[1])
ms = ARGS[2]
r = read_ms_to_gpu(ms; gpu=true); vis, meta = r[1], r[3]
mjd = ms_mjd(ms); geometric_uvw!(meta, mjd)
src = sky[findfirst(s -> get_name(s) == "Cas A", sky)]
coh = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=true)
genvis_exact!(coh, meta, src, mjd); coh.flags .= vis.flags
best(f, n=3) = (f(); minimum(@elapsed(f()) for _ in 1:n))
sq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=true)
@printf("PROF makesquare=%.4f\n", best(() -> TTCalX.gpu_makesquare!(sq, vis, meta)))
@printf("PROF flag_short_baselines=%.4f\n", best(() -> TTCalX.flag_short_baselines!(sq, meta, 5.0)))
isdefined(TTCalX, :flag_short_baselines_gpu!) &&
    @printf("PROF flag_short_baselines_gpu=%.4f\n", best(() -> TTCalX.flag_short_baselines_gpu!(sq, meta, 5.0)))
cal = calibration_type(GPUZestingSource(src), meta.Nant, meta.Nfreq; gpu=true)
k = ntuple(_ -> CUDA.zeros(ComplexF64, meta.Nant, meta.Nfreq), 4)
g = (cal.xx, cal.xy, cal.yx, cal.yy)
sq2 = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=true)
TTCalX.gpu_makesquare!(sq, vis, meta); TTCalX.flag_short_baselines!(sq, meta, 5.0)
TTCalX.gpu_makesquare!(sq2, vis, meta)
isdefined(TTCalX, :flag_short_baselines_gpu!) && (TTCalX.flag_short_baselines_gpu!(sq2, meta, 5.0);
    @printf("PROF flags_identical=%s\n", all(Array(sq.xx) .== Array(sq2.xx)) && all(Array(sq.yx) .== Array(sq2.yx))))
gc = deepcopy_gpu(cal); gc.xx .+= 0.01 .* CUDA.randn(ComplexF64, size(gc.xx)); gc.xy .+= 0.01 .* CUDA.randn(ComplexF64, size(gc.xy))
k1 = ntuple(_ -> CUDA.zeros(ComplexF64, meta.Nant, meta.Nfreq), 4); k2 = ntuple(_ -> CUDA.zeros(ComplexF64, meta.Nant, meta.Nfreq), 4)
TTCalX.gpu_stefcal_step_full!(k1..., gc, sq, sq, meta)
TTCalX._stefcal_steps!(k2, (gc.xx, gc.xy, gc.yx, gc.yy), gc, sq, sq, meta, nothing)
@printf("PROF step_maxdiff_vs_original=%.3e\n", maximum(maximum(abs, Array(k1[c]) .- Array(k2[c])) for c in 1:4))
@printf("PROF one_step_original=%.5f\n", best(() -> TTCalX.gpu_stefcal_step_full!(k1..., gc, sq, sq, meta)))
@printf("PROF one_step=%.5f\n", best(() -> TTCalX._stefcal_steps!(k, g, cal, sq, sq, meta, nothing)))
@printf("PROF solve_rk4=%.4f\n", best(() -> begin
    c = calibration_type(GPUZestingSource(src), meta.Nant, meta.Nfreq; gpu=true)
    gpu_stefcal_exp!(c, vis, coh, meta; maxiter=5, tolerance=1e-4, minuvw=5.0, solver=:rk4)
end))
@printf("PROF corrupt=%.4f residual_power=%.4f\n",
        best(() -> gpu_corrupt!(deepcopy_gpu(coh), cal, meta)),
        best(() -> sum(abs2, vis.xx .* .!vis.flags) + sum(abs2, vis.yy .* .!vis.flags)))
