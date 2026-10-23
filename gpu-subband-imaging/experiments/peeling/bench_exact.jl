# Timing and equality of the CPU-exact model and the host (CPU) backend.
#   julia -t N --project=$TTCALX_EXP bench_exact.jl SOURCES RFI MS [MS ...]
# MS files are only read.  Prints BENCHX lines.
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf

init_pycasacore() || error("python-casacore not available")
sky = read_gpu_sources(ARGS[1])
rfi = read_gpu_sources(ARGS[2])
set_verbosity(:quiet)
@printf("BENCHX threads=%d device=%s\n", Threads.nthreads(), CUDA.name(CUDA.device()))

function timed(f, n=3)
    f()                       # compile / warm up
    ts = [(@elapsed f()) for _ in 1:n]
    minimum(ts)
end
relerr(a, b) = maximum(abs, Array(a) .- Array(b)) / max(maximum(abs, Array(b)), eps())

for ms in ARGS[3:end]
    name = basename(dirname(ms))
    t_read = time(); r = read_ms_to_gpu(ms; gpu=true); t_read = time() - t_read
    vis, meta = r[1], r[3]
    t_rh = time(); r = read_ms_to_gpu(ms; gpu=false); t_rh = time() - t_rh
    hvis, hmeta = r[1], r[3]
    mjd = ms_mjd(ms)
    geometric_uvw!(meta, mjd); geometric_uvw!(hmeta, mjd)
    @printf("BENCHX %s read_gpu=%.2f read_host=%.2f Nbase=%d Nfreq=%d Nant=%d\n",
            name, t_read, t_rh, meta.Nbase, meta.Nfreq, meta.Nant)
    for src in sky
        ref = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=true)
        g = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=true)
        h = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        t_ref = timed(() -> (fill!(ref.xx, 0); fill!(ref.xy, 0); fill!(ref.yx, 0); fill!(ref.yy, 0);
                             genvis_exact_reference!(ref, meta, src, mjd)), 1)
        t_gpu = timed(() -> (fill!(g.xx, 0); fill!(g.xy, 0); fill!(g.yx, 0); fill!(g.yy, 0);
                             genvis_exact!(g, meta, src, mjd)))
        t_host = timed(() -> (fill!(h.xx, 0); fill!(h.xy, 0); fill!(h.yx, 0); fill!(h.yy, 0);
                              genvis_exact!(h, hmeta, src, mjd)))
        e_gpu = maximum(relerr(getfield(g, f), getfield(ref, f)) for f in (:xx, :xy, :yx, :yy))
        e_host = maximum(relerr(getfield(h, f), getfield(ref, f)) for f in (:xx, :xy, :yx, :yy))
        @printf("BENCHX %s genvis src=\"%s\" ref=%.3f gpu=%.4f host=%.4f relerr_gpu=%.2e relerr_host=%.2e\n",
                name, get_name(src), t_ref, t_gpu, t_host, e_gpu, e_host)
    end
    # one rk4 zest solve (Cas A model, data as measured), GPU vs host
    src = sky[findfirst(s -> get_name(s) == "Cas A", sky)]
    for (label, v, m) in (("gpu", vis, meta), ("host", hvis, hmeta))
        coh = GPUVisibilities(m.Nbase, m.Nfreq; gpu=label == "gpu")
        genvis_exact!(coh, m, src, mjd)
        coh.flags .= v.flags
        local c
        t = timed(() -> begin
            c = calibration_type(GPUZestingSource(src), m.Nant, m.Nfreq; gpu=label == "gpu")
            gpu_stefcal_exp!(c, v, coh, m; maxiter=5, tolerance=1e-4, minuvw=5.0, solver=:rk4)
        end, 2)
        tc = timed(() -> (x = deepcopy_gpu(coh); gpu_corrupt!(x, c, m); gpu_subsrc!(deepcopy_gpu(v), x)), 2)
        @printf("BENCHX %s solve_%s=%.3f corrupt_sub_%s=%.3f\n", name, label, t, label, tc)
        label == "gpu" ? (global cg = c) : (global ch = c)
    end
    e = maximum(relerr(getfield(ch, f), getfield(cg, f)) for f in (:xx, :xy, :yx, :yy))
    @printf("BENCHX %s gains relerr_host_vs_gpu=%.2e\n", name, e)
end
println("BENCHX_DONE")
