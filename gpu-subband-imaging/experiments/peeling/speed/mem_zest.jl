# In-memory zest (no MS written): times sky-then-RFI peeling with the
# experimental daemon's code on GPU or host threads, dumps I = (XX+YY)/2.
#   MEM_MS=a.ms,b.ms MEM_OUT=dir MEM_TAG=tag julia -t N --project=$TTCALX mem_zest.jl SKY < /dev/null
include(ENV["DAEMON"])           # stdin is empty: definitions only, the MS loop exits at once
function run_one(ms)
    r = read_ms_to_gpu(ms; gpu=use_gpu, column=col)
    vis, meta = r[1], r[3]
    mjd = ms_mjd(ms)
    geometric_uvw!(meta, mjd)
    t = @elapsed begin
        safe_zest_gpu!(vis, meta, sources; ms=ms, mjd=mjd)
        safe_zest_gpu!(vis, meta, rfi_sources; pass="rfi", maxiter=rfi_mi, tolerance=rfi_tol,
                       minuvw=rfi_muvw, peeliter=rfi_piter, ms=ms, mjd=mjd)
    end
    vis, t
end
mslist = String.(split(ENV["MEM_MS"], ","))
run_one(mslist[1])               # warm-up (compilation)
for ms in mslist
    vis, t = run_one(ms)
    snap = basename(dirname(ms))
    @printf("MEMTIME %s %s threads=%d peel=%.3f\n", ENV["MEM_TAG"], snap, Threads.nthreads(), t)
    I = ComplexF32.((Array(vis.xx) .+ Array(vis.yy)) ./ 2)
    open(joinpath(ENV["MEM_OUT"], "$(ENV["MEM_TAG"])_$(snap).bin"), "w") do io
        write(io, Int64(size(I, 1)), Int64(size(I, 2))); write(io, I); write(io, UInt8.(Array(vis.flags)))
    end
end
println("MEM_DONE")
