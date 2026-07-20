# Keep TTCalX and CUDA loaded while peeling MS paths read from stdin. Emits:
#   ZESTED <path> <secs>   or   FAILED <path> <msg>
# Settings use ZEST_COLUMN/MAXITER/TOLERANCE/MINUVW/PEELITER.
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
using CUDA
using Printf

sources_path = ARGS[1]
col   = get(ENV, "ZEST_COLUMN", "DATA")
mi    = parse(Int,     get(ENV, "ZEST_MAXITER",  "30"))
tol   = parse(Float64, get(ENV, "ZEST_TOLERANCE","1e-4"))
muvw  = parse(Float64, get(ENV, "ZEST_MINUVW",   "10.0"))
piter = parse(Int,     get(ENV, "ZEST_PEELITER", "3"))

init_pycasacore() || error("python-casacore not available")
sources = read_gpu_sources(sources_path)
println("DAEMON_READY sources=$(length(sources)) device=$(CUDA.name(CUDA.device()))")
flush(stdout)

for line in eachline(stdin)
    ms = String(strip(line))
    isempty(ms) && continue
    ms == "QUIT" && break
    t0 = time()
    try
        vis, cal, meta, bl, nrows = read_ms_to_gpu(ms; gpu=true, column=col)
        zest_gpu!(vis, meta, sources; maxiter=mi, tolerance=tol, minuvw=muvw,
                  peeliter=piter, phase_center_ra=meta.phase_center_ra,
                  phase_center_dec=meta.phase_center_dec, lst=0.0)
        write_gpu_to_ms!(ms, vis, bl, nrows; column=col)
        @printf("ZESTED %s %.2f\n", ms, time() - t0)
    catch e
        println("FAILED $ms $(sprint(showerror, e))")
    end
    flush(stdout)
    CUDA.reclaim()
end
println("DAEMON_EXIT"); flush(stdout)
