# read_ms_to_gpu vs read_ms_fast: identical outputs and timing (MS only read).
push!(LOAD_PATH, joinpath(ENV["TTCALX_DIR"], "src"))
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX, CUDA, Printf
init_pycasacore(); set_verbosity(:quiet)
for ms in ARGS
    for gpu in (true, false)
        a = read_ms_to_gpu(ms; gpu=gpu); b = read_ms_fast(ms; gpu=gpu)
        ta = minimum(@elapsed(read_ms_to_gpu(ms; gpu=gpu)) for _ in 1:3)
        tb = minimum(@elapsed(read_ms_fast(ms; gpu=gpu)) for _ in 1:3)
        same(x, y) = (typeof(x) == typeof(y) || println("TYPE ", typeof(x), " vs ", typeof(y))) && Array(x) == Array(y)
        va, ca, ma, da, na = a; vb, cb, mb, db, nb = b
        checks = Dict("vis" => all(same(getfield(va, f), getfield(vb, f)) for f in fieldnames(typeof(va))),
                      "cal" => all(f == :is_diagonal ? getfield(ca, f) == getfield(cb, f) : same(getfield(ca, f), getfield(cb, f)) for f in fieldnames(typeof(ca))),
                      "meta" => all(begin x = getfield(ma, f); y = getfield(mb, f); x isa AbstractArray ? same(x, y) : x == y end for f in fieldnames(typeof(ma))),
                      "dict" => da == db, "nrows" => na == nb)
        @printf("READTEST %s gpu=%s original=%.2f fast=%.2f identical=%s %s\n", basename(dirname(ms)), gpu, ta, tb,
                all(values(checks)), join(["$k=$v" for (k, v) in checks], " "))
    end
end
