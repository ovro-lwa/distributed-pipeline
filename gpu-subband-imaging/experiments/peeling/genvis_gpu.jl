# TTCalX model visibilities -> DATA of an MS copy.
#   TTCALX_DIR=... julia --project=$TTCALX_DIR genvis_gpu.jl MS SOURCES_JSON [as_is|auto|geom|geom_neg|exact] [uvw|none] [x,y,z]
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX, CUDA
init_pycasacore()
ms, path = ARGS[1], ARGS[2]
conv = length(ARGS) >= 3 ? ARGS[3] : "as_is"
geometry = Symbol(length(ARGS) >= 4 ? ARGS[4] : "uvw")
position = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : nothing
vis, cal, meta, bl, nrows = read_ms_to_gpu(ms; gpu=true, column="DATA")
for v in (vis.xx, vis.xy, vis.yx, vis.yy); fill!(v, 0); end
fill!(vis.flags, false)
if conv == "auto"
    s0, sign, rms = uvw_geometry(meta)
    println("UVW_GEOMETRY s0=$s0 sign=$sign rms=$rms")
    apply_uvw_convention!(meta)
elseif startswith(conv, "geom")
    rms, off = geometric_uvw!(meta, ms_mjd(ms); sign=conv == "geom" ? 1 : -1)
    println("UVW_GEOMETRIC rms_w_vs_ms=$rms phase_centre_offset_deg=$off")
end
mjd = ms_mjd(ms)
for s in read_gpu_sources(path)
    if conv == "exact" && !TTCalX.is_rfi(s)
        genvis_exact!(vis, meta, s, mjd)
    elseif isdefined(TTCalX, :is_rfi) && TTCalX.is_rfi(s)
        gpu_genvis!(vis, meta, s, meta.phase_center_ra, meta.phase_center_dec, 0.0;
                    geometry=geometry, position=position)
    else
        gpu_genvis!(vis, meta, s, meta.phase_center_ra, meta.phase_center_dec, 0.0)
    end
end
write_gpu_to_ms!(ms, vis, bl, nrows; column="DATA")
println("GENVIS_GPU done ", ms)
