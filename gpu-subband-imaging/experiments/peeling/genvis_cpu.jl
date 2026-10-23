# CPU TTCal (Julia 0.6) model visibilities -> DATA of an MS copy.
#   julia genvis_cpu.jl MS SOURCES_JSON
using TTCal, CasaCore.Tables
ms = Tables.open(ARGS[1], write=true)
meta = TTCal.Metadata(ms)
sky = TTCal.readsky(ARGS[2])
model = TTCal.genvis(meta, TTCal.ConstantBeam(), sky, polarization=TTCal.Full)
ms["DATA"] = convert.(Complex64, TTCal.ttcal_to_array(model))
Tables.close(ms)
println("GENVIS_CPU done ", ARGS[1])
