include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX
init_pycasacore()
s = phase_center_itrf(ARGS[1]); println("JL_S0 ", s[1], " ", s[2], " ", s[3])
println("JL_RFI ", join(wgs84_to_itrf(-118.62673, 37.53159, 2072.0), " "))
