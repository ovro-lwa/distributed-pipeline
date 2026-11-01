# julia tests/julia/test_zest_smooth.jl  (no TTCalX needed; run by tests/test_zest_smooth_julia.py)
using Test
include(joinpath(@__DIR__, "..", "..", "workers", "zest_smooth.jl"))

@testset "smooth_exempt_by_elevation" begin
    names = ["Cas A", "Tau A", "Hya A", "Vir A", "RFI"]
    el = [29.999, 30.0, 12.0, -3.0, NaN]
    @test smooth_exempt_by_elevation(names, el, 0.0, 30.0) == ["Cas A", "Hya A"]   # 30.0 not < 30
    @test smooth_exempt_by_elevation(names, el, 12.0, 30.0) == ["Cas A", "Hya A"]  # min_el inclusive
    @test smooth_exempt_by_elevation(names, el, 12.5, 30.0) == ["Cas A"]           # below min_el out
    @test smooth_exempt_by_elevation(names, el, -90.0, 90.0) == ["Cas A", "Tau A", "Hya A", "Vir A"]
    @test isempty(smooth_exempt_by_elevation(names, el, 0.0, 0.0))                 # off
    @test smooth_exempt_by_elevation(String[], Float64[], 0.0, 30.0) == String[]
    @test eltype(smooth_exempt_by_elevation(SubString.(["a b"], 1, 1), [5.0], 0.0, 30.0)) == String
end
