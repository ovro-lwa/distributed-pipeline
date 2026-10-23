# Unit tests for the Tukey phase-prior update (no GPU needed).
#   TTCALX_DIR=... julia --project=$TTCALX_DIR test_regularize.jl
include(joinpath(ENV["TTCALX_DIR"], "src", "TTCalX.jl"))
using .TTCalX, LinearAlgebra, Random, Test
const R = TTCalX

cost(z, z0, s, st, wd, lam) = (P = R.tukey_prior(abs(angle(z)), st, wd); P <= 0 ? Inf : s * abs2(z - z0) - lam * log(P))

function brute(z0, s, st, wd, lam)
    best = (Inf, z0)
    r0 = abs(z0)
    for φ in range(-π, π; length=4001), r in range(0, 1.5r0; length=301)
        z = r * cis(φ)
        c = cost(z, z0, s, st, wd, lam)
        c < best[1] && (best = (c, z))
    end
    best
end

Random.seed!(1)
@testset "regularize_element" begin
    # inactive prior / inside the flat region -> unchanged
    @test R.regularize_element(1.0 + 0.1im, 5.0, Inf, 0.3, 1.0) == 1.0 + 0.1im
    @test R.regularize_element(cis(0.2), 5.0, 0.3, 0.3, 1.0) == cis(0.2)
    # beyond the taper end -> pulled inside |φ| < start + width
    z = R.regularize_element(cis(2.0), 1e6, 0.3, 0.3, 1.0)
    @test abs(angle(z)) < 0.6
    @test abs(angle(R.regularize_element(cis(-2.0), 1e6, 0.3, 0.3, 1.0)) + angle(z)) < 1e-12  # symmetric
    # matches brute force over random cases
    worst = 0.0
    for _ in 1:200
        z0 = (0.3 + rand()) * cis(2π * rand() - π)
        s = 10.0^(4rand() - 2); st = rand() * 1.5; wd = 0.05 + rand(); lam = 10.0^(2rand() - 1)
        z = R.regularize_element(z0, s, st, wd, lam)
        cb, _ = brute(z0, s, st, wd, lam)
        worst = max(worst, (cost(z, z0, s, st, wd, lam) - cb) / max(abs(cb), 1e-9))
    end
    @show worst
    @test worst < 1e-3
end

@testset "Schur reduction (full Jones column)" begin
    # minimise (x-x0)' D (x-x0) - lam log P(arg conj(x1)) over x in C^2 by brute
    # force on x1 (x2 at its conditional optimum) and compare with the kernel's math
    for _ in 1:20
        A = randn(ComplexF64, 2, 2); D = A' * A + 0.1I
        x0 = randn(ComplexF64, 2)
        st, wd, lam = 0.2, 0.4, 2.0
        s1 = real(D[1, 1]) - abs2(D[1, 2]) / real(D[2, 2])
        z = R.regularize_element(conj(x0[1]), s1, st, wd, lam)
        δ1 = conj(z) - x0[1]
        x = [x0[1] + δ1, x0[2] - D[2, 1] * δ1 / real(D[2, 2])]
        f(x) = (d = x - x0; P = R.tukey_prior(abs(angle(conj(x[1]))), st, wd); P <= 0 ? Inf : real(d' * D * d) - lam * log(P))
        best = Inf
        for φ in range(-π, π; length=721), r in range(0, 2abs(x0[1]) + 1; length=241)
            x1 = r * cis(φ); d1 = x1 - x0[1]
            best = min(best, f([x1, x0[2] - D[2, 1] * d1 / real(D[2, 2])]))
        end
        @test f(x) <= best + 1e-3 * abs(best) + 1e-9
    end
end

@testset "phase limits" begin
    pos = [-200.0 -100.0 0.0 100.0 1000.0; 0.0 0.0 0.0 0.0 0.0; 6.371e6 6.371e6 6.371e6 6.371e6 6.371e6]
    d = R.antenna_planar_distances(pos)
    @test d[4] ≈ 100.0 atol = 1e-6
    @test d[3] ≈ 0.0 atol = 1e-9
    k = 2π * 1000.0 * 30e6 / R.SPEED_OF_LIGHT
    @test k * sind(1.0) ≈ 2π * 1000 * 30e6 / 299792458 * sind(1.0)
end
println("TEST_REGULARIZE_DONE")
