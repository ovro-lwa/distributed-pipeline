# Helpers of workers/zest_daemon.jl that need no TTCalX (tested by tests/julia/test_zest_smooth.jl).

"""
    smooth_exempt_by_elevation(names, elevations_deg, min_elevation_deg, below_deg) -> Vector{String}

Names of the sky sources that keep per-channel gains under band-gain smoothing in this
snapshot: elevation in `[min_elevation_deg, below_deg)`.  Elevations are the daemon's
`elevation_deg` (asin(n) about the MS phase centre, the same as for
min_source_elevation_deg); NaN (RFI emitters) is never listed.
"""
smooth_exempt_by_elevation(names, elevations_deg, min_elevation_deg, below_deg) =
    String[String(n) for (n, e) in zip(names, elevations_deg)
           if !isnan(e) && min_elevation_deg <= e < below_deg]
