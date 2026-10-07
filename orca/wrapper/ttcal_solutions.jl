# Julia 0.6 adapter for the installed TTCal Dataset/Calibration API.
# Same solver and MS write as TTCal.run_zest; no installed packages are modified.
# Use a raw exchange format because these environments do not include NPZ.jl.
import TTCal
import JSON
using CasaCore.Tables

function zest_and_export(msname, sourcesfile, beamname, minuvw, maxiter, tolerance, outdir)
    ms = Tables.open(msname, write=true)
    try
        column = Tables.column_exists(ms, "CORRECTED_DATA") ? "CORRECTED_DATA" : "DATA"
        dataset = TTCal.Dataset(ms, column=column, polarization=TTCal.Full)
        sky = TTCal.readsky(sourcesfile)
        beam = TTCal.select_beam(beamname)
        # peel! filters below-horizon sources internally. Preserve that exact mapping.
        frame = TTCal.ReferenceFrame(dataset.metadata)
        selected = find(s -> TTCal.isabovehorizon(frame, s), sky.sources)
        calibrations = TTCal.peel!(dataset, beam, sky,
            peeliter=3, maxiter=maxiter, tolerance=tolerance, minuvw=minuvw,
            collapse_frequency=false)
        length(calibrations) == length(selected) || error("Source/solution count mismatch")
        nant = TTCal.Nant(dataset)
        nfreq = TTCal.Nfreq(dataset)
        ntime = TTCal.Ntime(dataset)
        # Both supported installations read one integration per MS.
        ntime == 1 || error("Unsupported TTCal time metadata; expected one integration")
        gains = zeros(Complex128, length(calibrations), 4, nant, nfreq, ntime)
        for source = 1:length(calibrations), t = 1:ntime, f = 1:nfreq, ant = 1:nant
            jones = calibrations[source][f, t][ant]
            gains[source, 1, ant, f, t] = jones.xx
            gains[source, 2, ant, f, t] = jones.xy
            gains[source, 3, ant, f, t] = jones.yx
            gains[source, 4, ant, f, t] = jones.yy
        end
        open(joinpath(outdir, "gains.bin"), "w") do io
            # Explicit little endian, including on hosts with a different byte order.
            write(io, htol.(reinterpret(UInt64, vec(gains))))
        end
        metadata = Dict(
            "shape" => collect(size(gains)),
            "source_indices" => selected .- 1,
            "source_names" => [sky.sources[i].name for i in selected],
            "frequencies_hz" => TTCal.ustrip.(dataset.metadata.frequencies),
            "times_mjd_seconds" => [ms["TIME", 1]],
            "column" => column)
        open(joinpath(outdir, "metadata.json"), "w") do io
            JSON.print(io, metadata)
        end
        # Preserve CLI behavior: update data, leave FLAG/FLAG_ROW unchanged.
        ms[column] = convert.(Complex64, TTCal.ttcal_to_array(dataset))
    finally
        Tables.close(ms)
    end
end

zest_and_export(ARGS[1], ARGS[2], ARGS[3], parse(Float64, ARGS[4]),
                parse(Int, ARGS[5]), parse(Float64, ARGS[6]), ARGS[7])
