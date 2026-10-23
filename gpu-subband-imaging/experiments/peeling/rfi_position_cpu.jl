# Print CPU TTCal's ITRF position for the RFI emitter (Julia 0.6, ttcal_dev).
using TTCal, CasaCore.Measures
sky = TTCal.readsky(ARGS[1])
for s in sky.sources, shape in s.shapes
    p = shape.position
    println("CPU_RFI_POS input=", p)
    frame = ReferenceFrame()
    q = measure(frame, p, pos"ITRF")
    println("CPU_RFI_ITRF ", q.x, " ", q.y, " ", q.z)
end
