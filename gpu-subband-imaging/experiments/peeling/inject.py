"""Inject unmodelled point sources near a peel target into a copy of a prep MS.

    python inject.py PREP_MS OUT_MS --near CasA --offsets 1,2,4,8 --flux 300 [--json OUT]

Sources are placed at the given angular offsets (deg) from the target, at
position angles 0/90/180/270 deg (cycled), flat spectrum, Stokes I only.
Visibilities use the geometry verified against the data:
    V = S exp(-i k (r2 - r1)·(ŝ - ŝ0)),  ŝ, ŝ0 apparent ITRF directions.
DATA(out) = DATA(prep) + V.  The injected positions are written to --json so
the image analysis can measure how much of each source survives peeling.
"""
import argparse
import json
import shutil

import astropy.units as u
import numpy as np
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
from casacore.tables import table

TARGETS = {"CasA": SkyCoord("23h23m24s", "+58d48m54s"), "CygA": SkyCoord("19h59m28.36s", "+40d44m02.1s"),
           "VirA": SkyCoord("12h30m49.4s", "+12d23m28s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep"); ap.add_argument("out")
    ap.add_argument("--near", default="CasA")
    ap.add_argument("--offsets", default="1,2,4,8")
    ap.add_argument("--flux", type=float, default=300.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    shutil.copytree(a.prep, a.out)
    t = table(a.out, readonly=False, ack=False)
    a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2")
    time = t.getcell("TIME", 0)
    pos = table(a.out + "/ANTENNA", ack=False).getcol("POSITION")
    freq = table(a.out + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
    pd = table(a.out + "/FIELD", ack=False).getcol("PHASE_DIR")[0, 0]
    obstime = Time(time / 86400.0, format="mjd", scale="utc")
    itrs = ITRS(obstime=obstime)
    s0 = SkyCoord(pd[0] * u.rad, pd[1] * u.rad, frame="icrs").transform_to(itrs).cartesian.xyz.value
    b = pos[a2] - pos[a1]
    k = 2 * np.pi * freq / 299792458.0
    centre = TARGETS[a.near]
    offs = [float(x) for x in a.offsets.split(",")]
    model = np.zeros((len(a1), len(freq)), complex)
    placed = []
    for i, o in enumerate(offs):
        c = centre.directional_offset_by((90 * i) * u.deg, o * u.deg)
        s = c.transform_to(itrs).cartesian.xyz.value
        model += a.flux * np.exp(-1j * k[None, :] * (b @ (s - s0))[:, None])
        placed.append({"name": f"inj{o:g}", "ra": c.ra.deg, "dec": c.dec.deg, "offset_deg": o,
                       "pa_deg": 90 * i, "flux": a.flux})
    model[a1 == a2] = 0
    data = t.getcol("DATA")
    data[:, :, 0] += model
    data[:, :, 3] += model
    t.putcol("DATA", data)
    t.close()
    if a.json:
        json.dump({"near": a.near, "sources": placed}, open(a.json, "w"), indent=1)
    print("INJECTED", a.out, [p["name"] for p in placed])


if __name__ == "__main__":
    main()
