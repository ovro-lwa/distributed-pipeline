"""Empirical sign/geometry checks against the data (no gains).

    python geometry_probe.py PREP_MS [--rfi-lon -118.62673 --rfi-lat 37.53159 --rfi-h 2072]

1. MS UVW vs antenna positions: is UVW = r2 - r1 or r1 - r2 (projected)?
2. Correlation of the data with near-field RFI fringes for sign variants:
   V = exp(i s1 k (d2 - d1) + i s2 k w)  for s1, s2 in {+1, -1}.
3. Correlation with a far-field point source at the phase centre offset of
   Cas A / Cyg A / Vir A for both exp(-ik b.Δs) and exp(+ik b.Δs).
"""
import argparse
import numpy as np
from casacore.tables import table
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, ITRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u

ap = argparse.ArgumentParser()
ap.add_argument("ms")
ap.add_argument("--rfi-lon", type=float, default=-118.62673)
ap.add_argument("--rfi-lat", type=float, default=37.53159)
ap.add_argument("--rfi-h", type=float, default=2072.0)
a = ap.parse_args()

t = table(a.ms, ack=False)
data = t.getcol("DATA"); flag = t.getcol("FLAG"); a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2")
uvw = t.getcol("UVW"); time = t.getcell("TIME", 0)
pos = table(a.ms + "/ANTENNA", ack=False).getcol("POSITION")
freq = table(a.ms + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
pd = table(a.ms + "/FIELD", ack=False).getcol("PHASE_DIR")[0, 0]
k = 2 * np.pi * freq / 299792458.0
cross = a1 != a2
I = 0.5 * (data[:, :, 0] + data[:, :, 3])
good = (~(flag[:, :, 0] | flag[:, :, 3])) & cross[:, None]
I = np.where(good, I, 0)

obstime = Time(time / 86400.0, format="mjd", scale="utc")
center = EarthLocation.from_geocentric(*pos.mean(0), unit=u.m)
# Phase centre unit vector in ITRF (astropy: apparent, of date)
pc = SkyCoord(pd[0] * u.rad, pd[1] * u.rad, frame="icrs")
s0 = pc.transform_to(ITRS(obstime=obstime)).cartesian.xyz.value
s0 /= np.linalg.norm(s0)
b = pos[a2] - pos[a1]
w_geom = b @ s0
x = cross
fit = np.polyfit(w_geom[x], uvw[x, 2], 1)
print(f"MS w vs s0.(r2-r1): slope={fit[0]:+.5f} offset={fit[1]:+.3f} m  (|b| rms {np.sqrt(np.mean(np.sum(b[x]**2,1))):.1f} m)")
print(f"  rms(w_MS - w_geom)={np.std(uvw[x,2]-w_geom[x]):.3f}  rms(w_MS + w_geom)={np.std(uvw[x,2]+w_geom[x]):.3f}")
uvw_len = np.linalg.norm(uvw[x], axis=1); b_len = np.linalg.norm(b[x], axis=1)
print(f"  |uvw| vs |b|: max diff {np.max(np.abs(uvw_len-b_len)):.3e} m")


def corr(model):
    m = np.where(good, model, 0)
    per = [abs(np.vdot(m[:, c], I[:, c])) / (np.linalg.norm(m[:, c]) * np.linalg.norm(I[:, c]) + 1e-30)
           for c in range(I.shape[1])]
    return np.median(per)


P = EarthLocation.from_geodetic(a.rfi_lon * u.deg, a.rfi_lat * u.deg, a.rfi_h * u.m)
P = np.array([P.x.value, P.y.value, P.z.value])
d = np.linalg.norm(P[None, :] - pos, axis=1)
dd = (d[a2] - d[a1])[:, None]
print("RFI near-field fringe variants: corr(I, model) per-channel median")
for s1 in (+1, -1):
    for name, w in (("w_MS", uvw[:, 2]), ("w_geom", w_geom), ("no_w", 0 * w_geom)):
        for s2 in ((+1, -1) if name != "no_w" else (+1,)):
            ph = k[None, :] * (s1 * dd + s2 * w[:, None])
            print(f"  s1={s1:+d} {name:7s} s2={s2:+d}: {corr(np.exp(1j*ph)):.3f}")

print("Far-field point sources (b = r2 - r1, ITRF, apparent)")
for name, c in (("CasA", SkyCoord("23h23m24s", "+58d48m54s")), ("CygA", SkyCoord("19h59m28.36s", "+40d44m02.1s")),
                ("VirA", SkyCoord("12h30m49.4s", "+12d23m28s"))):
    alt = c.transform_to(AltAz(obstime=obstime, location=center)).alt.deg
    if alt < 0:
        continue
    s = c.transform_to(ITRS(obstime=obstime)).cartesian.xyz.value; s /= np.linalg.norm(s)
    bd = (b @ (s - s0))[:, None]
    print(f"  {name} alt={alt:.1f}: exp(-ik b.ds)={corr(np.exp(-1j*k*bd)):.3f}  exp(+ik b.ds)={corr(np.exp(1j*k*bd)):.3f}")
