"""Which sky frame do MS UVW live in?  Cas A model vs data (no gains).

    python frame_probe.py PREP_MS

(a) astropy apparent ITRF direction with b = r2 - r1 (CPU TTCal-like)
(b) TTCalX: J2000 l,m,n about PHASE_DIR with -UVW (auto convention)
(c) as (b) but source and phase centre both in apparent (CIRS/TETE) RA/Dec
"""
import sys
import numpy as np
from casacore.tables import table
from astropy.coordinates import SkyCoord, ITRS, TETE, CIRS, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

ms = sys.argv[1]
t = table(ms, ack=False)
data = t.getcol("DATA"); flag = t.getcol("FLAG"); a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2")
uvw = -t.getcol("UVW"); time = t.getcell("TIME", 0)
pos = table(ms + "/ANTENNA", ack=False).getcol("POSITION")
freq = table(ms + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
pd = table(ms + "/FIELD", ack=False).getcol("PHASE_DIR")[0, 0]
k = 2 * np.pi * freq / 299792458.0
cross = a1 != a2
I = 0.5 * (data[:, :, 0] + data[:, :, 3])
good = (~(flag[:, :, 0] | flag[:, :, 3])) & cross[:, None]
I = np.where(good, I, 0)
obstime = Time(time / 86400.0, format="mjd", scale="utc")
loc = EarthLocation.from_geocentric(*pos.mean(0), unit=u.m)
b = pos[a2] - pos[a1]


def corr(model, longonly=False):
    sel = good.copy()
    if longonly:
        sel &= (np.linalg.norm(b, axis=1) > 1000)[:, None]
    m = np.where(sel, model, 0); d = np.where(sel, I, 0)
    per = [abs(np.vdot(m[:, c], d[:, c])) / (np.linalg.norm(m[:, c]) * np.linalg.norm(d[:, c]) + 1e-30)
           for c in range(I.shape[1])]
    return np.median(per)


def lmn(ra, dec, ra0, dec0):
    dra = ra - ra0
    return (np.cos(dec) * np.sin(dra),
            np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra),
            np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra))


pc = SkyCoord(pd[0] * u.rad, pd[1] * u.rad, frame="icrs")
s0 = pc.transform_to(ITRS(obstime=obstime)).cartesian.xyz.value
print(f"PHASE_DIR ra={np.degrees(pd[0]):.4f} dec={np.degrees(pd[1]):.4f}; MJD {time/86400:.4f}")
for fr in (TETE(obstime=obstime), CIRS(obstime=obstime)):
    p = pc.transform_to(fr)
    print(f"  phase centre in {fr.name}: ra={p.ra.deg:.4f} dec={p.dec.deg:.4f}")
for name, c in (("CasA", SkyCoord("23h23m24s", "+58d48m54s")), ("CygA", SkyCoord("19h59m28.36s", "+40d44m02.1s")),
                ("VirA", SkyCoord("12h30m49.4s", "+12d23m28s"))):
    alt = c.transform_to(AltAz(obstime=obstime, location=loc)).alt.deg
    if alt < 5:
        continue
    s = c.transform_to(ITRS(obstime=obstime)).cartesian.xyz.value
    ma = np.exp(-1j * k[None, :] * (b @ (s - s0))[:, None])
    out = {"a_itrf": ma}
    for tag, fr in (("b_j2000", None), ("c_tete", TETE(obstime=obstime)), ("c_cirs", CIRS(obstime=obstime))):
        if fr is None:
            l, m, n = lmn(c.ra.rad, c.dec.rad, pd[0], pd[1])
        else:
            cs = c.transform_to(fr); ps = pc.transform_to(fr)
            l, m, n = lmn(cs.ra.rad, cs.dec.rad, ps.ra.rad, ps.dec.rad)
        for su in (1, -1):
            for sv in (1, -1):
                ph = (su * uvw[:, 0] * l + sv * uvw[:, 1] * m + uvw[:, 2] * (n - 1))[:, None] * k[None, :]
                out[f"{tag}{su:+d}{sv:+d}"] = np.exp(1j * ph)
    print(f"{name} alt={alt:.1f}: " + "  ".join(f"{t}={corr(v):.3f}/{corr(v, True):.3f}" for t, v in out.items())
          + "   (all/long>1km)")
