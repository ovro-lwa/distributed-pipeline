"""A small synthetic OVRO-LWA-like raw snapshot MS for the gsi v3 prep tests (needs casacore).

Rows are every antenna pair with autos (a1 <= a2, raw MS order), one integration,
``nchan`` channels, 4 correlations, WEIGHT = 1 and SIGMA = 9999 as in the raw MSes.
The phase centre is the zenith at the integration centre and the stored UVW are
rotated about w by ``stored_rot_deg`` (the correlator's rotation that fix_uvw removes).
"""
from __future__ import annotations

import numpy as np

OVRO = np.array([-2409150.0, -4478573.0, 3838617.0])     # ITRF, m
T_START = 5299554608.706455
INTERVAL = 10.031020408163265


def _rot_w(x, deg):
    a = np.radians(deg)
    out = x.copy()
    out[:, 0] = np.cos(a) * x[:, 0] - np.sin(a) * x[:, 1]
    out[:, 1] = np.sin(a) * x[:, 0] + np.cos(a) * x[:, 1]
    return out


def make_raw_ms(path, nant=10, nchan=16, f0=54.74e6, df=23925.78125, seed=3,
                flag_frac=0.2, stored_rot_deg=-0.3, time_centroid_offset=0.0):
    """Write the MS; return a dict of what went in (positions, freqs, ra/dec, ...)."""
    from casacore import tables
    from casacore.measures import measures
    from casacore.quanta import quantity as q
    from gpu_subband_imaging.stages import uvw

    rng = np.random.default_rng(seed)
    enu = rng.uniform(-600, 600, size=(nant, 3)) * [1, 1, 0.003]
    lat, lon = np.radians(37.2398), np.radians(-118.2817)
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.cross(east, north)
    pos = OVRO + enu[:, :1] * east + enu[:, 1:2] * north + enu[:, 2:] * up
    tc = T_START + time_centroid_offset
    me = measures()
    me.do_frame(me.epoch("UTC", q(tc, "s")))
    me.do_frame(me.position("ITRF", *[q(x, "m") for x in pos.mean(0)]))
    z = me.measure(me.direction("AZEL", q(0, "deg"), q(90, "deg")), "J2000")
    ra, dec = z["m0"]["value"], z["m1"]["value"]

    a1, a2 = np.triu_indices(nant)
    nrow = len(a1)
    A = uvw.antenna_uvw(pos, tc, ra, dec)
    true = A[a1] - A[a2]
    freqs = f0 + df * np.arange(nchan)

    desc = tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, valuetype="complex", ndim=2)])
    t = tables.default_ms(str(path), desc)
    t.addrows(nrow)
    data = (rng.normal(size=(nrow, nchan, 4)) + 1j * rng.normal(size=(nrow, nchan, 4))) * 50
    flag = rng.random((nrow, nchan, 4)) < flag_frac
    flag[3] = True                                  # one fully flagged row
    flag[5, : nchan // 2] = True                    # a row half flagged
    t.putcol("ANTENNA1", a1.astype(np.int32))
    t.putcol("ANTENNA2", a2.astype(np.int32))
    t.putcol("UVW", _rot_w(true, stored_rot_deg))
    t.putcol("TIME", np.full(nrow, T_START))
    t.putcol("TIME_CENTROID", np.full(nrow, tc))
    t.putcol("INTERVAL", np.full(nrow, INTERVAL))
    t.putcol("EXPOSURE", np.full(nrow, INTERVAL))
    t.putcol("DATA", data.astype(np.complex64))
    t.putcol("FLAG", flag)
    t.putcol("FLAG_ROW", np.zeros(nrow, bool))
    t.putcol("WEIGHT", np.ones((nrow, 4), np.float32))
    t.putcol("SIGMA", np.full((nrow, 4), 9999.0, np.float32))
    for c in ("ARRAY_ID", "DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID", "PROCESSOR_ID",
              "STATE_ID", "FEED1", "FEED2"):
        t.putcol(c, np.zeros(nrow, np.int32))
    t.putcol("SCAN_NUMBER", np.ones(nrow, np.int32))
    t.close()

    def sub(name):
        return tables.table(f"{path}/{name}", readonly=False, ack=False)

    with sub("ANTENNA") as an:
        an.addrows(nant)
        an.putcol("POSITION", pos)
        an.putcol("NAME", [f"LWA{i:03d}" for i in range(nant)])
        an.putcol("STATION", ["OVRO_LWA"] * nant)
        an.putcol("TYPE", ["GROUND-BASED"] * nant)
        an.putcol("MOUNT", ["X-Y"] * nant)
        an.putcol("DISH_DIAMETER", np.full(nant, 2.0))
    with sub("FEED") as fd:
        fd.addrows(nant)
        fd.putcol("ANTENNA_ID", np.arange(nant, dtype=np.int32))
        fd.putcol("FEED_ID", np.zeros(nant, np.int32))
        fd.putcol("SPECTRAL_WINDOW_ID", np.full(nant, -1, np.int32))
        fd.putcol("TIME", np.full(nant, tc))
        fd.putcol("NUM_RECEPTORS", np.full(nant, 2, np.int32))
        fd.putcol("BEAM_OFFSET", np.zeros((nant, 2, 2)))
        fd.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]] * nant))
        fd.putcol("POL_RESPONSE", np.tile(np.eye(2, dtype=np.complex64), (nant, 1, 1)))
        fd.putcol("POSITION", np.zeros((nant, 3)))
        fd.putcol("RECEPTOR_ANGLE", np.tile([0.0, np.pi / 2], (nant, 1)))
    with sub("FIELD") as f:
        f.addrows(1)
        for c in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            f.putcol(c, np.array([[[ra, dec]]]))
        f.putcol("NAME", ["zenith"])
        f.putcol("TIME", np.array([tc]))
        f.putcol("SOURCE_ID", np.array([0], np.int32))
    with sub("SPECTRAL_WINDOW") as s:
        s.addrows(1)
        s.putcol("NUM_CHAN", np.array([nchan], np.int32))
        s.putcol("CHAN_FREQ", freqs[None, :])
        for c in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
            s.putcol(c, np.full((1, nchan), df))
        s.putcol("REF_FREQUENCY", np.array([freqs[0]]))
        s.putcol("TOTAL_BANDWIDTH", np.array([df * nchan]))
        s.putcol("MEAS_FREQ_REF", np.array([5], np.int32))
        s.putcol("NAME", [f"IF 1, {nchan} channels"])
    with sub("POLARIZATION") as p:
        p.addrows(1)
        p.putcol("NUM_CORR", np.array([4], np.int32))
        p.putcol("CORR_TYPE", np.array([[9, 10, 11, 12]], np.int32))
        p.putcol("CORR_PRODUCT", np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], np.int32))
    with sub("DATA_DESCRIPTION") as dd:
        dd.addrows(1)
        dd.putcol("SPECTRAL_WINDOW_ID", np.array([0], np.int32))
        dd.putcol("POLARIZATION_ID", np.array([0], np.int32))
    with sub("OBSERVATION") as o:
        o.addrows(1)
        o.putcol("TELESCOPE_NAME", ["OVRO_LWA"])
        o.putcol("OBSERVER", ["gsi-test"])
        o.putcol("TIME_RANGE", np.array([[T_START, T_START + INTERVAL]]))
    tables.default_ms_subtable("SOURCE", f"{path}/SOURCE").close()
    with tables.table(str(path), readonly=False, ack=False) as t:
        t.putkeyword("SOURCE", f"Table: {path}/SOURCE")
    with sub("SOURCE") as so:
        so.addrows(1)
        so.putcol("SOURCE_ID", np.array([0], np.int32))
        so.putcol("SPECTRAL_WINDOW_ID", np.array([-1], np.int32))
        so.putcol("NAME", ["zenith"])
        so.putcol("TIME", np.array([tc]))
        so.putcol("DIRECTION", np.array([[ra, dec]]))
    with sub("PROCESSOR") as pr:
        pr.addrows(1)
        pr.putcol("TYPE", ["CORRELATOR"])
    return {"pos": pos, "ra": ra, "dec": dec, "freqs": freqs, "a1": a1, "a2": a2,
            "true_uvw": true, "nant": nant}


def jones_for(nant, nchan, seed=5, dead=((1, 0),)):
    """Synthetic (J, dead) as fastcal.jones returns them: (nant, nchan, 2) gains."""
    rng = np.random.default_rng(seed)
    J = (1 + 0.3 * rng.normal(size=(nant, nchan, 2))) * np.exp(1j * rng.uniform(-3, 3, (nant, nchan, 2)))
    dd = np.zeros((nant, 2), bool)
    for a, p in dead:
        dd[a, p] = True
    return J, dd
