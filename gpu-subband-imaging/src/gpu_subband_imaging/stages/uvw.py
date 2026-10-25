"""Recompute the UVW column of a snapshot MS in J2000 (snapshot astrometry fix).

The OVRO-LWA correlator MSes label PHASE_DIR (and the UVW frame used by WSClean)
J2000, but their stored (u, v) axes are rotated about w by ~0.33 deg (2026)
relative to the J2000 (u, v) axes for that PHASE_DIR; w itself is exact at
TIME_CENTROID.  WSClean takes UVW and PHASE_DIR as given, so every snapshot
image is rotated by that angle about its phase centre (~15' at 40 deg from
zenith).  The main pipeline is immune because ``chgcentre`` recomputes UVW.

``fix_uvw`` replaces UVW with casacore-measures J2000 UVW for the MS PHASE_DIR
(per-antenna ``to_uvw``, baseline uvw = A[ANTENNA1] - A[ANTENNA2], the sign
convention of these MSes) at each row's TIME_CENTROID (``epoch="centroid"``,
the instant PHASE_DIR and the stored w refer to; the raw MS TIME is the start
of the 10-s integration) or at TIME (``epoch="time"``, what ``chgcentre``
does).  Any change in w is applied to the visibilities as in ``chgcentre``
(vis *= exp(2 pi i (w_old - w_new) / lambda)); at the centroid it is
sub-millimetre and skipped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("gsi.uvw")

C_LIGHT = 299792458.0
EPOCHS = ("centroid", "time")


@dataclass(frozen=True)
class UVWFixStats:
    n_times: int
    rot_deg: float          # rotation about w taking the stored (u, v) to the new ones
    rms_duv_m: float        # rms |(u, v) new - stored|
    max_dw_m: float         # max |w new - stored|
    rephased: bool


def antenna_uvw(ant_pos: np.ndarray, time_s: float, ra: float, dec: float) -> np.ndarray:
    """Per-antenna J2000 uvw [m], shape (nant, 3), at UTC MJD seconds ``time_s``."""
    from casacore.measures import measures
    from casacore.quanta import quantity as q
    me = measures()
    me.do_frame(me.epoch("UTC", q(time_s, "s")))
    me.do_frame(me.position("ITRF", *[q(float(x), "m") for x in ant_pos.mean(0)]))
    me.do_frame(me.direction("J2000", q(ra, "rad"), q(dec, "rad")))
    b = me.baseline("ITRF", q(ant_pos[:, 0], "m"), q(ant_pos[:, 1], "m"),
                    q(ant_pos[:, 2], "m"))
    return np.array(me.to_uvw(b)["xyz"].get_value()).reshape(-1, 3)


def rotation_about_w_deg(uvw_from: np.ndarray, uvw_to: np.ndarray) -> float:
    """Least-squares angle [deg] of the rotation about w taking uvw_from (u, v) to uvw_to."""
    u0, v0, u1, v1 = uvw_from[:, 0], uvw_from[:, 1], uvw_to[:, 0], uvw_to[:, 1]
    return float(np.degrees(np.arctan2(np.sum(u0 * v1 - v0 * u1), np.sum(u0 * u1 + v0 * v1))))


def _data_columns(t) -> Sequence[str]:
    return [c for c in ("DATA", "CORRECTED_DATA", "MODEL_DATA") if c in t.colnames()]


def fix_uvw(ms: Path, epoch: str = "centroid", max_dw_m: float = 1.0,
            rephase_tol_rad: float = 1e-3, dry_run: bool = False) -> UVWFixStats:
    """Rewrite UVW (and rephase if w changes) for the MS's own J2000 PHASE_DIR.

    ``dry_run`` only measures how far the stored UVW are from J2000 (a
    consistency check: rot_deg ~ 0 and max_dw_m ~ 0 for a correct MS).
    Raises ValueError if any |w new - stored| exceeds ``max_dw_m`` (the MS
    phase centre and stored UVW disagree; nothing is written then).
    """
    if epoch not in EPOCHS:
        raise ValueError(f"uvw epoch must be one of {EPOCHS}, got {epoch!r}")
    from casacore.tables import table

    with table(str(Path(ms) / "FIELD"), ack=False) as f:
        if f.nrows() != 1:
            raise ValueError(f"{ms}: expected one FIELD row, got {f.nrows()}")
        ref = f.getcolkeywords("PHASE_DIR").get("MEASINFO", {}).get("Ref", "J2000")
        if ref != "J2000":
            raise ValueError(f"{ms}: PHASE_DIR frame {ref}, expected J2000")
        ra, dec = (float(x) for x in f.getcol("PHASE_DIR")[0, 0])
    with table(str(Path(ms) / "ANTENNA"), ack=False) as a:
        ant_pos = a.getcol("POSITION")
    with table(str(Path(ms) / "SPECTRAL_WINDOW"), ack=False) as s:
        if s.nrows() != 1:
            raise ValueError(f"{ms}: expected one spectral window, got {s.nrows()}")
        freqs = s.getcol("CHAN_FREQ")[0]

    with table(str(ms), ack=False, readonly=dry_run) as t:
        old = t.getcol("UVW")
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        tcol = t.getcol("TIME_CENTROID" if epoch == "centroid" else "TIME")
        new = np.empty_like(old)
        times = np.unique(tcol)
        for ts in times:
            rows = tcol == ts
            A = antenna_uvw(ant_pos, float(ts), ra, dec)
            new[rows] = A[a1[rows]] - A[a2[rows]]
        dw = new[:, 2] - old[:, 2]
        max_dw = float(np.max(np.abs(dw))) if dw.size else 0.0
        if max_dw > max_dw_m and not dry_run:
            raise ValueError(f"{ms}: |w new - stored| up to {max_dw:.3f} m > {max_dw_m} m; "
                             "PHASE_DIR and stored UVW disagree, UVW left unchanged")
        cross = a1 != a2
        rot = rotation_about_w_deg(old[cross], new[cross]) if cross.any() else 0.0
        rms_duv = float(np.sqrt(np.mean(np.sum((new[cross, :2] - old[cross, :2]) ** 2, 1)))) \
            if cross.any() else 0.0
        rephase = (not dry_run and
                   max_dw * 2 * np.pi * float(np.max(freqs)) / C_LIGHT > rephase_tol_rad)
        if rephase:
            ph = np.exp(-2j * np.pi * dw[:, None] * freqs[None, :] / C_LIGHT)[:, :, None]
            for col in _data_columns(t):
                d = t.getcol(col)
                t.putcol(col, (d * ph).astype(d.dtype))
        if not dry_run:
            t.putcol("UVW", new)
    return UVWFixStats(n_times=int(times.size), rot_deg=rot, rms_duv_m=rms_duv,
                       max_dw_m=max_dw, rephased=bool(rephase))
