"""Reject integrations taken with the Sun too high (Phase 1 policy).

Morning (Sun rising): refuse frames with the Sun above -12 deg.
Evening (Sun setting): refuse frames with the Sun above -18 deg.

Frame times come from the archive file names (``YYYYMMDD_HHMMSS_...``, UTC).
Whether the Sun is rising or setting is taken from the sign of its altitude
change over one minute, so the cut works for any date and UTC hour.
"""
import os
import re
import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

logger = logging.getLogger(__name__)

OVRO_LOC = EarthLocation(lat=37.23977727 * u.deg, lon=-118.2816667 * u.deg,
                         height=1222 * u.m)
SUN_MAX_ALT_MORNING_DEG = -12.0
SUN_MAX_ALT_EVENING_DEG = -18.0

_TS_RE = re.compile(r'(\d{8})_(\d{6})')


def frame_time(path: str) -> datetime:
    """UTC start time from an archive file name like ``20250421_120004_50MHz_averaged.ms``."""
    m = _TS_RE.search(os.path.basename(path.rstrip('/')))
    if not m:
        raise ValueError(f"No YYYYMMDD_HHMMSS timestamp in {path}")
    return datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M%S')


def sun_altitude(times: Time) -> np.ndarray:
    """Geometric Sun altitude (deg, no refraction) at OVRO for *times*."""
    frame = AltAz(obstime=times, location=OVRO_LOC)
    return get_sun(times).transform_to(frame).alt.deg


def sun_ok_mask(times: Time,
                morning_max_deg: float = SUN_MAX_ALT_MORNING_DEG,
                evening_max_deg: float = SUN_MAX_ALT_EVENING_DEG,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ok, alt_deg, rising)`` arrays for *times*."""
    alt = np.atleast_1d(sun_altitude(times))
    rising = np.atleast_1d(sun_altitude(times + 60 * u.s)) > alt
    limit = np.where(rising, morning_max_deg, evening_max_deg)
    return alt <= limit, alt, rising


def filter_sun(ms_files: List[str],
               morning_max_deg: float = SUN_MAX_ALT_MORNING_DEG,
               evening_max_deg: float = SUN_MAX_ALT_EVENING_DEG,
               ) -> Tuple[List[str], List[str]]:
    """Split *ms_files* into (kept, refused) by the Sun-altitude policy."""
    if not ms_files:
        return [], []
    times = Time([frame_time(f) for f in ms_files], scale='utc')
    ok, alt, rising = sun_ok_mask(times, morning_max_deg, evening_max_deg)
    kept = [f for f, k in zip(ms_files, ok) if k]
    refused = [f for f, k in zip(ms_files, ok) if not k]
    if refused:
        bad = ~ok
        logger.info(
            f"Sun cut: refused {len(refused)}/{len(ms_files)} frames "
            f"({int(np.sum(bad & rising))} morning > {morning_max_deg}°, "
            f"{int(np.sum(bad & ~rising))} evening > {evening_max_deg}°; "
            f"Sun alt {alt[bad].min():.1f}..{alt[bad].max():.1f}°)"
        )
    return kept, refused
