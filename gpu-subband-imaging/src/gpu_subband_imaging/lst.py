"""LST-hour grouping, the Sun cut and the main-pipeline-style output layout.

With ``layout: lst`` the products of one band and LST hour go to

  <output_root>/<LST>h/<date>/<run_label>/<band>MHz/
      snapshots/        <band>MHz-Pilot-Snapshot-<YYYYMMDD_HHMMSS>-image.fits.fz   dirty V
      snapshots_clean/  <band>MHz-Clean-Snapshot-<YYYYMMDD_HHMMSS>-image.fits.fz   CLEAN I
      I/deep, I/10min, V/deep, V/10min                                            1-h / 10-min
      QA/               snapshot_qa_<band>MHz.csv, pilot_qa_rejected.txt, plots, timings
      Movies/           <band>MHz_I_Clean.mp4
      provenance.json

as the main pipeline's ``<archive>/<LST>h/<date>/Run_<stamp>/<band>MHz/``.  A
snapshot belongs to the LST hour of its start time (mean sidereal time at
OVRO-LWA), and the Sun cut is the main pipeline's Phase 1 policy
(orca/utils/sun_cut.py: Sun at most -12 deg rising, -18 deg setting).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

OVRO_LAT_DEG = 37.23977727
OVRO_LON_DEG = -118.2816667
OVRO_HEIGHT_M = 1222.0
SUN_MAX_ALT_MORNING_DEG = -12.0
SUN_MAX_ALT_EVENING_DEG = -18.0

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def frame_time(name: str) -> datetime:
    """UTC start time from a name like ``20261026_031144_55MHz.ms.tar``."""
    m = _TS_RE.search(Path(name.rstrip("/")).name)
    if not m:
        raise ValueError(f"no YYYYMMDD_HHMMSS timestamp in {name}")
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")


def timestamp(name: str) -> str:
    """``YYYYMMDD_HHMMSS`` part of a snapshot name."""
    m = _TS_RE.search(Path(name.rstrip("/")).name)
    if not m:
        raise ValueError(f"no YYYYMMDD_HHMMSS timestamp in {name}")
    return f"{m.group(1)}_{m.group(2)}"


def _times(names: Sequence[str]):
    from astropy.time import Time
    return Time([frame_time(n) for n in names], scale="utc")


def _location():
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    return EarthLocation(lat=OVRO_LAT_DEG * u.deg, lon=OVRO_LON_DEG * u.deg,
                         height=OVRO_HEIGHT_M * u.m)


def lst_hours(names: Sequence[str]) -> List[int]:
    """Integer LST hour of each name's start time (mean sidereal time at OVRO-LWA, as
    pipeline/subband_celery.py generate_lst_segments)."""
    if not names:
        return []
    import astropy.units as u
    lst = _times(names).sidereal_time("mean", longitude=OVRO_LON_DEG * u.deg)
    return [int(h) % 24 for h in lst.hour]


def sun_ok(names: Sequence[str],
           morning_max_deg: float = SUN_MAX_ALT_MORNING_DEG,
           evening_max_deg: float = SUN_MAX_ALT_EVENING_DEG) -> List[bool]:
    """Main pipeline Sun cut for each name's start time."""
    if not names:
        return []
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import AltAz, get_sun
    t = _times(names)
    loc = _location()

    def alt(times):
        return np.atleast_1d(get_sun(times).transform_to(AltAz(obstime=times, location=loc)).alt.deg)

    a = alt(t)
    rising = alt(t + 60 * u.s) > a
    limit = np.where(rising, morning_max_deg, evening_max_deg)
    return [bool(x) for x in a <= limit]


def label(lst_hour: int) -> str:
    return f"{int(lst_hour):02d}h"


def phase_centre(lst_hour: int) -> str:
    """Main pipeline hour phase centre '<LST>h30m00s 37d12m57.057s'."""
    return f"{int(lst_hour):02d}h30m00s 37d12m57.057s"


def group(names: Iterable[str], hours: Sequence[int], sun_cut: bool = True
          ) -> Dict[int, List[str]]:
    """{LST hour: names} for the names in ``hours`` (and passing the Sun cut)."""
    names = sorted(names, key=lambda n: Path(n).name)
    lst = lst_hours(names)
    ok = sun_ok(names) if sun_cut else [True] * len(names)
    want = {int(h) for h in hours}
    out: Dict[int, List[str]] = {}
    for n, h, k in zip(names, lst, ok):
        if k and h in want:
            out.setdefault(h, []).append(n)
    return out


# ---- layout -----------------------------------------------------------------

def band_dir(output_root: str, lst_hour: int, date: str, run_label: str, band: int) -> Path:
    return Path(output_root) / label(lst_hour) / date / run_label / f"{band}MHz"


def pilot_name(band: int, ts: str) -> str:
    return f"{band}MHz-Pilot-Snapshot-{ts}-image.fits"


def clean_name(band: int, ts: str) -> str:
    return f"{band}MHz-Clean-Snapshot-{ts}-image.fits"
