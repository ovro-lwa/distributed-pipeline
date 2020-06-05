import logging
from orca.wrapper import ttcal
from typing import List

import casacore.tables as tables
from astropy import coordinates, time, units as u

log = logging.getLogger(__name__)
CORRECTED_DATA = 'CORRECTED_DATA'
DATA = 'DATA'


def peel_with_ttcal(ms: str, sources_json: str):
    with tables.table(ms, readonly=False) as t:
        # Copied from https://github.com/casacore/python-casacore/blob/master/casacore/tables/msutil.py#L48
        column_names = t.colnames()
        if CORRECTED_DATA not in column_names:
            dminfo = t.getdminfo(DATA)
            cdesc = t.getcoldesc(DATA)
            dminfo['NAME'] = 'correcteddata'
            cdesc['comment'] = 'The corrected data column'
            t.addcols(pt.maketabdesc(pt.makecoldesc(CORRECTED_DATA, cdesc)), dminfo)
            t.putcol(CORRECTED_DATA, t.getcol(DATA))
        else:
            log.info(f'{ms} already has {CORRECTED_DATA} column. Not peeling.')
            return ms
    # corrected data column stuff
    ttcal.peel_with_ttcal(ms, sources_json)
    return ms


def write_peeling_sources_json(ms: str, out_json: str, include_rfi_source: bool):
    pass


def get_peeling_sources_dict(ms: str, include_rfi_source: bool) -> List[dict]:
    sources = []
    with tables.table(ms) as t:
        obs_time = time.Time(t.getcell('TIME', 0))
    # From Google Earth
    observing_location = coordinates.EarthLocation(lat=37.2398 * u.deg, lon=-118.282 * u.deg, height=1216 * u.m)
    local_altaz_frame = coordinates.AltAz(location=observing_location, obstime=obs_time)
    # Is CasA up?

    # Is CygA up?

    if include_rfi_source:
        sources.append(
            {
                "name": "Noise Power Lines2",
                "sys": "WGS84",
                "long": -118.3852914162684,
                "lat": 37.3078474772316,
                "el": 1214.248326037079,
                "rfi-frequencies": [2.60e7, 2.87e7, 3.13e7, 3.39e7, 3.65e7, 3.91e7, 4.18e7, 4.44e7, 4.70e7, 4.96e7,
                                    5.22e7, 5.48e7, 5.75e7, 6.00e7, 6.27e7, 6.53e7, 6.79e7, 7.05e7, 7.32e7, 7.58e7,
                                    7.84e7, 8.10e7],
                "rfi-I": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0]
            }
        )
    return sources
