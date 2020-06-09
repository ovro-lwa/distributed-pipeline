from datetime import datetime
from os import path
from typing import Union

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, ICRS, get_sun, AltAz, Angle
from astropy import units as u

"""
Coordinates utilities having to do with local coordinates. For image plane pixel numbers, use the fits header and 
astropy.WCS.
"""

# Google Earth
OVRO_LWA_LOCATION = EarthLocation(lat=37.2398 * u.deg, lon=-118.282 * u.deg, height=1216 * u.m)

CAS_A = SkyCoord('23h23m24s', '+58deg48m54s', frame=ICRS)
CYG_A = SkyCoord('19h59m28.36s', '+40deg44m02.09s', frame=ICRS)
TAU_A = SkyCoord('05h34m31s +22deg00m52s', frame=ICRS)

MOUNTAIN_AZ_DEG, MOUNTAIN_ALT_DEG = np.loadtxt(f'{path.dirname(__file__)}/../resources/mountain_azel.csv', unpack=True)
MOUNTAIN_AZ_INC = MOUNTAIN_AZ_DEG[1] - MOUNTAIN_AZ_DEG[0]


def sun_icrs(utc_time: datetime) -> SkyCoord:
    sun_coords_gcrs = get_sun(Time(utc_time, scale='utc'))
    return SkyCoord(ra=sun_coords_gcrs.ra, dec=sun_coords_gcrs.dec)


def is_visible(coordinates: Union[SkyCoord, ICRS], utc_time: datetime, altitude_limit: u.Quantity = 5 * u.degree,
               check_mountain: bool = True) -> bool:
    """
    Check if a source with coord is altitude_limit (default to 5 deg)
    above the horizon (and optionally the mountains)
    :param coordinates: coordinates of the object to check for visibility
    :param utc_time: utc time
    :param altitude_limit: altitude limit above the horizon/mountain top
    :param check_mountain: determines whether to take the altitude of the mountain into account
    :return:
    """
    altaz = get_altaz_at_ovro(coordinates, utc_time)
    # get the mountain stuff
    if check_mountain:
        return altaz.alt > altitude_limit + _get_interpolated_mountain_alt(altaz.az)
    else:
        return altaz.alt > altitude_limit


def get_altaz_at_ovro(coordinates: SkyCoord, utc_time: datetime) -> SkyCoord:
    return coordinates.transform_to(AltAz(location=OVRO_LWA_LOCATION, obstime=Time(utc_time, scale='utc')))


def _get_interpolated_mountain_alt(az: Union[u.Quantity, Angle]) -> u.Quantity:
    assert 0 * u.deg <= az <= 360 * u.deg, f'Azimuth angle must be between 0 and 360 degrees, got {az}'
    closest_index = int(round((az.to(u.deg).value - MOUNTAIN_AZ_DEG[0]) / MOUNTAIN_AZ_INC))
    return MOUNTAIN_ALT_DEG[closest_index] * u.deg