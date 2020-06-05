from datetime import datetime
from typing import Union

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, ICRS, get_sun
from astropy import units as u

# Google Earth
OVRO_LWA_LOCATION = EarthLocation(lat=37.2398 * u.deg, lon=-118.282 * u.deg, height=1216 * u.m)

CAS_A = SkyCoord('23h23m24s', '+58deg48m54s', frame=ICRS)
CYG_A = SkyCoord('19h59m28.36s', '+40deg44m02.09s', frame=ICRS)
TAU_A = SkyCoord('05h34m31s +22deg00m52s', frame=ICRS)


def sun_icrs(utc_time: datetime) -> SkyCoord:
    sun_coords_gcrs = get_sun(Time(utc_time, scale='utc'))
    return SkyCoord(ra=sun_coords_gcrs.ra, dec=sun_coords_gcrs.dec)


def is_visible(coord: Union[SkyCoord, ICRS], elevation_limit: u.Quantity = 5 * u.degree,
               check_mountain: bool = True) -> bool:
    return False
