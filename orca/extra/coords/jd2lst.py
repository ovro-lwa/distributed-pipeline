"""Julian Date to Local Sidereal Time conversion.

This module converts Julian Date to Local Sidereal Time (LST),
defaulting to the OVRO longitude.
"""
from __future__ import division
from math import *
import numpy as np

OVRO_longitude = -118.281667


def jd2lst(jd, longitude=OVRO_longitude):
    """Convert Julian Date to Local Sidereal Time.

    Uses the USNO algorithm for GMST calculation.

    Args:
        jd: Julian Date.
        longitude: Observer longitude in degrees. Defaults to OVRO.

    Returns:
        Local Sidereal Time in hours.

    Reference:
        http://aa.usno.navy.mil/faq/docs/GAST.php
    """
    D    = jd - 2451545.0
    GMST = 18.697374558 + 24.06570982441908*D
    LST = np.mod(GMST + longitude * 24./360.,24)
    return LST
