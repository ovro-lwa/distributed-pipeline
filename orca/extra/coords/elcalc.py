"""Elevation calculation for celestial sources.

This module calculates the elevation (altitude) of a celestial source
given its RA/Dec and the zenith coordinates, assuming SIN projection.
"""
from __future__ import division
from math import *


def elcalc(ra, dec, zenra, zendec, returnaz=False):
    """Calculate elevation of a source above the horizon.

    Uses inverse spherical coordinate rotation with SIN projection.

    Args:
        ra: Right ascension of source in degrees.
        dec: Declination of source in degrees.
        zenra: Right ascension of zenith in degrees.
        zendec: Declination of zenith in degrees.
        returnaz: If True, also return azimuth. Defaults to False.

    Returns:
        Elevation in degrees, or tuple (elevation, azimuth) if returnaz=True.
    """
    if zendec >= 90.:
        phi_p = 0.
    elif zendec < 90.:
        phi_p = 180.
    # inverse spherical coordinate rotation
    phi   = radians(phi_p) + atan2( sin(radians(dec))*cos(radians(zendec)) -
        cos(radians(dec))*sin(radians(zendec))*cos(radians(ra-zenra)),
        -cos(radians(dec))*sin(radians(ra-zenra)) )
    theta = asin( sin(radians(dec))*sin(radians(zendec)) +
        cos(radians(dec))*cos(radians(zendec))*cos(radians(ra-zenra)) )

    if returnaz:
        return theta*180./pi, phi*180./pi - 270.
    else:
        return theta*180./pi
