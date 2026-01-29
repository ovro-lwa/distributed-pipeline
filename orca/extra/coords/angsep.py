"""Angular separation calculation using Vincenty formula.

This module provides accurate angular separation calculation between
two points on the celestial sphere using the Vincenty formula for
great-circle distance.

Note:
    For new code, prefer astropy.coordinates.SkyCoord.separation().
"""
from __future__ import division
from math import *
import numpy as np


def angsep(phi1, theta1, phi2, theta2):
    """Calculate angular separation between two points on the sphere.

    Uses the Vincenty formula for accurate great-circle distance.

    Args:
        phi1: Latitude (declination/elevation) of point 1 in degrees.
        theta1: Longitude (RA/azimuth) of point 1 in degrees.
        phi2: Latitude (declination/elevation) of point 2 in degrees.
        theta2: Longitude (RA/azimuth) of point 2 in degrees.

    Returns:
        Angular separation in degrees.

    Note:
        phi corresponds to latitude/dec/elevation.
        theta corresponds to longitude/ra/azimuth.
    """
    deltatheta  = np.abs(theta2-theta1)
    numerator   = np.sqrt( (np.cos(np.radians(phi2))*np.sin(np.radians(deltatheta)))**2. 
            + (np.cos(np.radians(phi1))*np.sin(np.radians(phi2)) 
                - np.sin(np.radians(phi1))*np.cos(np.radians(phi2))*np.cos(np.radians(deltatheta)))**2. )
    denominator = np.sin(np.radians(phi1))*np.sin(np.radians(phi2)) \
            + np.cos(np.radians(phi1))*np.cos(np.radians(phi2))*np.cos(np.radians(deltatheta))
    separation  = np.arctan2(numerator , denominator)
    return separation * 180./pi
