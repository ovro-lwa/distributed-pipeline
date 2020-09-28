from __future__ import division
from math import *

def elcalc(ra,dec,zenra,zendec,returnaz=False):
    '''
    Returns elevation of source in deg, given source ra+dec, ra+dec of zenith, and assuming
    SIN projection. If returnaz=True, will also return azimuth of source in deg.
    Last edit: 23 Nov 2016
    '''
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
