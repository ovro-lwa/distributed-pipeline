from __future__ import division
from math import *
import numpy as np

OVRO_longitude = -118.281667

def jd2lst(jd,longitude=OVRO_longitude):
    '''
    Convert from Julian date to LST, defaults to the longitude of OVRO. 
    Output: LST in hours
    Equation taken from: http://aa.usno.navy.mil/faq/docs/GAST.php
    Last edit: 18 Nov 2016
    '''
    D    = jd - 2451545.0
    GMST = 18.697374558 + 24.06570982441908*D
    LST = np.mod(GMST + longitude * 24./360.,24)
    return LST
