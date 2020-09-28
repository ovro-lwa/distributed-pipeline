from __future__ import division
from math import *
import numpy as np

def angsep(phi1,theta1,phi2,theta2):
    '''
    Input:  phi1,theta1 - latitude and longitude of point1, respectively, in degrees
            phi2,theta2 - latitude and longitude of point2, respectively, in degrees
    Output: Returns the angular separation between the two points, in degrees.
    Uses the Vincenty formula -- https://en.wikipedia.org/wiki/Great-circle_distance
    Note: phi ==> latitude ==> dec ==> elevation
          theta ==> longitude ==> ra ==> azimuth
    Last edit: 17 Nov 2016
    '''
    deltatheta  = np.abs(theta2-theta1)
    numerator   = np.sqrt( (np.cos(np.radians(phi2))*np.sin(np.radians(deltatheta)))**2. 
            + (np.cos(np.radians(phi1))*np.sin(np.radians(phi2)) 
                - np.sin(np.radians(phi1))*np.cos(np.radians(phi2))*np.cos(np.radians(deltatheta)))**2. )
    denominator = np.sin(np.radians(phi1))*np.sin(np.radians(phi2)) \
            + np.cos(np.radians(phi1))*np.cos(np.radians(phi2))*np.cos(np.radians(deltatheta))
    separation  = np.arctan2(numerator , denominator)
    return separation * 180./pi
