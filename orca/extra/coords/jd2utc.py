from __future__ import division
from math import *

def jd2utc(jd):
    '''
    Go from Julian date to UTC.
    Source:    http://stackoverflow.com/questions/29627533/conversion-of-julian-date-number-to-normal-date-utc-in-javascript/29627963#29627963
    Last edit: 04 April 2016
    '''
    X = jd + 0.5
    Z = floor(X)
    F = X-Z
    Y = floor((Z-1867216.25)/36524.24)
    A = Z+1+Y-floor(Y/4)
    B = A+1524
    C = floor((B-122.1)/365.25)
    D = floor(365.25*C)
    G = floor((B-D)/30.6001)
    if G < 13.5:
        month=G-1
    else:
        month=G-13
    if month<2.5:
        year = C-4715
    else:
        year = C-4716

    UT = B-D-floor(30.6001*G)+F
    day = floor(UT)
    UT -= floor(UT)
    UT *= 24
    hour = floor(UT)
    UT -= floor(UT)
    UT *= 60
    minute = floor(UT)
    UT -= floor(UT)
    UT *= 60
    second = round(UT)
    return '%04d-%02d-%02d %02d:%02d:%02d' % (year,month,day,hour,minute,second)
