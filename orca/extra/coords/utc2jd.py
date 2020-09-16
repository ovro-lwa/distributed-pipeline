from __future__ import division
from math import *

def utc2jd(year, month, day, hour, minute, second):
    '''
    Go from UTC to Julian date.
    If you don't trust the answer, check here:
        http://aa.usno.navy.mil/data/docs/JulianDate.php
    Example usage: utc2jd(*filter(None,re.split('[-:T]','2015-03-29T06:04:46.0')))
    Last edit: 18 Nov 2016
    '''
    year   = float(year)
    month  = float(month)
    day    = float(day)
    hour   = float(hour)
    minute = float(minute)
    second = float(second)
    a      = int((14 - month)/12.)
    y      = year + 4800 - a
    m      = month + 12*a - 3
    jdn    = day + int((153*m + 2)/5.) + 365*y + int(y/4.) - int(y/100.) + int(y/400.) - 32045
    utc    = (hour - 12)/24. + minute/1440. + second/86400.
    jd     = jdn + utc
    return jd
