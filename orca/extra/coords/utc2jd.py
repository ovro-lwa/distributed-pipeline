"""UTC to Julian Date conversion.

This module converts UTC datetime components to Julian Date.
"""
from __future__ import division
from math import *


def utc2jd(year, month, day, hour, minute, second):
    """Convert UTC datetime to Julian Date.

    Args:
        year: Year (e.g., 2024).
        month: Month (1-12).
        day: Day of month.
        hour: Hour (0-23).
        minute: Minute (0-59).
        second: Second (0-59).

    Returns:
        Julian Date as float.

    Example:
        >>> utc2jd(2024, 1, 1, 12, 0, 0)
        2460311.0
    """
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
