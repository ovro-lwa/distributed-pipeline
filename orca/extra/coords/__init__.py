"""Coordinate transformation utilities.

This subpackage provides coordinate conversion and angular calculation utilities
for astronomical observations. Includes conversions between Julian Date, UTC,
and LST, as well as angular separation and elevation calculations.

Modules:
    AngularCoordinate: Angular coordinate parsing and formatting.
    angsep: Angular separation using Vincenty formula.
    elcalc: Source elevation calculation.
    jd2lst: Julian Date to Local Sidereal Time conversion.
    jd2utc: Julian Date to UTC conversion.
    utc2jd: UTC to Julian Date conversion.

Note:
    Many functions are superseded by astropy.coordinates for new code.
"""
