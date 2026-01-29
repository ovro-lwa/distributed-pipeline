"""Angular coordinate representation and conversion.

This module provides a simple AngularCoordinate class for parsing and
formatting sexagesimal coordinate strings. An astropysics.coords equivalent.

Note:
    For new code, prefer astropy.coordinates.Angle or SkyCoord.
"""
from __future__ import division
from math import *
import re
import numpy as np


class AngularCoordinate:
    """Angular coordinate with sexagesimal parsing and formatting.

    Parses coordinate strings in HH:MM:SS (RA) or +/-dd:mm:ss (Dec)
    format and provides conversion to degrees and formatted output.

    Attributes:
        pos: Original position value (string or numeric).
        angtype: Type of angle ('ra' or 'dec').

    Example:
        >>> coord = AngularCoordinate('12:30:00')
        >>> coord.d  # Returns degrees
        187.5
    """
    def __init__(self,pos):
        self.pos = pos
        if isinstance(pos,str):     # 'HH:MM:SS' or '+/-dd:mm:ss'
            if (pos[0] == '+') or (pos[0] == '-'):
                self.angtype = 'dec'
            else:
                self.angtype = 'ra'

    @property
    def d(self):
        """Convert coordinate to degrees.

        Returns:
            Position in degrees. For RA input, converts hours to degrees.
        """
        # separate string
        posarr = re.split('[:]',self.pos)
        if self.angtype == 'dec':
            dval = np.abs(float(posarr[0])) + 1/60*float(posarr[1]) + 1/60*1/60*float(posarr[2])
            if posarr[0][0] == '+':
                return dval
            elif posarr[0][0] == '-':
                return dval*-1.0
        elif self.angtype == 'ra':
            dval = 15 * (float(posarr[0]) + 1/60*float(posarr[1]) + 1/60*1/60*float(posarr[2]))
            return dval

    @property
    def dms(self):
        """Format numeric position as degrees/arcmin/arcsec string.

        Returns:
            Formatted string in 'DDdMMmSS.Ss' format.
        """
        dg = int(self.pos)
        mi = int((self.pos - dg) * 60.)
        se = (((self.pos - dg) * 60.) - mi) * 60
        return '%02dd%02dm%04.1fs' % (dg, abs(mi), abs(se))

    @property
    def hms(self):
        """Format numeric position as hours/min/sec string.

        Returns:
            Formatted string in 'HHhMMmSS.Ss' format.
        """
        if self.pos < 0:
            self.pos += 360
        valhr = self.pos / 15.
        hr = int(valhr)
        mi = int((valhr - hr) * 60.)
        se = (((valhr - hr) * 60.) - mi) * 60
        return '%02dh%02dm%04.1fs' % (hr, abs(mi), abs(se))
