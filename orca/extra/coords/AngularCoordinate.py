from __future__ import division
from math import *
import re
import numpy as np

class AngularCoordinate:
    '''
    An astropysics.coords.AngularCoordinate equivalent.
    Last edit: 21 September 2017
    '''
    def __init__(self,pos):
        self.pos = pos
        if isinstance(pos,str):     # 'HH:MM:SS' or '+/-dd:mm:ss'
            if (pos[0] == '+') or (pos[0] == '-'):
                self.angtype = 'dec'
            else:
                self.angtype = 'ra'

    @property
    def d(self):
        '''
        A coords.AngularCoordinate('HH:MM:SS').d equivalent.
        OUTPUT: position in degrees.
        '''
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
        '''
        A coords.AngularCoordinate(xxx.x).dms equivalent.
        OUTPUT: position in string format.
        '''
        dg = int(self.pos)
        mi = int((self.pos - dg) * 60.)
        se = (((self.pos - dg) * 60.) - mi) * 60
        return '%02dd%02dm%04.1fs' % (dg, abs(mi), abs(se))

    @property
    def hms(self):
        '''
        A coords.AngularCoordinate(xxx.x).hms equivalent.
        OUTPUT: position in string format.
        '''
        if self.pos < 0:
            self.pos += 360
        valhr = self.pos / 15.
        hr = int(valhr)
        mi = int((valhr - hr) * 60.)
        se = (((valhr - hr) * 60.) - mi) * 60
        return '%02dh%02dm%04.1fs' % (hr, abs(mi), abs(se))
