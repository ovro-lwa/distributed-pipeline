import pytest

import numpy as np
from math import radians, atan2, sin, cos, asin, pi, sqrt, degrees
import math
from orca.utils import fitsutils
from tests.common import TEST_FITS


def test_wcs_zenith_from_marin():
    im, header = fitsutils.read_image_fits(TEST_FITS)
    w = __WCSFromMarin(header)
    check_convert_marin(92.9907752, 37.0295406, w, 2047, 2047)


def check_convert_marin(ra, dec, w, x, y):
    pix = w.sky2pix(ra, dec)
    assert sqrt((pix[0] - x) ** 2 + (pix[1] - y) ** 2) < 0.5
    sky = w.pix2sky(x, y)
    assert sqrt((sky[0] - ra) ** 2 + (sky[1] - dec) ** 2) < 0.01


def test_wcs_low_el_from_marin():
    im, header = fitsutils.read_image_fits(TEST_FITS)
    w = __WCSFromMarin(header)
    check_convert_marin(139.1670454, -12.9494991, w, 758, 975)

def test_wcs_below_horizon_from_marin():
    im, header = fitsutils.read_image_fits(TEST_FITS)
    w = __WCSFromMarin(header)
    assert w.sky2pix(91., -60.) == (np.nan, np.nan)
    sky =  w.pix2sky(90., 2000)
    assert math.isnan(sky[0])
    assert math.isnan(sky[1])

class __WCSFromMarin:
    '''
    From /lustre/mmanders/LWA/modules/WCS/WCS.py by Marin Anderson.

    Copied here to validate WCS with astropy.
    Created for using the World Coordinate System FITS convention with LWA data
    on the ASTM.
    NOT GENERALLY APPLICABLE. Assumes that CTYPE = 'SIN', that is, zenithal slant
    orthographic projection.	[phi_0 = 0, theta_0 = 90, ra_0 = ra_p, dec_0 = dec_p,
                                phi_0 = phi_c, theta_0 = phi_0]
    If pixel is below the horizon (e.g. theta==elevation <= 0, returns nan).
    Source: Calabretta & Greison 2002
            www.aanda.org/articles/aa/pdf/2002/45/aah3860.pdf
    Last edit: 23 April 2015
    '''

    def __init__(self, FITSheader):
        self.CRVAL1 = FITSheader['CRVAL1']  # in degrees
        self.CRVAL2 = FITSheader['CRVAL2']  # in degrees
        self.CDELT1 = FITSheader['CDELT1']
        self.CDELT2 = FITSheader['CDELT2']
        self.CRPIX1 = FITSheader['CRPIX1']
        self.CRPIX2 = FITSheader['CRPIX2']
        self.phi_0 = 0.  # in degrees
        self.theta_0 = 90.  # in degrees

        if self.CRVAL2 >= self.theta_0:
            self.phi_p = 0.
        elif self.CRVAL2 < self.theta_0:
            self.phi_p = 180.

    def sky2pix(self, ra, dec):
        '''
        A pywcs.wcs.sky2pix equivalent.
        INPUT:	ra 	- in decimal degrees
            dec	- in decimal degrees
        OUTPUT:	Corresponding pixel location.
        '''
        # inverse spherical coordinate rotation
        phi = radians(self.phi_p) + atan2(sin(radians(dec)) * cos(radians(self.CRVAL2)) -
                                          cos(radians(dec)) * sin(radians(self.CRVAL2)) * cos(
            radians(ra - self.CRVAL1)),
                                          -cos(radians(dec)) * sin(radians(ra - self.CRVAL1)))
        theta = asin(sin(radians(dec)) * sin(radians(self.CRVAL2)) +
                     cos(radians(dec)) * cos(radians(self.CRVAL2)) * cos(radians(ra - self.CRVAL1)))

        # if ra,dec is below the horizon, return nan
        if (theta * 180. / pi) <= 0:
            py = np.nan
            px = np.nan
        else:
            # native longitude and latitude to projection plane coordinates
            x = 180. / pi * cos(theta) * sin(phi)
            y = -180. / pi * cos(theta) * cos(phi)

            # intermediate world coordinates to pixel coordinates
            # the -1. is to account for python's 0-indexing
            py = x / self.CDELT1 + self.CRPIX1 - 1.
            px = y / self.CDELT2 + self.CRPIX2 - 1.

        return px, py

    def pix2sky(self, px, py):
        '''
        A pywcs.wcs.pix2sky equivalent.
        INPUT:	px	- pixel value in x-dimension
            py	- pixel value in y-dimension
        OUTPUT:	Corresponding ra,dec in decimal degrees
        '''
        # pixel coordinates to intermediate world coordinates
        # the +1. is to account for python's 0-indexing
        x = self.CDELT1 * (py - self.CRPIX1 + 1.)
        y = self.CDELT2 * (px - self.CRPIX2 + 1.)

        # projection plane coordinates to native longitude and latitude
        try:
            thetap = asin(sqrt(-((pi / 180. * x) ** 2. + (pi / 180. * y) ** 2. - 1)))
            try:
                thetam = asin(-sqrt(-((pi / 180. * x) ** 2. + (pi / 180. * y) ** 2. - 1)))
                # both +/- are valid
                if abs(pi / 2. - thetap) > abs(pi / 2. - thetam):
                    theta = thetam
                elif abs(pi / 2. - thetap) < abs(pi / 2. - thetam):
                    theta = thetap
            except ValueError:
                theta = thetap  # only + is valid
        except ValueError:
            try:
                thetam = asin(-sqrt(-((pi / 180. * x) ** 2. + (pi / 180. * y) ** 2. - 1)))
                theta = thetam
            except:  # neither +/- are valid
                theta = np.nan
        phi = atan2(-pi / 180. * y, pi / 180. * x)

        # spherical coordinate rotation
        ra = radians(self.CRVAL1) + atan2(-cos(theta) * sin(phi - radians(self.phi_p)),
                                          sin(theta) * cos(radians(self.CRVAL2)) -
                                          cos(theta) * sin(radians(self.CRVAL2)) * cos(phi - radians(self.phi_p)))
        dec = asin(sin(theta) * sin(radians(self.CRVAL2)) +
                   cos(theta) * cos(radians(self.CRVAL2)) * cos(phi - radians(self.phi_p)))

        return degrees(ra), degrees(dec)