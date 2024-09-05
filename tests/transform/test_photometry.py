import pytest

from astropy.coordinates import SkyCoord
import numpy as np

from orca.utils import fitsutils

from orca.transform import photometry

DATADIR = '/lustre/yuping/orca-test-resource/coadd-test/'

def test_average_with_no_rms_threshold():
    photometry.average_with_rms_threshold(
        [DATADIR + 'test2.fits', DATADIR + 'test3.fits', DATADIR + 'test100.fits'],
        DATADIR + 'out.fits',
        SkyCoord('11h45m11s', '+2d18m40s'),
        500, np.inf)
    im, header = fitsutils.read_image_fits(DATADIR + 'out.fits')
    assert im[1000, 2000] == float(3 + 100 + 2) / 3


def test_average_with_rms_threshold():
    photometry.average_with_rms_threshold(
        [DATADIR + 'test2.fits', DATADIR + 'test3.fits', DATADIR + 'test100.fits'],
        DATADIR + 'out.fits',
        SkyCoord('11h45m11s', '+2d18m40s'),
        500, 2)
    im, header = fitsutils.read_image_fits(DATADIR + 'out.fits')
    assert im[1000, 2000] == float(3 + 2) / 2