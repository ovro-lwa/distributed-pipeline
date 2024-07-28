import pytest
import tempfile

import numpy as np
from astropy import wcs

from orca.utils import fitsutils
from ..common import TEST_FITS

def test_get_sample_header():
    assert fitsutils.get_sample_header()['SIMPLE']


def test_sample_header_to_wcs():
    from astropy.wcs import WCS
    WCS(fitsutils.get_sample_header())


def test_write_fits_mask_with_box():
    with tempfile.NamedTemporaryFile() as f:
        fitsutils.write_fits_mask_with_box_xy_coordindates(f.name, 4096, [(1032, 700)], [3])
        im, header = fitsutils.read_image_fits(f.name)
        assert np.all(im.T[1031: 1034, 699:702] == 1.)
        assert np.sum(im.T == 0.) == 4096**2 - 9


def test_write_fits_mask_with_multiple_boxes():
    with tempfile.NamedTemporaryFile() as f:
        fitsutils.write_fits_mask_with_box_xy_coordindates(f.name, 4096, [(1032, 700), (400, 600)], [3, 81])
        im, header = fitsutils.read_image_fits(f.name)
        assert np.all(im.T[1031: 1034, 699:702] == 1.)
        assert np.all(im.T[360: 441, 560: 641] == 1.)
        assert np.sum(im.T == 0.) == 4096 ** 2 - 3 * 3 - 81 * 81


def test_coadd():
    with tempfile.NamedTemporaryFile() as f:
        fitsutils.co_add([TEST_FITS] * 51, f.name)
        av_im, _ = fitsutils.read_image_fits(f.name)
        im, _ = fitsutils.read_image_fits(TEST_FITS)
        np.array_equal(av_im, im)

def test_get_peak_around_source():
    x_ans = 1157
    y_ans = 2568
    im = np.zeros(shape=(4096, 4096))
    im[x_ans, y_ans] = 6
    xp, yp = fitsutils.get_peak_around_src(im, wcs.utils.pixel_to_skycoord(x_ans, y_ans,
                                                                            wcs.WCS(fitsutils.get_sample_header())),
                                            wcs.WCS(fitsutils.get_sample_header()))
    assert xp == x_ans
    assert yp == y_ans