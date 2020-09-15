import pytest
from orca.transform import image_sub
from orca.utils import fitsutils

import os
import numpy as np
import tempfile

from ..common import TEST_FITS


@pytest.fixture(scope='module')
def empty_image():
    fn = '/tmp/empty.fits'
    im, header = fitsutils.read_image_fits(TEST_FITS)
    fitsutils.write_image_fits(fn, header, np.zeros(shape=im.shape), header)
    yield fn
    # teardown
    os.remove(fn)


def test_image_sub(empty_image):
    with tempfile.TemporaryDirectory() as dir:
        out = image_sub.image_sub(empty_image, TEST_FITS, dir)
        ans, _ = fitsutils.read_image_fits(out)
        expected, _ = fitsutils.read_image_fits(TEST_FITS)
        assert np.all(ans - expected == 0.)


def test_image_sub_list(empty_image):
    with tempfile.TemporaryDirectory() as dir:
        out = image_sub.image_sub([empty_image, empty_image], [TEST_FITS, TEST_FITS], dir)
        ans, _ = fitsutils.read_image_fits(out)
        expected, _ = fitsutils.read_image_fits(TEST_FITS)
        assert np.all((ans - expected) == 0.)
