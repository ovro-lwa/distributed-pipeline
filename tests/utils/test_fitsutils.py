import pytest
import tempfile

import numpy as np

from orca.utils import fitsutils


def test_get_sample_header():
    assert fitsutils.get_sample_header()['SIMPLE']


def test_write_fits_mask_with_box():
    with tempfile.NamedTemporaryFile() as f:
        fitsutils.write_fits_mask_with_box(f.name, 4096, (1032, 700), 3)
        im, header = fitsutils.read_image_fits(f.name)
        assert np.all(im[1031: 1034, 699:702] == 1.)
        assert np.sum(im == 0.) == 4096**2 - 9
