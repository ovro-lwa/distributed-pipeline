import pytest
import numpy as np

from astropy import wcs, coordinates, time, units as u

from orca.transform import imaging
from orca.utils import fitsutils


def test_get_peak_around_source():
    x_ans = 1157
    y_ans = 2568
    im = np.zeros(shape=(4096, 4096))
    im[x_ans, y_ans] = 6
    xp, yp = imaging.get_peak_around_source(im, wcs.utils.pixel_to_skycoord(x_ans, y_ans,
                                                                            wcs.WCS(fitsutils.get_sample_header())),
                                            wcs.WCS(fitsutils.get_sample_header()))
    assert xp == x_ans
    assert yp == y_ans



"""
TODO mock orca.wrapper.wsclean, spy on the args and then check them.
"""