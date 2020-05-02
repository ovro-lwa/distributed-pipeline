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


def test_astropy_get_sun():
    t = time.Time('2018-03-23T16:26:18', format='isot', scale='utc')
    sun_coord = coordinates.get_sun(t)
    ans = coordinates.SkyCoord('00h09m23s +1d11m30s', frame=coordinates.ICRS)
    assert sun_coord.separation(ans) < coordinates.Angle('15arcmin')

def test_astropy_never_transform_to_icrs_sun_location():
    """
    This does not work. I think it's because ICRS is barycentric and the Sun's coordinate (in 3D) is too close to
    the coordinate center.
    """
    t = time.Time('2018-03-23T16:26:18', format='isot', scale='utc')
    sun_coord = coordinates.get_sun(t)
    # The Same GCRS but much farther away. This should convert to the "right" ICRS coordindate
    # (i.e. what one reads off an image).
    sun_coord2 = coordinates.SkyCoord(ra=sun_coord.ra, dec=sun_coord.dec, distance=10 * u.pc, frame=coordinates.GCRS)
    sun_coord2_icrs = sun_coord2.transform_to(coordinates.ICRS)
    sun_coord_icrs = sun_coord.transform_to(coordinates.ICRS)
    # These would look very different
    assert sun_coord2_icrs.separation(sun_coord_icrs) > coordinates.Angle('10 deg')
    # But if you use GCRS as the base frame then it should work
    assert sun_coord.separation(sun_coord2_icrs) < coordinates.Angle('5 arcmin')


"""
TODO mock orca.wrapper.wsclean, spy on the args and then check them.
"""