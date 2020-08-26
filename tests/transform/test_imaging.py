import pytest
from mock import patch

from os import path

import numpy as np
from datetime import datetime
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


@pytest.mark.parametrize('timestamp, expected_subtract_source_count, more_args, extra_args_check', [
    (datetime(2018, 3, 22, 2, 7, 0), 1, None, lambda extra_args: '-channelsout' not in extra_args),  # Crab only
    (datetime(2018, 3, 22, 16, 30, 0), 1, None, lambda extra_args: '-channelsout' in extra_args),  # Sun only
    (datetime(2018, 9, 22, 16, 30, 0), 2, None, lambda extra_args: '-channelsout' in extra_args),  # Crab and Sun
    (datetime(2018, 9, 22, 2, 30, 0), 0, None, lambda extra_args: '-channelsout' not in extra_args),  # no Crab or Sun
    (datetime(2018, 9, 22, 2, 30, 0), 0, ['-beep'],
        lambda extra_args: '-channelsout' not in extra_args and '-beep' in extra_args),  # no Crab or Sun
])
@patch('orca.utils.fitsutils.write_fits_mask_with_box_xy_coordindates')
@patch('orca.transform.imaging.get_peak_around_source')
@patch('orca.transform.imaging.make_dirty_image')
@patch('os.renames')
@patch('orca.wrapper.wsclean.wsclean')
def test_make_residual_image_with_source_removed(wsclean, os_renames, make_dirty_image, get_peak_around_source,
                                                 write_fits_mask_with_box_xy_coordindates,
                                                 timestamp, expected_subtract_source_count, more_args,
                                                 extra_args_check):
    # Set up mocked functions since they're initialized lazily
    wsclean.return_value = None
    os_renames.return_value = None
    get_peak_around_source.return_value = (2000, 2000)
    write_fits_mask_with_box_xy_coordindates.return_value = '/some/fits/path/some-fits.fits'
    # Actual fits file but the details of this file does not matter
    make_dirty_image.return_value = f'{path.dirname(__file__)}/../resources/2018-03-22T02:07:54-dirty.fits'

    # Actual run
    if more_args:
        imaging.make_residual_image_with_source_removed(['beep'], timestamp, output_dir='/tmp/',
                                                        output_prefix='blah',
                                                        tmp_dir='/tmp/')
    else:
        imaging.make_residual_image_with_source_removed(['beep'], timestamp, output_dir='/tmp/',
                                                        output_prefix='blah',
                                                        tmp_dir='/tmp/', more_args=more_args)
    # Verify
    if expected_subtract_source_count:
        assert len(write_fits_mask_with_box_xy_coordindates.call_args.kwargs['center_list']) == \
               expected_subtract_source_count
        assert len(write_fits_mask_with_box_xy_coordindates.call_args.kwargs['width_list']) == \
               expected_subtract_source_count
        print(wsclean.call_args.kwargs['extra_arg_list'])
        assert extra_args_check(wsclean.call_args.kwargs['extra_arg_list'])
    else:
        write_fits_mask_with_box_xy_coordindates.assert_not_called()
        # since I mocked make_dirty_image() the wsclean called within that didn't count
        wsclean.assert_not_called()


@patch('orca.wrapper.wsclean.wsclean')
def test_make_dirty_image(wsclean):
    wsclean.return_value = None

    assert imaging.make_dirty_image(['1.ms', '2.ms'], '/tmp/', 'beep') == '/tmp//beep-image.fits'
    assert wsclean.call_args.kwargs['extra_arg_list'][:5] == ['-size', str(imaging.IMSIZE), str(imaging.IMSIZE),
                                                              '-scale', str(imaging.IM_SCALE_DEGREE)]


@patch('orca.wrapper.wsclean.wsclean')
def test_make_dirty_image_and_psf(wsclean):
    wsclean.return_value = None

    assert imaging.make_dirty_image(['1.ms', '2.ms'], '/tmp/', 'beep', make_psf=True) ==\
           ('/tmp//beep-image.fits', '/tmp//beep-psf.fits')
    assert wsclean.call_args.kwargs['extra_arg_list'][:5] == ['-size', str(2 * imaging.IMSIZE), str(2 * imaging.IMSIZE),
                                                              '-scale', str(imaging.IM_SCALE_DEGREE)]
