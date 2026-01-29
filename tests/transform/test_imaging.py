import pytest

from mock import patch

from os import path
import glob

import numpy as np
from datetime import datetime
from astropy import wcs, coordinates, time, units as u

from orca.transform import imaging
from orca.utils import fitsutils


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


@pytest.mark.skip(reason="Requires external /lustre/yuping/orca-test-resource with write permissions")
def test_coadd_fits():
    fns = glob.glob('/lustre/yuping/orca-test-resource/??MHz_2024-04-16T*I.image.fits')
    imaging.coadd_fits(fns, '/lustre/yuping/orca-test-resource/coadded.fits')
