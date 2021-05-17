import pytest
from mock import patch
from os import path
from datetime import datetime, timedelta

from orca.extra.sifting import OfflineSifter
from orca.metadata.pathsmanagers import OfflinePathsManager, SIDEREAL_DAY


@patch('orca.extra.sifting.SiftingWidget.__init__')
def test_init_offlinesifter(super_init):
    super_init.return_value = None
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             working_dir='/tmp/')
    OfflineSifter(pm, datetime(2019, 10, 28, 23, 2, 47), datetime(2019, 10, 28, 23, 3, 52), timedelta(seconds=13),
                  SIDEREAL_DAY,
                  'sidereal_narrow_diff', 'sidereal_long/before', 'sidereal_long/after', '_sfind_sift.fits')
    catalogs, diff_ims, before_ims, after_ims, outputs = super_init.call_args.args
    assert len(catalogs) == 5
    assert len(diff_ims) == 5
    assert len(before_ims) == 5
    assert len(after_ims) == 5
    assert len(outputs) == 5
    assert catalogs[0] == '/tmp//sidereal_narrow_diff/2019-10-28/hh=23/diff_2019-10-28T23:02:47_sfind.fits'
    assert diff_ims[0] == '/tmp//sidereal_narrow_diff/2019-10-28/hh=23/diff_2019-10-28T23:02:47.fits'
    assert before_ims[0] == '/tmp//sidereal_long/before/2019-10-28/hh=23/2019-10-28T23:02:47.fits'
    assert after_ims[0] == '/tmp//sidereal_long/after/2019-10-29/hh=22/2019-10-29T22:58:51.fits'
    assert outputs[0] == '/tmp//sidereal_narrow_diff/2019-10-28/hh=23/diff_2019-10-28T23:02:47_sfind_sift.fits'
