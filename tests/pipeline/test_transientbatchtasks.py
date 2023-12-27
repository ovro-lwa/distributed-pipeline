import pytest

from ..common import TEST_MS
from orca.pipeline import transientbatchtasks
import os
import shutil


def test_flag_frac(tmpdir):
    ms = f'{tmpdir}/2018-03-23T03:26:18/14_2018-03-23T03:26:18.ms'
    os.makedirs(os.path.dirname(ms), exist_ok=True)
    shutil.copytree(TEST_MS, ms)
    f = transientbatchtasks._max_flag_frac(os.path.dirname(ms), ['14'])
    assert f > 0


@pytest.mark.parametrize('p', [
    '/lustre//narrow/after/2018-03-23/hh=04/2018-03-23T04:04:13',
    '/lustre//narrow/after/2018-03-23/hh=04/2018-03-23T04:04:13/'
])
def test_ms_parent_to_product(p):
    p = '/lustre//narrow/after/2018-03-23/hh=04/2018-03-23T04:04:13'
    assert transientbatchtasks._ms_parent_to_product(p,
                                             '/lustre/output/', '.fits', '') == \
           '/lustre/output//2018-03-23/hh=04/2018-03-23T04:04:13.fits'
    assert transientbatchtasks._ms_parent_to_product(p,
                                              '/lustre/output/', '.fits', 'diff_') == \
           '/lustre/output//2018-03-23/hh=04/diff_2018-03-23T04:04:13.fits'
