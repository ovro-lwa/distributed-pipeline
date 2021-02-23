import pytest

from ..common import TEST_MS
from orca.proj import transientbatchtasks
import os
import shutil


def test_flag_frac(tmpdir):
    ms = f'{tmpdir}/2018-03-23T03:26:18/14_2018-03-23T03:26:18.ms'
    os.makedirs(os.path.dirname(ms), exist_ok=True)
    shutil.copytree(TEST_MS, ms)
    f = transientbatchtasks._max_flag_frac(os.path.dirname(ms), ['14'])
    assert f > 0

