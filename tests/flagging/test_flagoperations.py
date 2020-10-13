import pytest
import tempfile
import shutil
from os import path

from casacore.tables import table
import numpy as np

from ..common import TEST_MS

from orca.flagging import flagoperations

MS_NAME = path.basename(TEST_MS)

with table(TEST_MS, readonly=True, ack=False) as t:
    FLAG_COL_SHAPE = t.getcol('FLAG').shape


def test_merge_flags(tmpdir):
    ms1 = f'{tmpdir}/{MS_NAME}'
    ms2 = f'{tmpdir}/2_{MS_NAME}'
    flag1 = random_flag_ms(ms1)
    flag2 = random_flag_ms(ms2)
    flagoperations.merge_flags(ms1, ms2)
    with table(ms1, readonly=False, ack=False) as t1, table(ms2, readonly=False, ack=False) as t2:
        assert np.all(t1.getcol('FLAG') == t2.getcol('FLAG'))
        # True in random_flags must be True in the flag column
        assert np.all(t1.getcol('FLAG') >= flag2)
        assert np.all(t1.getcol('FLAG') >= flag1)


def test_merged_group_flags(tmpdir):
    n_ms = 4
    ms_list = []
    random_flags_list = []
    for i in range(n_ms):
        ms = f'{tmpdir}/{i}_{MS_NAME}'
        random_flags = random_flag_ms(ms)
        ms_list.append(ms)
        random_flags_list.append(random_flags)

    flagoperations.merge_group_flags(ms_list)

    for i in range(1, n_ms):
        with table(ms_list[i], ack=False) as t1, table(ms_list[i-1]) as t2:
            assert np.all(t1.getcol('FLAG') == t2.getcol('FLAG'))
    with table(ms_list[0], ack=False) as t0:
        for f in random_flags_list:
            print(np.all(t0.getcol('FLAG') >= f))


def random_flag_ms(ms):
    shutil.copytree(TEST_MS, ms)
    random_flags = np.random.rand(*FLAG_COL_SHAPE) > 0.5
    with table(ms, readonly=False, ack=False) as t:
        t.putcol('FLAG', random_flags)
    return random_flags

