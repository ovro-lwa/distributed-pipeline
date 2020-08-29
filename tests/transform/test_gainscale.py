import pytest
from mock import patch

import shutil

import numpy as np
from pytest import approx
from orca.transform import gainscaling
from casacore.tables import table

from ..common import TEST_MS


def test_apply_gain_scale():
    n_ant = 10
    n_chan = 40
    data = np.random.rand(n_ant * (n_ant - 1) // 2 + n_ant, n_chan, 4)
    scale_spectrum = np.random.rand(n_ant, n_chan, 2)
    expected = np.zeros_like(data)
    # Calculate the answer the slow way
    aggregate_index = 0
    for i in range(n_ant):
        for j in range(i, n_ant):
            expected[aggregate_index, :, 0] = data[aggregate_index, :, 0] * scale_spectrum[i, :, 0] * \
                                              scale_spectrum[j, :, 0]
            expected[aggregate_index, :, 1] = data[aggregate_index, :, 1] * scale_spectrum[i, :, 0] * \
                                              scale_spectrum[j, :, 1]
            expected[aggregate_index, :, 2] = data[aggregate_index, :, 2] * scale_spectrum[i, :, 1] * \
                                              scale_spectrum[j, :, 0]
            expected[aggregate_index, :, 3] = data[aggregate_index, :, 3] * scale_spectrum[i, :, 1] * \
                                              scale_spectrum[j, :, 1]
            aggregate_index += 1
    gainscaling.apply_gain_scale_in_place(data, scale_spectrum)
    assert data == approx(expected)


def test_calculate_gain_scale():
    test_shape = (10, 6, 4)
    test_scale = 0.3
    test_to_scale = np.random.rand(*test_shape)
    # baseline x gain**2 = target
    test_target = test_to_scale * (test_scale ** 2)
    flag_arr1 = np.full(test_shape, False)
    flag_arr2 = np.full(test_shape, False)
    flag_arr1[0, 4, 3] = True
    flag_arr1[7, 1, 2] = True
    expected = np.full(test_shape, test_scale)
    expected[0, 4, 3] = 1.
    expected[7, 1, 2] = 1.

    ans = gainscaling.calculate_gain_scale(test_to_scale, flag_arr1, test_target, flag_arr2)
    assert np.all((ans - expected)/expected < 1e-7)


def test_correct_scaling(tmp_path):
    ms_1 = (tmp_path / 'test1.ms').as_posix()
    ms_2 = (tmp_path / 'test2.ms').as_posix()
    scale = 2.
    shutil.copytree(TEST_MS, ms_1)
    shutil.copytree(TEST_MS, ms_2)

    with table(ms_2, readonly=False, ack=False) as t:
        c = t.getcol('DATA')
        t.putcol('DATA', c * scale)
    gainscaling.correct_scaling(to_scale_ms=ms_1, target_ms=ms_2, data_column='DATA')

    with table(ms_2, ack=False) as t:
        assert np.all((np.abs(t.getcol('DATA') - scale * c)) < 1e-16)

    with table(ms_1, ack=False) as t:
        flags = t.getcol('FLAG')
        dat = t.getcol('DATA')
        # No nans where there is valid data
        assert np.all(~np.isnan(np.where(flags, 1., dat)))
        # relative precision to 1e-5
        assert np.all(np.where(flags, True, np.abs(dat - c * scale) < (1e-6 * np.abs(dat))))
