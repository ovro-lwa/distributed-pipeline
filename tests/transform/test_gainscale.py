import pytest
from mock import patch

import numpy as np
from pytest import approx
from orca.transform import gainscaling


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


@patch('orca.transform.gainscaling.auto_corr_data_and_flag')
def test_calculate_gain_scale(auto_corr_data_and_flag):
    test_shape = (10, 6, 4)
    test_scale = 0.3
    test_baseline = np.random.rand(*test_shape)
    # baseline x gain**2 = target
    test_target = test_baseline / (test_scale ** 2)
    flag_arr1 = np.full(test_shape, False)
    flag_arr2 = np.full(test_shape, False)
    flag_arr1[0, 4, 3] = True
    flag_arr1[7, 1, 2] = True
    expected = np.full(test_shape, test_scale)
    expected[0, 4, 3] = 1.
    expected[7, 1, 2] = 1.

    # return a different value on each call
    auto_corr_data_and_flag.side_effect = [(test_baseline, flag_arr1), (test_target, flag_arr2)]

    ans = gainscaling.calculate_gain_scale('', '')
    assert ans == approx(expected)
