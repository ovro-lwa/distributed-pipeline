import pytest

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
    actual = gainscaling.apply_gain_scale(data, scale_spectrum)
    assert actual == approx(expected)

