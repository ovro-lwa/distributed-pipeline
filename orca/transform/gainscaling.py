from typing import Tuple

import numpy as np
from casacore.tables import table


def auto_corr_data_and_flag(ms: str, data_column: str) -> Tuple[np.ndarray, np.ndarray]:
    with table(ms) as t:
        t_cross = t.query('ANTENNA1==ANTENNA2')
        data = t_cross.getcol(data_column)
        flag = t_cross.getcol('FLAG')
    return data, flag


def calculate_gain_scale(baseline_ms: str, target_ms: str, data_column: str = 'CORRECTED_DATA'):
    baseline_data, baseline_flag = auto_corr_data_and_flag(baseline_ms, data_column)
    target_data, target_flag = auto_corr_data_and_flag(target_ms, data_column)
    return np.sqrt(np.where(baseline_flag, np.nan, baseline_data)/np.where(target_flag, np.nan, target_data))


def apply_gain_scale(data: np.ndarray, scale_spectrum: np.ndarray) -> np.ndarray:
    """
    Apply single pol scaling factor per antenna to cross-correlated data.
    :param data: Cross-correlated data with shape (N_vis, N_chan, 4), ordered by antennas
    :param scale_spectrum: Single polarization scaling factors; shape (N_ant, N_chan, 2), ordered by antennas
    :return: Scaled data with
    """
    # repeat the input scale_spectrum many times and then do two vector multiplications
    if not (data.shape[1] == scale_spectrum.shape[1] and
            data.shape[2] == 4 and
            scale_spectrum.shape[2] == 2 and
            data.shape[0] == scale_spectrum.shape[0] * (scale_spectrum.shape[0] - 1) // 2
            + scale_spectrum.shape[0]):
        raise ValueError(f'Incompatible shapes for data {data.shape} and scale_spectrum {scale_spectrum.shape}')

    # Generate pairs of antenna indices corresponding to the visibility antenna pair ordering
    upper_triangle_matrix_indices = np.triu_indices(scale_spectrum.shape[0])
    # This computes the outer product along the last axis
    data *= (scale_spectrum[upper_triangle_matrix_indices[0], :, :, np.newaxis] *
             scale_spectrum[upper_triangle_matrix_indices[1], :, np.newaxis, :]).reshape(data.shape)
    return data


def correct_scaling(baseline_ms: str, target_ms: str, data_column: str = 'CORRECTED_DATA'):
    scale_spectrum = calculate_gain_scale(baseline_ms, target_ms, data_column)
    with table(target_ms, readonly=False) as t:
        data = apply_gain_scale(t.getcol(data_column), scale_spectrum[:, :, (0, 3)])
        t.putcol(data_column, data)
