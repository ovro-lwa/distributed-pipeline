"""Transforms that relate to amplitude scaling.
"""
from typing import Tuple

import numpy as np
from casacore.tables import table

import logging

log = logging.getLogger(__name__)


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


def apply_gain_scale_in_place(data: np.ndarray, scale_spectrum: np.ndarray) -> None:
    """ Apply single pol scaling factor per antenna to cross-correlated data.
    This is similar to applycal in CASA. It multiples a cross-correlation by the scaling factor that corresponds to
    the two antennas (each of which has 2 polarizations) involved.
    Warning: this mutates data in place while returning a reference to it.

    Args:
        data: Cross-correlated data with shape (N_vis, N_chan, 4), ordered by antennas
        scale_spectrum: Single polarization scaling factors; shape (N_ant, N_chan, 2), ordered by antennas

    Returns: data multiplied by scale_spectrum

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


def correct_scaling(baseline_ms: str, target_ms: str, data_column: str = 'CORRECTED_DATA'):
    """ Correct for per-antenna per-pol per-channel scaling between two measurement sets.
    Scales data in target_ms such that the autocorrelation for baseline_ms and target_ms are the same.

    Args:
        baseline_ms: Measurement set to scale to
        target_ms: Measurement set that this function modifies so that the autocorr is the same as baseline_ms
        data_column: The data column to apply this operation to.

    Returns:

    """
    scale_spectrum = calculate_gain_scale(baseline_ms, target_ms, data_column)
    with table(target_ms, readonly=False, ack=False) as t:
        data = t.getcol(data_column)
        apply_gain_scale_in_place(data, scale_spectrum[:, :, (0, 3)])
        t.putcol(data_column, data)
