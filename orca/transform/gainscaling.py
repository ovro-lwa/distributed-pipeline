"""Transforms that relate to amplitude scaling.

It uses autocorrelation to figure out the scaling factor between two snapshots on a per-antenna per-channel per-pol
basis.

NOTE: It uses the autocorrelation  flags to figure out which antennas are flagged and does not solve
for those antennas.
"""
from typing import Tuple

import numpy as np
from casacore.tables import table

import logging

log = logging.getLogger(__name__)


def auto_corr_data_and_flag(t: table, data_column: str) -> Tuple[np.ndarray, np.ndarray]:
    t_auto = t.query('ANTENNA1==ANTENNA2')
    data = t_auto.getcol(data_column)
    flag = t_auto.getcol('FLAG')
    return data, flag


def calculate_gain_scale(to_scale_data: np.array, to_scale_flag: np.array, target_data: np.array, target_flag: np.array):
    """
    Calcualte the gain scaling factor required to scale to_scale to target.

    Args:
        to_scale_data:
        to_scale_flag:
        target_data:
        target_flag:

    Returns:

    """
    quotient = np.sqrt(target_data/to_scale_data)
    return np.where(np.logical_or(to_scale_flag, target_flag), 1., quotient)


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


def correct_scaling(to_scale_ms: str, target_ms: str, data_column: str = 'CORRECTED_DATA'):
    """ Correct for per-antenna per-pol per-channel scaling between two measurement sets.
    Scales data in target_ms such that the autocorrelation for baseline_ms and target_ms are the same.

    Args:
        to_scale_ms: Measurement set to scale to
        target_ms: Measurement set that this function modifies so that the autocorr is the same as baseline_ms
        data_column: The data column to apply this operation to.

    Returns:

    """
    log.info(f'Applying gainscaling change to {to_scale_ms}.')
    with table(target_ms, ack=False) as target:
        target_auto, target_flag = auto_corr_data_and_flag(target, data_column)

    with table(to_scale_ms, readonly=False, ack=False) as to_scale:
        to_scale_auto, to_scale_flag = auto_corr_data_and_flag(to_scale, data_column)
        scale_spectrum = calculate_gain_scale(to_scale_auto, to_scale_flag, target_auto, target_flag)
        to_scale_data = to_scale.getcol(data_column)
        apply_gain_scale_in_place(to_scale_data, scale_spectrum[:, :, (0, 3)])
        to_scale.putcol(data_column, to_scale_data)
