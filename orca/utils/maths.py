"""Mathematical helper functions.

Provides utility functions for array slicing and mathematical operations
commonly used in radio interferometry data processing.
"""
import numpy as np


def core_outrigger_slices(ant_num_arr: np.ndarray, outriggers: list) -> tuple:
    """Create boolean slices for core and outrigger antennas.

    Args:
        ant_num_arr: 1D array of antenna numbers.
        outriggers: List of outrigger antenna numbers.

    Returns:
        Tuple of (core_slice, outrigger_slice) boolean arrays.
    """
    assert ant_num_arr.ndim == 1, 'ant_num_arr must be 1D.'
    outrigger_slice = np.isin(ant_num_arr, outriggers)
    core_slice = ~outrigger_slice
    return core_slice, outrigger_slice
