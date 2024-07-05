import numpy as np

def core_outrigger_slices(ant_num_arr, outriggers):
    assert ant_num_arr.ndim == 1, 'ant_num_arr must be 1D.'
    outrigger_slice = np.isin(ant_num_arr, outriggers)
    core_slice = ~outrigger_slice
    return core_slice, outrigger_slice
