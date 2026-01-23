"""Pre-calibration flagging based on autocorrelation power.

Identifies dead or malfunctioning antennas before calibration by
analyzing autocorrelation power levels relative to the array median.

Functions
---------
is_fft_overflow
    Check for FFT overflow conditions (not yet implemented).
find_dead_ants
    Identify antennas with abnormally low power.
"""
from typing import List
import numpy as np
import logging

from casacore.tables import table

from orca.configmanager import telescope

logger = logging.getLogger(__name__)

DEAD_POWER_THRESHOLD = 0.1


def is_fft_overflow(ms: str) -> bool:
    """Check if the measurement set shows signs of FFT overflow.

    Args:
        ms: Path to the measurement set.

    Returns:
        True if overflow detected, False otherwise.

    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError


def find_dead_ants(ms: str) -> List[int]:
    """Identify antennas with abnormally low power.

    Compares each antenna's total autocorrelation power to the array
    median for core and outrigger antennas separately.

    Args:
        ms: Path to the measurement set.

    Returns:
        Sorted list of antenna indices with power below threshold.
    """
    with table(ms, ack=False) as t:
        t_auto = t.query('ANTENNA1=ANTENNA2')
        ants = t_auto.getcol('ANTENNA1')
        autocorr = t_auto.getcol('DATA')[:,:,(0,3)].real
    
    assert np.all(np.diff(ants) >= 0) and len(ants) == autocorr.shape[0], 'Data not indexed by antenna number.'
    is_outrigger = np.zeros(len(ants), dtype=bool)
    is_outrigger[telescope.outriggers] = True
    n_outriggers = np.sum(is_outrigger)
    powers = autocorr.sum(axis=(1))
    total_power = powers.sum()
    total_outrigger_power = powers[is_outrigger].sum()
    mean_outrigger_power = total_outrigger_power / n_outriggers
    mean_core_power = (total_power - total_outrigger_power) / (len(ants) - n_outriggers)
    bad_cores = []
    bad_outriggers = []
    for i, amp in enumerate(powers):
        if is_outrigger[i]:
            if (amp < DEAD_POWER_THRESHOLD * mean_outrigger_power).any():
                bad_outriggers.append(i)
        else:
            if (amp < DEAD_POWER_THRESHOLD * mean_core_power).any():
                bad_cores.append(i)

    logger.info(f'Found {len(bad_cores)} dead core antennas and {len(bad_outriggers)} dead outriggers.')
    return sorted(bad_cores + bad_outriggers)
