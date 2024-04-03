from typing import List
import numpy as np
import logging

from casacore.tables import table

from orca.configmanager import telescope

logger = logging.getLogger(__name__)

DEAD_POWER_THRESHOLD = 0.1

def get_bad_ants(ms: str) -> List[int]:
    """ Get list of dead antennas.
    """
    with table(ms, ack=False) as t:
        t_auto = t.query('ANTENNA1=ANTENNA2')
        ants = t_auto.getcol('ANTENNA1')
        autocorr = t_auto.getcol('DATA')[:,:,0,3].real
    
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
            if amp < DEAD_POWER_THRESHOLD * mean_outrigger_power:
                bad_outriggers.append(i)
        else:
            if amp < DEAD_POWER_THRESHOLD * mean_core_power:
                bad_cores.append(i)

    logger.info(f'Found {len(bad_cores)} bad core antennas and {len(bad_outriggers)} bad outriggers.')
    return bad_cores + bad_outriggers
