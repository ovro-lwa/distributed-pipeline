"""Make a phased measurement set out of many measurement sets.
"""
from typing import List, Optional
import os
import logging

import numpy as np
from casatasks import concat, virtualconcat
from casacore.tables import table

from orca.wrapper import change_phase_centre

log = logging.getLogger(__name__)


def integrate(ms_list: List[str], out_ms: str, phase_center: Optional[str] = None,
              use_virtualconcat: bool = False) -> str:
    """Integrate a list of ms (assumed to be time sorted)
    chgcentre, concat, and then changing the field id.
    Common phase center is defaulted to the phase center of the first scan in the list.

    Args:
        ms_list: List of (time-sorted) measurement sets to integrate.
        out_ms: output measurement set path.
        phase_center: Phase center for the integrated ms, a string like '09h18m05.8s -12d05m44s'
        use_virtualconcat: Whether to use virtualconcat, which MOVES the data to create a multi-ms. The original data
            file will not stay there.

    Returns: Path to integrated measurement set.

    """
    if os.path.exists(out_ms):
        raise FileExistsError(f"Can't concat when output ms {out_ms} already exists.")
    phase_center = phase_center if phase_center else change_phase_centre.get_phase_center(ms_list[0])
    for ms in ms_list:
        change_phase_centre.change_phase_center(ms, phase_center)
    if use_virtualconcat:
        log.warning('If virtualconcat does not work. See https://github.com/ovro-lwa/distributed-pipeline/issues/20')
        virtualconcat(ms_list, out_ms)
    else:
        concat(ms_list, out_ms)
    with table(out_ms, readonly=False) as t:
        fid = t.getcol('FIELD_ID')
        fid_new = np.zeros(fid.shape, dtype=int)
        t.putcol('FIELD_ID', fid_new)
    return out_ms
