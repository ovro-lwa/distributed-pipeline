"""Measurement set integration transforms.

Provides functions to combine multiple measurement sets into a single
integrated measurement set with a common phase center. Used for creating
multi-scan or multi-snapshot datasets for imaging.
"""
from typing import List, Optional
import os
import logging

import numpy as np
from astropy.coordinates import SkyCoord
from casatasks import concat, virtualconcat
from casacore.tables import table

from orca.wrapper import change_phase_centre

from orca.celery import app

log = logging.getLogger(__name__)

@app.task
def integrate(ms_list: List[str], out_ms: str, phase_center: Optional[SkyCoord] = None,
              use_virtualconcat: bool = False) -> str:
    """Integrate a list of ms (assumed to be time sorted)
    chgcentre, concat, and then changing the field id.
    Common phase center is defaulted to the phase center of the first scan in the list.

    Args:
        ms_list: List of (time-sorted) measurement sets to integrate.
        out_ms: output measurement set path.
        phase_center: Phase center for the integrated ms.
        use_virtualconcat: Whether to use virtualconcat, which MOVES the data to create a multi-ms. The original data
            file will not stay there.

    Returns: Path to integrated measurement set.

    """
    if os.path.exists(out_ms):
        raise FileExistsError(f"Can't concat when output ms {out_ms} already exists.")
    phase_center = phase_center if phase_center else change_phase_centre.get_phase_center(ms_list[0])
    for ms in ms_list:
        change_phase_centre.change_phase_center(ms, phase_center.to_string('hmsdms'))
        with table(f'{ms}/FIELD', ack=False) as t:
            new_dir = t.getcol('PHASE_DIR')
        with table(f'{ms}/SOURCE', readonly=False, ack=False) as tsrc:
            tsrc.putcol('DIRECTION', new_dir[0])
    if use_virtualconcat:
        log.warning('If virtualconcat does not work. See https://github.com/ovro-lwa/distributed-pipeline/issues/20')
        virtualconcat(ms_list, out_ms)
    else:
        concat(ms_list, out_ms)
    with table(out_ms, readonly=False, ack=False) as t:
        fid = t.getcol('FIELD_ID')
        fid_new = np.zeros(fid.shape, dtype=int)
        t.putcol('FIELD_ID', fid_new)
    return out_ms
