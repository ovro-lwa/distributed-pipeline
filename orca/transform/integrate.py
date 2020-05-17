from typing import List, Optional
import os

import numpy as np
from casatasks import concat
from casacore.tables import table

from orca.wrapper import change_phase_centre


def integrate(ms_list: List[str], out_ms: str, phase_center: Optional[str]) -> str:
    """
    "integrate" a list of ms (assumed to be time sorted) by running chgcentre, concat, and then
    changing the field id. Common phase center is defaulted to the phase center of the scan in the middle of the list.
    :param ms_list:
    :param out_ms:
    :param phase_center:
    :return:
    """
    # error out if out_ms exist.
    if os.path.isfile(out_ms):
        raise FileExistsError(f"Can't concat when output ms {out_ms} already exists.")
    phase_center = phase_center if phase_center else change_phase_centre.get_phase_center(ms_list[len(ms_list)//2])
    for ms in ms_list:
        change_phase_centre.change_phase_center(ms, phase_center)
    concat(ms_list, out_ms)
    with table(out_ms, readonly=False) as t:
        fid = t.getcol('FIELD_ID')
        fid_new = np.zeros(fid.shape, dtype=int)
        t.putcol('FIELD_ID', fid_new)
    return out_ms
