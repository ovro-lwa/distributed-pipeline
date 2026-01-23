"""Five-hour comparison imaging pipeline.

This script integrates multiple measurement sets from the nighttime data
and produces Stokes I/V images for transient analysis. Concatenates
data across a 5-hour observation window for improved sensitivity.
"""
from multiprocessing import Pool

from orca.metadata.stageiii import spws
from orca.transform.imaging import stokes_IV_imaging
from orca.utils import coordutils
from typing import List

from casacore.tables import table
from casatasks import concat
import numpy as np

import glob

from datetime import datetime, timedelta

SCRATCH = '/lustre/yuping/5-hr-comp/imaged-altogether2'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/yuping/5-hr-comp/imaged-altogether'

def integrate(ms_list: List[str], out_ms: str) -> str:
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
    for ms in ms_list:
        with table(f'{ms}/FIELD', ack=False) as t:
            new_dir = t.getcol('PHASE_DIR')
        with table(f'{ms}/SOURCE', readonly=False, ack=False) as tsrc:
            tsrc.putcol('DIRECTION', new_dir[0])
    concat(ms_list, out_ms, timesort=True)

    with table(out_ms, readonly=False, ack=False) as t:
        fid = t.getcol('FIELD_ID')
        fid_new = np.zeros(fid.shape, dtype=int)
        t.putcol('FIELD_ID', fid_new)
    return out_ms

if __name__ == '__main__':
    """
    with Pool(10) as p:
        start_time = datetime(2024, 3, 2, 8, 0, 0)
        dt = timedelta(minutes=15)
        n_hours = 5
        n_chunks = n_hours * 4

        phase_center = coordutils.zenith_coord_at_ovro(
            datetime(2024, 3, 2, 10, 30, 0)
        )
        futures = []
        for i in range(n_chunks):
            end_time = start_time + dt
            for spw in ['50MHz']:
                futures.append(p.apply_async(stokes_IV_imaging, ([spw], start_time, end_time,
                                        NIGHTTIME_DIR, WORK_DIR, SCRATCH),
                                        {'phase_center': phase_center,
                                        'keep_scratch_dir': True, 'partitioned_by_hour': False}))
            start_time += dt
        for f in futures:
            print(f.get())
    """
    ms_l = glob.glob('/fast/yuping/imaged-altogether2/tmp-*/50MHz.ms')
    integrate(ms_l, '/lustre/yuping/5-hr-comp/imaged-altogether/concat/50MHz-integrated.ms')
    """
    with Pool(16) as p:
        results = []
        for spw in spws:
            ms_l = glob.glob(f'{WORK_DIR}/tmp-*/{spw}.ms')
            print(f'found {spw}: {len(ms_l)} measurement sets')
            if len(ms_l) == 0:
                continue
            results.append(p.apply_async(integrate, (ms_l, f'{WORK_DIR}/concat/{spw}-integrated.ms')))
        for f in results:
            print(f.get())
    """
