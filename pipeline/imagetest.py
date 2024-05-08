
from datetime import datetime
from os import path
import os

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.calibration import applycal_data_col

SCRATCH_DIR = '/fast/celery/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    s = datetime(2024, 4 ,15, 6, 0, 0)
    e = datetime(2024, 4 ,15, 15, 0, 0)
    """
    for spw in spws[1:]:
        pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
        bcal_path = pm.get_bcal_path(s.date())
        print(len(pm.ms_list))
        for ts, ms in pm.ms_list:
            outpath = pm.data_product_path(ts, 'ms')
            if not path.exists(path.dirname(outpath)):
                os.makedirs(path.dirname(outpath))
            applycal_data_col.delay(ms, bcal_path, out_ms=outpath)
    """