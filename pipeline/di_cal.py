from datetime import datetime
from os import path
import os

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.calibration import di_cal_multi

SCRATCH_DIR = '/fast/celery/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    for d in range(1, 32):
        # TODO a few days need redo.
        s = datetime(2024, 1 ,d, 3, 30, 0)
        e = datetime(2024, 1 ,d, 3, 55, 0)
        for spw in spws[1:]:
            pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
            if len(pm.ms_list) == 0:
                continue
            bcal_path = pm.get_bcal_path(s.date())
            if not path.exists(path.dirname(bcal_path)):
                os.makedirs(path.dirname(bcal_path))
            di_cal_multi.delay([m for _, m in pm.ms_list], SCRATCH_DIR, out=bcal_path)
