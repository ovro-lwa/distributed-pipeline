from datetime import datetime
from os import path
import os

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.calibration import di_cal_multi
from orca.utils import calibrationutils
SCRATCH_DIR = '/fast/celery/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    s = datetime(2023, 12 ,22, 5, 0, 0)
    e = datetime(2023, 12 ,22, 5, 25, 0)
    for spw in spws:
        pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
        bcal_path = pm.get_bcal_path(s.date())
        if not path.exists(path.dirname(bcal_path)):
            os.makedirs(path.dirname(bcal_path))
        di_cal_multi.delay(pm.ms_list, SCRATCH_DIR, out=bcal_path)