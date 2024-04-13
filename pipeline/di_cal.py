from datetime import datetime

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.calibration import di_cal_multi


SCRATCH_DIR = '/fast/celery/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == 'main':
    s = datetime(2023, 12 ,22, 3, 0, 0)
    e = datetime(2023, 12 ,22, 4, 0, 0)
    pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, '14MHz', s, e)
    di_cal_multi.delay(pm.ms_list, SCRATCH_DIR, out=pm.get_bcal_path(s.date()))