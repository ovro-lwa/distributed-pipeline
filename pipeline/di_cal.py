from datetime import datetime
from os import path
import os

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.calibration import di_cal_multi, di_cal_multi_v2

SCRATCH_DIR = '/fast/celery/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/v2'

MINUTES_TO_CAL = 15

if __name__ == '__main__':
    cal_hr_early = {11: 1, 12: 2, 1: 2}
    cal_hr_late = {2:14, 3:13, 4: 12, 5: 11, 6:11}

    year = 2024
    month = 5
    for d in range(24, 25):
        if not path.exists(f'{NIGHTTIME_DIR}55MHz/{year}-{month:02d}-{d:02d}'):
            continue
        if month in cal_hr_late:
            s = datetime(year, month, d, cal_hr_late[month], 20, 0)
            e = datetime(year, month, d, cal_hr_late[month], 35, 0)
        elif month in cal_hr_early:
            s = datetime(year, month, d, cal_hr_early[month], 10, 0)
            e = datetime(year, month, d, cal_hr_early[month], 25, 0)
        else:
            raise ValueError('Calibration hour not specified.')

        for spw in spws:
            pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
            to_cal = [ m for _, m in pm.ms_list ]
            if len(pm.ms_list) < 100:
                if month in cal_hr_late:
                    pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw,
                                            datetime(year, month, d, 10, 3, 0)
                                            , e)
                    to_cal = [ m for _, m in pm.ms_list[-(MINUTES_TO_CAL * 60 // 10):] ]
                else:
                    pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw,
                                            s
                                            , datetime(year, month, d, 10, 3, 0))
                    to_cal = [ m for _, m in pm.ms_list[:(MINUTES_TO_CAL * 60 // 10)] ]


            bcal_path = pm.get_bcal_path(s.date())
            os.makedirs(path.dirname(bcal_path), exist_ok=True)
            di_cal_multi_v2.delay(to_cal, SCRATCH_DIR, out=bcal_path)
    """

    s = datetime(2024, 4 ,29, 12, 00, 0)
    e = datetime(2024, 4 ,29, 12, 25, 0)
    for spw in spws[1:]:
        pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
        bcal_path = pm.get_bcal_path(s.date())
        if not path.exists(path.dirname(bcal_path)):
            os.makedirs(path.dirname(bcal_path))
        di_cal_multi.delay([m for _, m in pm.ms_list], SCRATCH_DIR, out=bcal_path)
    """