from orca import celery
from orca.transform.imaging import stokes_IV_imaging

from datetime import datetime
from orca.metadata.stageiii import spws

SCRATCH = '/fast/celery'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    start_time = datetime(2023, 12, 22, 13, 0, 0)
    end_time = datetime(2023, 12, 22, 13, 30, 0)
    
    stokes_IV_imaging.delay(spws[1:5], start_time, end_time,
                            NIGHTTIME_DIR, WORK_DIR, SCRATCH, taper_inner_tukey=20)
    stokes_IV_imaging.delay(spws[5:], start_time, end_time,
                            NIGHTTIME_DIR, WORK_DIR, SCRATCH)