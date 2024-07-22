from orca import celery
from orca.transform.imaging import stokes_IV_imaging

from datetime import datetime, timedelta
from orca.metadata.stageiii import spws

SCRATCH = '/fast/celery'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    start_time = datetime(2024, 4, 16, 4, 0, 0)
    dt = timedelta(hours=1)
    n_hours = 8

    for i in range(n_hours):
        end_time = start_time + dt
        for spw in spws:
            stokes_IV_imaging.delay([spw], start_time, end_time,
                                    NIGHTTIME_DIR, WORK_DIR, SCRATCH)
            stokes_IV_imaging.delay([spw], start_time, end_time,
                                    NIGHTTIME_DIR, WORK_DIR, SCRATCH)
        start_time += dt