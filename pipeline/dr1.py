from orca import celery
from orca.transform.imaging import stokes_IV_imaging
from orca.utils import coordutils

from datetime import datetime, timedelta
from orca.metadata.stageiii import spws

SCRATCH = '/fast/celery'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    start_time = datetime(2024, 2, 26, 3, 0, 0)
    dt = timedelta(minutes=15)
    n_hours = 12
    n_chunks = n_hours * 4

    for i in range(n_chunks):
        hour_and_half_mark = datetime(
            start_time.year, start_time.month, start_time.day, start_time.hour, 30, 0)
        phase_center = coordutils.zenith_coord_at_ovro(hour_and_half_mark)
        end_time = start_time + dt
        for spw in spws:
            stokes_IV_imaging.delay([spw], start_time, end_time,
                                    NIGHTTIME_DIR, WORK_DIR, SCRATCH,
                                    phase_center=phase_center)
        start_time += dt