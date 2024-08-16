from orca import celery
from orca.transform.imaging import stokes_IV_imaging
from orca.utils import coordutils

from datetime import datetime, timedelta
from orca.metadata.stageiii import spws

SCRATCH = '/fast/celery'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    start_time = datetime(2024, 5, 24, 7, 0, 0)
    dt = timedelta(minutes=60)
    n_hours = 1

    for i in range(n_hours):
        hour_and_half_mark = datetime(
            start_time.year, start_time.month, start_time.day, start_time.hour, 30, 0)
        phase_center = coordutils.zenith_coord_at_ovro(hour_and_half_mark)
        end_time = start_time + dt
        stokes_IV_imaging.delay(spws[1:4], start_time, end_time,
                                NIGHTTIME_DIR, WORK_DIR, SCRATCH,
                                phase_center=phase_center, make_snapshots=True)
        stokes_IV_imaging.delay(spws[4:8], start_time, end_time,
                                NIGHTTIME_DIR, WORK_DIR, SCRATCH,
                                phase_center=phase_center, make_snapshots=True)
        stokes_IV_imaging.delay(spws[8:], start_time, end_time,
                                NIGHTTIME_DIR, WORK_DIR, SCRATCH,
                                phase_center=phase_center, make_snapshots=True)
        start_time += dt