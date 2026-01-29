"""Makeup imaging runs for missed observations.

This script queues Stokes I/V imaging tasks for observations that
were missed in the regular processing schedule.
"""
import datetime
from orca import celery
from orca.transform.imaging import stokes_IV_imaging

if __name__ == '__main__':
    stokes_IV_imaging.delay(['69MHz'], datetime.datetime(2024, 2, 25, 13, 15), datetime.datetime(2024, 2, 25, 13, 30), '/lustre/pipeline/night-time/', '/lustre/celery/', '/fast/celery')
    stokes_IV_imaging.delay(['82MHz'], datetime.datetime(2024, 2, 25, 6, 45), datetime.datetime(2024, 2, 25, 7, 0), '/lustre/pipeline/night-time/', '/lustre/celery/', '/fast/celery')
    stokes_IV_imaging.delay	(['78MHz'], datetime.datetime(2024, 2, 25, 6, 45), datetime.datetime(2024, 2, 25, 7, 0), '/lustre/pipeline/night-time/', '/lustre/celery/', '/fast/celery')
    stokes_IV_imaging.delay(['78MHz'], datetime.datetime(2024, 2, 25, 6, 30), datetime.datetime(2024, 2, 25, 6, 45), '/lustre/pipeline/night-time/', '/lustre/celery/', '/fast/celery')