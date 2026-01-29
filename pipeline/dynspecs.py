"""Dynamic spectrum extraction pipeline.

This script extracts dynamic spectra from measurement sets using
a subset of baselines and incoherent sum. Produces waterfall plots
across frequency and time for transient analysis.
"""
from datetime import datetime, timedelta, date
from os import path

from celery.canvas import chord, group
from orca.transform.spectrum import dynspec_map, dynspec_reduce
from orca.metadata.stageiii import StageIIIPathsManager, spws

import random
import time

NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    subband_nos = dict((s, i) for i, s in enumerate(spws))
    s = datetime(2024,2,22,1,0,0)
    e = datetime(2024,2,22,17,0,0)
    n_days = 5

    while n_days > 0:
        if ((not path.exists(f'/lustre/pipeline/night-time/18MHz/{s.date()}'))
            or (s.day == 22) or
            path.exists(f'/lustre/celery/baselines/incoherent-sum/{str(s.date())}-XX.fits')):
            s += timedelta(days=1)
            e += timedelta(days=1)
            continue
        print(f'Doing {s} to {e}')
        datetime_ms_map = {}
        for spw in spws:
            pm = StageIIIPathsManager(root_dir=NIGHTTIME_DIR, work_dir=WORK_DIR, subband=spw,
                                    start=s, end=e)
            # bcal_path = pm.get_bcal_path(s.date())
            bcal_path = pm.get_bcal_path(date(2024,4,8))
            for dt, ms in pm.ms_list:
                if dt not in datetime_ms_map:
                    datetime_ms_map[dt] = [(subband_nos[spw], ms, bcal_path)]
                else:
                    datetime_ms_map[dt].append((subband_nos[spw], ms, bcal_path))
        
        subtasks = []
        start_ts = None
        for i, timestamp in enumerate(sorted(datetime_ms_map.keys())):
            if i == 0:
                start_ts = timestamp
            for sb_no, ms, bcal in datetime_ms_map[timestamp]:
                subtasks.append(dynspec_map.s(sb_no, i, ms, bcal))

        random.shuffle(subtasks)
        print(len(subtasks))
        chord(subtasks)(dynspec_reduce.s(start_ts=start_ts, out_dir='/lustre/celery/baselines/'))
        s += timedelta(days=1)
        e += timedelta(days=1)
        n_days -= 1
        time.sleep(4800)