# dynamic spectra for subet of baselines and incoherent sum
from datetime import datetime

from celery.canvas import chord, group
from orca.transform.spectrum import dynspec_map, dynspec_reduce
from orca.metadata.stageiii import StageIIIPathsManager, spws

import random

NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

if __name__ == '__main__':
    datetime_ms_map = {}
    subband_nos = dict((s, i) for i, s in enumerate(spws))
    s = datetime(2024,5,21,4,0,0)
    e = datetime(2024,5,21,15,0,0)
    for spw in spws:
        pm = StageIIIPathsManager(root_dir=NIGHTTIME_DIR, work_dir=WORK_DIR, subband=spw,
                                  start=s, end=e)
        bcal_path = pm.get_bcal_path(s.date())
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
    res = chord(subtasks)(dynspec_reduce.s(start_ts=start_ts, out_dir='/lustre/celery/baselines/'))
    print(res.get())
