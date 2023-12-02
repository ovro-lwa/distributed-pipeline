from orca.proj.boilerplate import run_dada2ms, peel, apply_a_priori_flags, flag_chans, run_image_sub, run_co_add
from celery import group
from ..metadata.pathsmanagers import OfflinePathsManager, SIDEREAL_DAY
from orca.proj.transientbatchtasks import make_image_products

from datetime import datetime, timedelta
import itertools
import sys
import logging
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%YT%I:%M:%S %p')

flag_npy_mapping = dict(
    (datetime(2018, 3, dd, 12, 0, 0),
     f'/home/yuping/100-hr-a-priori-flags/20201026-day-flags/2018-03-{dd}-consolidated.npy') for dd in range(21, 27)
)
pm_whole = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                               dadafile_dir='/lustre/data/2018-03-20_100hr_run',
                               working_dir='/lustre/yuping/0-100-hr-reduction/final-narrow/',
                               gaintable_dir='/lustre/yuping/0-100-hr-reduction/bandpass/',
                               flag_npy_paths=flag_npy_mapping)

NARROW_ONLY = True
INTEGRATION_TIME = timedelta(seconds=13)
TMP_DIR = '/lustre/yuping/scratch'


def calibration_pipeline(start_time, end_time, cal_date):
    spw_l = [12]
    # spw_l = range(22)
    pm = pm_whole.time_filter(start_time=start_time,
                              end_time=end_time)
    group([
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), out_ms=pm.get_ms_path(t, f'{s:02d}'),
                      gaintable=pm.get_bcal_path(cal_date, f'{s:02d}')) |
        apply_a_priori_flags.s(flag_npy_path=pm.get_flag_npy_path(t)) |
        peel.s(t) |
        flag_chans.s(spw=s, uvcut_m=30)
        for t in pm.utc_times_mapping.keys() for s in spw_l])()


def imaging_steps(start_time_day1: datetime, end_time_day1: datetime, chunk_size: int):
    pm = pm_whole.time_filter(start_time=start_time_day1, end_time=end_time_day1)
    timestamp_chunks = pm.chunks_by_integration(chunk_size)
    timestamp_chunks_day2 = [[t + SIDEREAL_DAY for t in chunk] for chunk in timestamp_chunks]

    ms_parent_chunks = [[pm.get_ms_parent_path(ts) for ts in c] for c in timestamp_chunks]
    ms_parent_chunks_day2 = [[pm.get_ms_parent_path(ts) for ts in c] for c in timestamp_chunks_day2]

    last_timestamp = list(pm.utc_times_mapping.keys())[-1]
    ms_after_last_day1 = pm.get_ms_parent_path(last_timestamp + INTEGRATION_TIME)
    ms_after_last_day2 = pm.get_ms_parent_path(last_timestamp + INTEGRATION_TIME + SIDEREAL_DAY)
    assert len(ms_parent_chunks) == len(ms_parent_chunks_day2)
    logging.info(f'Chunk size is {len(ms_parent_chunks)} with last chunk size {len(ms_parent_chunks[-1])}.')

    narrow_long = 'narrow_long'
    long = 'long'

    sid_diff = 'sidereal_diff'
    sid_long = 'sidereal_long_diff'
    sid_narrow = 'sidereal_narrow_diff'

    logging.info('Making directories.')
    for out_dir in [sid_diff, sid_long, sid_narrow]:
        for c1, c2 in zip(timestamp_chunks, timestamp_chunks_day2):
            for ts in itertools.chain(c1, c2):
                os.makedirs(f'{pm.working_dir}/{out_dir}/{ts.date()}/hh={ts.hour:02d}', exist_ok=True)

    for out_dir in [f'{narrow_long}/before', f'{narrow_long}/after', f'{long}/before', f'{long}/after']:
        for c1, c2 in zip(timestamp_chunks, timestamp_chunks_day2):
            for ts in itertools.chain(c1, c2):
                os.makedirs(f'{pm.working_dir}/{out_dir}/{ts.date()}/hh={ts.hour:02d}', exist_ok=True)

    group([make_image_products.s(c1, c2,
                                 ms_parent_chunks[i+1][0],
                                 ms_parent_chunks_day2[i+1][0],
                                 f'{pm.working_dir}/snapshot',
                                 f'{pm.working_dir}/narrow',
                                 f'{pm.working_dir}/subsequent_diff',
                                 f'{pm.working_dir}/{narrow_long}',
                                 f'{pm.working_dir}/{sid_diff}',
                                 f'{pm.working_dir}/{sid_narrow}',
                                 f'{pm.working_dir}/{sid_long}',
                                 TMP_DIR, spw_list=[12])
           for i, (c1, c2) in enumerate(zip(ms_parent_chunks[:-1], ms_parent_chunks_day2[:-1]))
           ])()

    # TODO what if the last chunk has only one element?
    make_image_products.delay(ms_parent_chunks[-1], ms_parent_chunks_day2[-1],
                              ms_after_last_day1, ms_after_last_day2,
                              f'{pm.working_dir}/snapshot',
                              f'{pm.working_dir}/narrow',
                              f'{pm.working_dir}/subsequent_diff',
                              f'{pm.working_dir}/{narrow_long}',
                              f'{pm.working_dir}/{sid_diff}',
                              f'{pm.working_dir}/{sid_narrow}',
                              f'{pm.working_dir}/{sid_long}',
                              TMP_DIR, spw_list=[12])
