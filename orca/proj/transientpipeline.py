from orca.proj.boilerplate import run_dada2ms, peel, apply_a_priori_flags, flag_chans, run_image_sub, run_co_add
from celery import group, chord
from ..metadata.pathsmanagers import OfflinePathsManager, SIDEREAL_DAY
from orca.proj.transientbatchtasks import make_image_products

from datetime import datetime, date
import itertools
import sys
import logging
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%YT%I:%M:%S %p')

pm_whole = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                               dadafile_dir='/lustre/data/2018-03-20_100hr_run',
                               working_dir='/lustre/yuping/0-100-hr-reduction/final/',
                               gaintable_dir='/lustre/yuping/2019-10-100-hr-take-two/bandpass/',
                               flag_npy_paths='/home/yuping/100-hr-a-priori-flags/20191125-consolidated-flags/20200602-consolidated-flags.npy')


def calibration_pipeline(start_time, end_time, cal_date):
    pm = pm_whole.time_filter(start_time=start_time,
                              end_time=end_time)
    group([
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), out_ms=pm.get_ms_path(t, f'{s:02d}'),
                      gaintable=pm.get_bcal_path(cal_date, f'{s:02d}')) |
        apply_a_priori_flags.s(flag_npy_path=pm.get_flag_npy_path(t)) |
        peel.s(t) |
        flag_chans.s(spw=s)
        for t in pm.utc_times_mapping.keys() for s in range(22)])()


def imaging_steps(start_time_day1: datetime, end_time_day1: datetime, chunk_size: int):
    pm = pm_whole.time_filter(start_time=start_time_day1, end_time=end_time_day1)
    timestamp_chunks = pm.chunks_by_integration(chunk_size)
    timestamp_chunks_day2 = [[t + SIDEREAL_DAY for t in chunk] for chunk in timestamp_chunks]

    ms_parent_chunks = [[pm.get_ms_parent_path(ts) for ts in c] for c in timestamp_chunks]
    ms_parent_chunks_day2 = [[pm.get_ms_parent_path(ts) for ts in c] for c in timestamp_chunks_day2]

    assert len(ms_parent_chunks) == len(ms_parent_chunks_day2)
    logging.info(f'Chunk size is {len(ms_parent_chunks)} with last chunk size {len(ms_parent_chunks[-1])}.')
    group([make_image_products.s(c1, c2,
                                 ms_parent_chunks[i+1][0],
                                 ms_parent_chunks_day2[i+1][0],
                                 f'{pm.working_dir}/snapshot',
                                 f'{pm.working_dir}/narrow',
                                 f'{pm.working_dir}/subsequent_diff',
                                 '/pipedata/workdir/yuping/')
           for i, (c1, c2) in enumerate(zip(ms_parent_chunks[:-1], ms_parent_chunks_day2[:-1]))
           ])()
    # TODO what if the last chunk has only one element?
    make_image_products.delay(ms_parent_chunks[-1][:-1], ms_parent_chunks_day2[-1][:-1],
                              ms_parent_chunks[-1][-1], ms_parent_chunks_day2[-1][-1],
                              f'{pm.working_dir}/snapshot',
                              f'{pm.working_dir}/narrow',
                              f'{pm.working_dir}/subsequent_diff',
                              '/pipedata/workdir/yuping/')


def subtraction_step(start_time_day1: datetime, end_time_day1: datetime, chunksize: int):
    pm = pm_whole.time_filter(start_time=start_time_day1,
                              end_time=end_time_day1)
    timestamp_chunks = pm.chunks_by_integration(chunksize)
    timestamp_chunks_day2 = [[t + SIDEREAL_DAY for t in chunk] for chunk in timestamp_chunks]

    snapshot='snapshot'
    narrow = 'narrow'
    sid = 'sidereal_diff'
    sid_long = 'sidereal_long_diff'
    sid_narrow = 'sidereal_narrow_diff'

    logging.info('Making directories.')
    for out_dir in [sid, sid_long, sid_narrow]:
        for c1, c2 in zip(timestamp_chunks, timestamp_chunks_day2):
            for ts in itertools.chain(c1, c2):
                os.makedirs(f'{pm.working_dir}/{out_dir}/{ts.date()}/hh={ts.hour:02d}', exist_ok=True)

    logging.info('Dispatching tasks...')
    for c1, c2 in zip(timestamp_chunks, timestamp_chunks_day2):
        # for each image, subtract each snapshot, add subtracted images across time.
        chord(run_image_sub.s(pm.dpp(ts1, snapshot, '.fits', 'diff'),
                              pm.dpp(ts2, snapshot, '.fits', 'diff'),
                              pm.dpp(ts1, sid, '.fits')) for ts1, ts2 in zip(c1, c2))(
            run_co_add.s(output_fits_path=pm.dpp(c1[0], sid_long, '.fits')))()
        # for narrow band, just co-add and subtract
        pass
