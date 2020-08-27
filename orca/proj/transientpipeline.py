from orca.proj.boilerplate import run_dada2ms, peel, apply_a_priori_flags, flag_chans, run_merge_flags
from celery import group
from ..metadata.pathsmanagers import OfflinePathsManager, SIDEREAL_DAY
from orca.proj.transientbatchtasks import make_image_products
from datetime import datetime, date
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%YT%I:%M:%S %p')

pm_whole = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                               dadafile_dir='/lustre/data/2018-03-20_100hr_run',
                               working_dir='/lustre/yuping/0-100-hr-reduction/final/',
                               gaintable_dir='/lustre/yuping/2019-10-100-hr-take-two/bandpass/',
                               flag_npy_paths='/home/yuping/100-hr-a-priori-flags/20191125-consolidated-flags/20200602-consolidated-flags.npy')


def calibration_pipeline():
    cal_date = date(2018, 3, 22)
    pm = pm_whole.time_filter(start_time=datetime(2018, 3, 23, 11, 52, 8),
                              end_time=datetime(2018, 3, 23, 17, 52, 8))
    group([
        run_dada2ms.s(pm.get_dada_path(f'{s:02d}', t), out_ms=pm.get_ms_path(t, f'{s:02d}'),
                      gaintable=pm.get_bcal_path(cal_date, f'{s:02d}')) |
        apply_a_priori_flags.s(flag_npy_path=pm.get_flag_npy_path(t)) |
        peel.s(t) |
        flag_chans.s(spw=s)
        for t in pm.utc_times_mapping.keys() for s in range(22)])()


def imaging_steps():
    pm = pm_whole.time_filter(start_time=datetime(2018, 3, 22, 11, 56, 4),
                              end_time=datetime(2018, 3, 22, 17, 56, 4))
    timestamp_chunks = pm.chunks_by_integration(47)
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


def subtraction_step():
    pm = pm_whole.time_filter(start_time=datetime(2018, 3, 22, 11, 56, 9),
                              end_time=datetime(2018, 3, 22, 17, 56, 4))
    timestamp_chunks = pm.chunks_by_integration(47)
    timestamp_chunks_day2 = [[t + SIDEREAL_DAY for t in chunk] for chunk in timestamp_chunks]
