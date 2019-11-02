from .boilerplate import run_chgcentre
from .celery import app
from celery import group
import logging
import shutil
from ..metadata.pathsmanagers import OfflinePathsManager
import glob
import os

logging.basicConfig(level=logging.INFO)

REF_POSITION = '-02h13m03.31s 36d58m27.57s'

pm = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                         msfile_dir='/lustre/yuping/0-100-hr-reduction/msfile',
                         bcal_dir='/lustre/yuping/0-100-hr-reduction/day-1-bcal/',
                         flag_npy_path='/home/yuping/100-hr-a-priori-flags/consolidated_flags.npy')


@app.task
def run_average(ms_file_list, ref_ms_index, out_ms):
    temp_ms = '/dev/shm/yuping/' + os.path.basename(out_ms)
    average_ms(ms_file_list, ref_ms_index, temp_ms, 'DATA')
    logging.info('Finished averaging. Copying the final measurement set from /dev/shm back.')
    shutil.copytree(temp_ms, out_ms)


def chgcentre():
    msl = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/ms/*/*ms'))
    group(run_chgcentre.s(ms, REF_POSITION) for ms in msl)()


def average_ms():
    ref_ms_index = 36
    spws = [f'{i:02d}' for i in range(22)]
    for s in spws:
        # TODO generate this without having to stat. This doesn't scale well on lustre
        ms_list = sorted(glob.glob(f'/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/ms/*/{s}_*.ms'))
        out_ms = '/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/' + os.path.basename(ms_list[ref_ms_index])
        run_average.delay(ms_list, ref_ms_index, out_ms)