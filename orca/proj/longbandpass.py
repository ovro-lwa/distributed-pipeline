from .celery import app
from celery import group
from ..transform import dada2ms, change_phase_centre, averagems, peeling
from ..flagging import flag_bad_chans, merge_flags
from datetime import datetime
import os
import logging
import glob
import shutil


def dispatch_dada2ms(start_time, end_time, dada_prefix, out_dir, utc_times_txt):
    spws = [f'{i:02d}' for i in range(22)]
    utc_times = {}
    with open(utc_times_txt, 'r') as f:
        for line in f:
            l = line.split(' ')
            utc_times[datetime.strptime(l[0], "%Y-%m-%dT%H:%M:%S")] = l[1].rstrip('\n')
    for time, dada in utc_times.items():
        if start_time <= time < end_time:
            p = f'{out_dir}/{time.isoformat()}'
            if not os.path.exists(p):
                logging.info('Making directory ', p)
                os.mkdir(p)
    params = dada2ms.generate_params(utc_times, start_time, end_time, spws, dada_prefix, out_dir)
    group(run_dada2ms.s(**p) for p in params)()


@app.task
def run_dada2ms(dada_file, out_ms, gaintable=None):
    dada2ms.run_dada2ms(dada_file, out_ms, gaintable)

@app.task
def run_chgcentre(ms_file, direction):
    change_phase_centre.change_phase_center(ms_file, direction)

@app.task
def run_average(ms_file_list, ref_ms_index, out_ms):
    temp_ms = '/dev/shm/yuping/' + os.path.basename(out_ms)
    averagems.average_ms(ms_file_list, ref_ms_index, temp_ms, 'DATA')
    logging.info('Finished averaging. Copying the final measurement set from /dev/shm back.')
    shutil.copytree(temp_ms, out_ms)

@app.task
def add(x, y):
    return x, y


@app.task
def peel(ms_file, sources):
    # TODO make this idempotent somehow
    peeling.peel_with_ttcal(ms_file, sources)

@app.task
def apply_chan_flags(ms_file):
    flag_bad_chans(ms_file)

@app.task
def apply_a_priori_flags(ms_file, flag_npy_path):
    merge_flags.write_to_flag_column(ms_file, flag_npy_path,
                                     create_corrected_data_column=True)

def chgcentre():
    msl = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/ms/*/*ms'))
    group(run_chgcentre.s(ms, '-02h13m03.31s 36d58m27.57s') for ms in msl)()


def average_ms():
    ref_ms_index = 36
    spws = [f'{i:02d}' for i in range(22)]
    for s in spws:
        # TODO generate this without having to stat. This doesn't scale well on lustre
        ms_list = sorted(glob.glob(f'/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/ms/*/{s}_*.ms'))
        out_ms = '/lustre/yuping/0-100-hr-reduction/cal-2018-03-22/' + os.path.basename(ms_list[ref_ms_index])
        run_average.delay(ms_list, ref_ms_index, out_ms)


def get_data():
        s = datetime(2018, 3, 23, 0, 0, 0)
        e = datetime(2018, 3, 23, 0, 0, 13)
        # s = datetime(2018, 3, 22, 12, 32, 0)
        # e = datetime(2018, 3, 22, 18, 32, 0)
        dp = '/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-23/hh=00'
        dap = '/lustre/data/2018-03-20_100hr_run'
        dispatch_dada2ms(s, e, dap, dp, '/lustre/yuping/2018-09-100-hr-autocorr/utc_times_isot.txt')


def flag_and_peel():
    ms_list = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-23/hh=00/*/??_*ms'))
    group(apply_a_priori_flags.s(ms,  '/home/yuping/100-hr-a-priori-flags/201906_consolidated.npy') for ms in ms_list)()
    #group(peel.s(ms,  '/home/yuping/casA_resolved_rfi.json') for ms in ms_list)()
