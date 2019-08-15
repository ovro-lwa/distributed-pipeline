from .celery import app
from celery import group
from ..transform import dada2ms
from datetime import datetime
import os
import logging


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
    group(run_dada2ms.s(**p) for p in params)().get()


@app.task
def run_dada2ms(dada_file, out_ms):
    dada2ms.run_dada2ms(dada_file, out_ms)


@app.task
def add(x, y):
    return x, y


def main():
    s = datetime(2018, 3, 22, 17, 32, 0)
    e = datetime(2018, 3, 22, 17, 48, 0)
    dp = '/lustre/yuping/0-100-hr-reduction/ms'
    dap = '/lustre/data/2018-03-20_100hr_run'
    dispatch_dada2ms(s, e, dap, dp, '/lustre/yuping/2018-09-100-hr-autocorr/utc_times_isot.txt')
