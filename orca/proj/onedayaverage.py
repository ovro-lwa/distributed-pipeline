from ..transform.averagems import average_ms
from .celery import app
import shutil
import glob
import os
import logging

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


@app.task
def run_average(ms_file_list, ref_ms_index, out_ms, fault_tolerant=True):
    temp_ms = '/dev/shm/yuping/' + os.path.basename(out_ms)
    try:
        average_ms(ms_file_list, ref_ms_index, temp_ms, 'DATA')
    except RuntimeError as e:
        if not fault_tolerant:
            raise e
        else:
            log.exception("IO Error while handling measurment sets.")
    logging.info('Finished averaging. Copying the final measurement set from /dev/shm back.')
    shutil.copytree(temp_ms, out_ms)
    shutil.rmtree(temp_ms)


def do_average_ms():
    ref_ms_index = 0
    spws = [f'{i:02d}' for i in range(22)]
    for s in spws:
        # TODO generate this without having to stat. This doesn't scale well on lustre
        ms_list = sorted(glob.glob(f'/lustre/yuping/0-100-hr-reduction/blflag/msfile/2018-03-22/*/*/{s}_*.ms'))
        out_ms = '/lustre/yuping/0-100-hr-reduction/blflag/outms/' + os.path.basename(ms_list[ref_ms_index])
        run_average.delay(ms_list, ref_ms_index, out_ms)