from ..transform.averagems import average_ms
from .celery import app
from orca.wrapper import dada2ms
import shutil
import glob
import os
import logging
import uuid

from datetime import datetime
from casacore import tables
from typing import List

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


@app.task
def run_average(ms_file_list, ref_ms_index, out_ms, fault_tolerant=True):
    temp_ms = '/dev/shm/yuping/' + os.path.basename(out_ms)
    average_ms(ms_file_list, ref_ms_index, temp_ms, 'DATA', tolerate_ms_io_error=fault_tolerant)
    logging.info('Finished averaging. Copying the final measurement set from /dev/shm back.')
    shutil.copytree(temp_ms, out_ms)
    shutil.rmtree(temp_ms)


@app.task
def running_average(dada_list: List[str], bcal: str, spw: str, out_ms: str, scratch_dir: str):
    tmpdir = f'{scratch_dir}/{uuid.uuid4()}'
    os.makedirs(tmpdir)
    count = len(dada_list)
    dada2ms.dada2ms(dada_list[0], out_ms, bcal)
    with tables.table(out_ms) as t:
        avg = t.getcol('DATA') / count

    for i, dada in enumerate(dada_list[1:]):
        ms = dada2ms.dada2ms(dada, f'{tmpdir}/{spw}_{i+1}.ms', bcal)
        with tables.table(ms) as t:
            avg += t.getcol('DATA') / count
        shutil.rmtree(ms)

    with tables.table(out_ms, readonly=False, ack=False) as out_t:
        out_t.putcol('DATA', avg)
    shutil.rmtree(tmpdir)


def do_average_ms():
    ref_ms_index = 0
    spws = [f'{i:02d}' for i in range(22)]
    for s in spws:
        # TODO generate this without having to stat. This doesn't scale well on lustre
        ms_list = sorted(glob.glob(f'/lustre/yuping/0-100-hr-reduction/blflag/msfile/2018-03-22/*/*/{s}_*.ms'))
        out_ms = '/lustre/yuping/0-100-hr-reduction/blflag/outms2/' + os.path.basename(ms_list[ref_ms_index])
        run_average.delay(ms_list, ref_ms_index, out_ms)