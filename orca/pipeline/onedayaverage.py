from ..transform.averagems import average_ms
from orca.celery import app
import shutil
import os
import logging

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


@app.task
def run_average(ms_file_list, ref_ms_index, out_ms, temp_dir, datacol='DATA', fault_tolerant=True) -> str:
    temp_ms = f'{temp_dir}/{os.path.basename(out_ms)}'
    average_ms(ms_file_list, ref_ms_index, temp_ms, datacol, tolerate_ms_io_error=fault_tolerant)
    logging.info(f'Finished averaging. Copying the final measurement set from {temp_dir} back.')
    shutil.copytree(temp_ms, out_ms)
    shutil.rmtree(temp_ms)
    return out_ms