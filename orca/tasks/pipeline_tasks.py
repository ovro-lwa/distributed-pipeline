# orca/tasks/pipeline_tasks.py

from orca.celery import app
from orca.transform.flagging import flag_ants as original_flag_ants
from orca.transform.flagging import flag_with_aoflagger as original_flag_with_aoflagger
from orca.transform.calibration import applycal_data_col as original_applycal_data_col
from orca.wrapper.wsclean import wsclean as original_wsclean
from orca.wrapper.ttcal import peel_with_ttcal 

from typing import List

@app.task
def flag_ants_task(ms: str, ants: List[int]) -> str:
    return original_flag_ants(ms, ants)

@app.task
def flag_with_aoflagger_task(ms: str, strategy: str, in_memory: bool, n_threads: int) -> str:
    return original_flag_with_aoflagger(ms, strategy=strategy, in_memory=in_memory, n_threads=n_threads)

@app.task
def applycal_data_col_task(ms: str, gaintable: str) -> str:
#    return original_applycal_data_col(ms, gaintable)
    out_ms = ms.rstrip('/') + '_calibrated.ms'
    return original_applycal_data_col(ms, gaintable, out_ms)

@app.task
def wsclean_task(ms: str, out_dir: str, filename_prefix: str, extra_args: List[str],
                 num_threads: int, mem_gb: int) -> None:
    return original_wsclean([ms], out_dir, filename_prefix, extra_args, num_threads, mem_gb)

@app.task
def peel_with_ttcal_task(ms: str, sources: str) -> str:
    """
    Celery task to use TTCal to peel sources.
    """
    return peel_with_ttcal(ms, sources)
