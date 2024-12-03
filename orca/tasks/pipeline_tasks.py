# orca/tasks/pipeline_tasks.py

from orca.celery import app
from orca.transform.flagging import flag_ants as original_flag_ants
from orca.transform.flagging import flag_with_aoflagger as original_flag_with_aoflagger
from orca.transform.calibration import applycal_data_col as original_applycal_data_col
from orca.wrapper.wsclean import wsclean as original_wsclean
from orca.wrapper.ttcal import peel_with_ttcal 
from orca.transform.averagems import average_frequency
from orca.wrapper import change_phase_centre
from typing import List

@app.task
def flag_ants_task(ms: str, ants: List[int]) -> str:
    return original_flag_ants(ms, ants)

@app.task
def flag_with_aoflagger_task(ms: str, strategy: str, in_memory: bool, n_threads: int) -> str:
    return original_flag_with_aoflagger(ms, strategy=strategy, in_memory=in_memory, n_threads=n_threads)

@app.task
def applycal_data_col_task(ms: str, gaintable: str) -> str:
    """
    Celery task to apply calibration to an MS.
    """
#    return original_applycal_data_col(ms, gaintable)
    out_ms = ms.rstrip('/') + '_calibrated.ms'
    return original_applycal_data_col(ms, gaintable, out_ms)

@app.task
def wsclean_task(ms: str, out_dir: str, filename_prefix: str, extra_args: List[str],
                 num_threads: int, mem_gb: int) -> None:
    original_wsclean([ms], out_dir, filename_prefix, extra_args, num_threads, mem_gb)
    return ms
    #return original_wsclean([ms], out_dir, filename_prefix, extra_args, num_threads, mem_gb)

@app.task
def peel_with_ttcal_task(ms: str, sources: str) -> str:
    """
    Celery task to use TTCal to peel sources.
    """
    return peel_with_ttcal(ms, sources)

@app.task
def average_frequency_task(ms: str, chanbin: int = 4) -> str:
    """
    Celery task to perform frequency averaging on an MS.
    """
    output_vis = ms.rstrip('/') + '_averaged.ms'
    return average_frequency(vis=ms, output_vis=output_vis, chanbin=chanbin)

@app.task
def change_phase_center_task(ms: str, new_phase_center: str) -> str:
    """
    Celery task to change the phase center of a calibrated and averaged MS.
    """
    try:
        # Execute the phase center change
        updated_ms = change_phase_centre.change_phase_center(ms, new_phase_center)
        return updated_ms
    except Exception as e:
        raise RuntimeError(f"Failed to change phase center for {ms}: {e}")


