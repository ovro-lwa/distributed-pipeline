# orca/tasks/pipeline_tasks.py

import os
from orca.celery import app
from orca.transform.flagging import flag_ants as original_flag_ants
#from orca.transform.flagging import flag_with_aoflagger as original_flag_with_aoflagger
from orca.transform.flagging import flag_with_aoflagger, save_flag_metadata
from orca.transform.calibration import applycal_data_col as original_applycal_data_col
from orca.wrapper.wsclean import wsclean as original_wsclean
from orca.wrapper.ttcal import peel_with_ttcal 
from orca.transform.averagems import average_frequency
from orca.wrapper import change_phase_centre
from orca.utils.calibrationutils import build_output_paths
from typing import List
import shutil

@app.task
def copy_ms_task(original_ms: str, base_output_dir: str = '/lustre/pipeline/slow-averaged/') -> str:
    """
    Copy the MS from its original location to slow-averaged directory.
    Returns the path to the copied MS.
    """

    output_dir, ms_base = build_output_paths(original_ms, base_output_dir=base_output_dir)
    copied_ms = os.path.join(output_dir, ms_base + '.ms')
    shutil.copytree(original_ms, copied_ms)
    return copied_ms

@app.task
def copy_ms_nighttime_task(original_ms: str) -> str:
    """
    Copy the MS file to the same directory with a new name.
    The copied file will have '_copy' appended to the base name.

    Example:
        original_ms = '/lustre/pipeline/night-time/73MHz/2023-11-21/03/20231121_031000_73MHz.ms'
        copied_ms = '/lustre/pipeline/night-time/73MHz/2023-11-21/03/20231121_031000_73MHz_copy.ms'

    Returns:
        str: The path to the copied MS.
    """
    # Extract directory and filename
    dir_name = os.path.dirname(original_ms)  # e.g., /lustre/pipeline/night-time/73MHz/2023-11-21/03
    base_name = os.path.basename(original_ms)  # e.g., 20231121_031000_73MHz.ms
    name, ext = os.path.splitext(base_name)  # ('20231121_031000_73MHz', '.ms')

    # Create the new filename with _copy appended
    copied_ms = os.path.join(dir_name, f"{name}_copy{ext}")  # /lustre/pipeline/night-time/73MHz/2023-11-21/03/20231121_031000_73MHz_copy.ms

    # Copy the directory (measurement set) from the original to the new location
    shutil.copytree(original_ms, copied_ms)

    return copied_ms

@app.task
def remove_ms_task(ms_tuple: tuple) -> str:
    # ms_tuple = (original_ms, averaged_ms)
    import shutil
    original_ms, averaged_ms = ms_tuple
    shutil.rmtree(original_ms, ignore_errors=True)
    # Return the averaged_ms path to keep track of it
    return averaged_ms


@app.task
def flag_ants_task(ms: str, ants: List[int]) -> str:
    return original_flag_ants(ms, ants)


@app.task
def flag_with_aoflagger_task(ms: str, strategy: str='/opt/share/aoflagger/strategies/nenufar-lite.lua', in_memory: bool=False, n_threads:int=5) -> str:
    return flag_with_aoflagger(ms, strategy=strategy, in_memory=in_memory, n_threads=n_threads)

@app.task
def save_flag_metadata_task(ms: str) -> str:
    output_dir, ms_base = build_output_paths(ms)
    # Pass output_dir to the save_flag_metadata function
    return save_flag_metadata(ms, output_dir=output_dir)

@app.task
def save_flag_metadata_nighttime_task(ms: str) -> str:
    # Use a different base output directory for night-time
    output_dir, ms_base = build_output_paths(ms, base_output_dir='/lustre/pipeline/night-time/averaged/')
    return save_flag_metadata(ms, output_dir=output_dir)


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
    output_dir, ms_base = build_output_paths(ms)
    output_vis = os.path.join(output_dir, f"{ms_base}_averaged.ms")
    averaged_ms = average_frequency(vis=ms, output_vis=output_vis, chanbin=chanbin)
    # Return a tuple: (original_ms, averaged_ms)
    return (ms, averaged_ms)    

@app.task
def average_frequency_nighttime_task(ms: str, chanbin: int = 4) -> str:
    output_dir, ms_base = build_output_paths(ms, base_output_dir='/lustre/pipeline/night-time/averaged/')
    output_vis = os.path.join(output_dir, f"{ms_base}_averaged.ms")
    averaged_ms = average_frequency(vis=ms, output_vis=output_vis, chanbin=chanbin)
    # Return (original_ms, averaged_ms) to be consistent with remove_ms_task input
    return (ms, averaged_ms)

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

@app.task
def extract_original_ms_task(ms_tuple: tuple) -> str:
    """
    Extract the original MS path from the tuple (original_ms, averaged_ms) 
    returned by `average_frequency_task`.
    
    Args:
        ms_tuple (tuple): A tuple where the first element is the original MS 
                          and the second is the path to the averaged MS.
    
    Returns:
        str: Path to the original MS.
    """
    return ms_tuple[0]