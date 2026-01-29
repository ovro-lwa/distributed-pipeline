# orca/tasks/pipeline_tasks.py
"""Core pipeline processing tasks for OVRO-LWA data reduction.

Contains Celery tasks for the main data processing workflows including:
- Measurement set copying and management
- Calibration application and flagging
- Bandpass calibration pipeline
- Frequency averaging
- Image generation and cleanup

These tasks form the building blocks of the automated pipeline.
"""
import os
from orca.celery import app
from orca.transform.flagging import flag_ants as original_flag_ants
#from orca.transform.flagging import flag_with_aoflagger as original_flag_with_aoflagger
from orca.transform.flagging import flag_with_aoflagger, save_flag_metadata, flag_ants
from orca.transform.calibration import applycal_data_col as original_applycal_data_col
from orca.wrapper.wsclean import wsclean as original_wsclean
from orca.wrapper.ttcal import peel_with_ttcal, zest_with_ttcal 
from orca.transform.averagems import average_frequency
from orca.wrapper import change_phase_centre
from orca.utils.calibrationutils import build_output_paths
from typing import List, Tuple
import shutil
from orca.utils.calibrationutils import is_within_transit_window, get_lst_from_filename, get_relative_path

import logging
from casatasks import concat, clearcal, ft, bandpass
from orca.utils.calibratormodel import model_generation
from orca.utils.msfix import concat_issue_fieldid
#from orca.wrapper.change_phase_centre import get_phase_center
#from orca.utils.flagutils import get_bad_antenna_numbers
from orca.utils.calibrationutils import parse_filename
from orca.transform.qa_plotting import plot_bandpass_to_pdf_amp_phase
from orca.utils.paths import get_aoflagger_strategy

from orca.calibration.bandpass_pipeline import run_bandpass_calibration
from orca.utils.paths import get_aoflagger_strategy

from celery.exceptions import Retry
import time

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

#@app.task
#def remove_ms_task(ms_tuple: tuple) -> str:
#    # ms_tuple = (original_ms, averaged_ms)
#    import shutil
#    original_ms, averaged_ms = ms_tuple
#    shutil.rmtree(original_ms, ignore_errors=True)
#    # Return the averaged_ms path to keep track of it
#    return averaged_ms


@app.task
def flag_ants_task(ms: str, ants: List[int]) -> str:
    """Flag antennas in the measurement set using the provided antenna indices"""
    return original_flag_ants(ms, ants)


@app.task
def flag_with_aoflagger_task(ms: str, strategy: str='/opt/share/aoflagger/strategies/nenufar-lite.lua', in_memory: bool=False, n_threads:int=1) -> str:
    """Apply AOFlagger on the measurement set with specified strategy and options"""
    return flag_with_aoflagger(ms, strategy=strategy, in_memory=in_memory, n_threads=n_threads)

@app.task
def save_flag_metadata_task(ms: str) -> str:
    """Save flag metadata for the measurement set and return the updated path"""
    output_dir, ms_base = build_output_paths(ms)
    # Pass output_dir to the save_flag_metadata function
    return save_flag_metadata(ms, output_dir=output_dir)

@app.task
def save_flag_metadata_nighttime_task(ms: str) -> str:
    """Save flag metadata for a nighttime measurement set using the nighttime output directory"""
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
    """Run wsclean imaging on the measurement set and return the original MS path"""
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
    """Perform frequency averaging on the measurement set; returns a tuple (original_ms, averaged_ms)"""
    output_dir, ms_base = build_output_paths(ms)
    output_vis = os.path.join(output_dir, f"{ms_base}_averaged.ms")
    averaged_ms = average_frequency(vis=ms, output_vis=output_vis, chanbin=chanbin)
    # Return a tuple: (original_ms, averaged_ms)
    return (ms, averaged_ms)    

@app.task
def average_frequency_nighttime_task(ms: str, chanbin: int = 4) -> str:
    """Perform frequency averaging on a nighttime measurement set; returns (original_ms, averaged_ms)"""
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


@app.task
def run_entire_pipeline_on_one_cpu(vis: str, window_minutes: int=4, start_hour: int=11, end_hour: int=14, chanbin: int=4) -> str:
    """
    A single-task pipeline runner that combines all steps into one execution on the same node (one CPU).
    It uses the existing tasks but calls them directly in a single function.
    """
    def is_within_lst_range(ms: str, start_hour=11, end_hour=14) -> bool:
        lst = get_lst_from_filename(ms).hour
        return (start_hour <= lst <= end_hour)

    sources_in_window = is_within_transit_window(vis, window_minutes=window_minutes)
    in_lst_range = is_within_lst_range(vis, start_hour, end_hour)

    if sources_in_window or in_lst_range:
        # Scenario i)
        # Copy -> Flag -> Save flag meta -> Average freq -> Remove
        copied_ms = copy_ms_nighttime_task(vis)
        flagged_ms = flag_with_aoflagger_task(copied_ms)
        flagged_ms = save_flag_metadata_nighttime_task(flagged_ms)
        ms_tuple = average_frequency_nighttime_task(flagged_ms, chanbin=chanbin)
        averaged_ms = remove_ms_task(ms_tuple)  # returns averaged_ms
        return averaged_ms
    else:
        # Scenario ii)
        # Flag original -> Save flag meta -> Average freq -> Remove original
        flagged_ms = flag_with_aoflagger_task(vis)
        flagged_ms = save_flag_metadata_nighttime_task(flagged_ms)
        ms_tuple = average_frequency_nighttime_task(flagged_ms, chanbin=chanbin)
        averaged_ms = remove_ms_task(ms_tuple)
        return averaged_ms

@app.task
def copy_ms_to_nvme_task(original_ms: str, nvme_base_dir: str = '/fast/pipeline/') -> str:
    """
    Copy the MS from Lustre to NVMe (fast) storage, placing it directly in /fast/pipeline/.
    """
    ms_name = os.path.basename(original_ms)  # just get the filename
    nvme_ms = os.path.join(nvme_base_dir, ms_name)
    shutil.copytree(original_ms, nvme_ms)
    return nvme_ms

@app.task
def save_flag_metadata_nvme_task(ms: str) -> str:
    """
    Save flag metadata on NVMe directly in /fast/pipeline/.
    """
    ms_base = os.path.splitext(os.path.basename(ms))[0]
    output_file = os.path.join('/fast/pipeline', f"{ms_base}_flagmeta.bin")
    # reuse the save_flag_metadata function but specify the output_dir as '/fast/pipeline'
    save_flag_metadata(ms, output_dir='/fast/pipeline')
    return ms

@app.task
def average_frequency_nvme_task(ms: str, chanbin: int = 4) -> tuple:
    """
    Average frequency on NVMe, storing output in /fast/pipeline/.
    """
    ms_base = os.path.splitext(os.path.basename(ms))[0]
    output_vis = os.path.join('/fast/pipeline', f"{ms_base}_averaged.ms")
    averaged_ms = average_frequency(vis=ms, output_vis=output_vis, chanbin=chanbin)
    return (ms, averaged_ms)

@app.task
def remove_ms_task(ms_tuple: tuple) -> str:
    """Remove the original measurement set directory and return the averaged MS path"""
    original_ms, averaged_ms = ms_tuple
    shutil.rmtree(original_ms, ignore_errors=True)
    return averaged_ms

@app.task
def run_entire_pipeline_on_one_cpu_nvme(vis: str, window_minutes: int=4, start_hour: int=11, end_hour: int=14, chanbin: int=4) -> str:
    """
    NVMe-based pipeline run with simplified NVMe storage logic:
    - Copy from Lustre to /fast/pipeline/ with no subdirectories.
    - Flag, save metadata, average on NVMe (all in /fast/pipeline/).
    - Move the final averaged MS and flag metadata back to /lustre/pipeline/night-time/averaged/ 
      using the original vis path to determine final directory structure.
    - If scenario ii) (not in window or LST range), remove original MS from Lustre.
    """

    def is_within_lst_range(ms: str, start_hour=11, end_hour=14) -> bool:
        lst = get_lst_from_filename(ms).hour
        return (start_hour <= lst <= end_hour)

    sources_in_window = is_within_transit_window(vis, window_minutes=window_minutes)
    in_lst_range = is_within_lst_range(vis, start_hour, end_hour)

    # Determine final output paths on Lustre based on original vis
    final_output_dir, ms_base = build_output_paths(vis, base_output_dir='/lustre/pipeline/night-time/averaged/')
    final_averaged_ms = os.path.join(final_output_dir, f"{ms_base}_averaged.ms")

    # Copy from Lustre to NVMe
    nvme_ms = copy_ms_to_nvme_task(vis)

    # Flag on NVMe
    strategy = get_aoflagger_strategy("LWA_opt_GH1.lua")
    logging.info(f"[NVMe pipeline] Flagging {nvme_ms} with strategy {strategy}")
    flagged_ms = flag_with_aoflagger(ms=nvme_ms,strategy=strategy, in_memory=False, n_threads=1)

    # Save flag metadata on NVMe
    flagged_ms = save_flag_metadata_nvme_task(flagged_ms)

    # Average frequency on NVMe
    ms_tuple = average_frequency_nvme_task(flagged_ms, chanbin=chanbin)

    # Remove the original NVMe MS (the first in the tuple)
    averaged_ms_on_nvme = remove_ms_task(ms_tuple)

    # Move the final averaged MS from NVMe back to Lustre
    os.makedirs(os.path.dirname(final_averaged_ms), exist_ok=True)
    shutil.move(averaged_ms_on_nvme, final_averaged_ms)

    # Move the flag metadata file from NVMe to the exact same directory where the averaged MS was moved
    from os.path import basename, dirname, join, splitext

    # Use final_output_dir instead of recalculating it
    ms_base_no_averaged = os.path.splitext(os.path.basename(vis))[0]  # Base name of the original MS
    nvme_meta_file = os.path.join('/fast/pipeline', f"{ms_base_no_averaged}_flagmeta.bin")

    # Use final_output_dir to place the flag metadata
    final_flag_meta = os.path.join(final_output_dir, f"{ms_base_no_averaged}_flagmeta.bin")

    if os.path.exists(nvme_meta_file):
        os.makedirs(final_output_dir, exist_ok=True)  # Use final_output_dir directly
        shutil.move(nvme_meta_file, final_flag_meta)  # Move to the final directory

    # If not in transit window and not in LST range (scenario ii), remove the original MS from Lustre
    if not (sources_in_window or in_lst_range):
        shutil.rmtree(vis, ignore_errors=True)

    return final_averaged_ms



@app.task
def copy_ms_to_calibration_task(original_ms: str, calibration_base_dir: str = '/lustre/pipeline/calibration/') -> str:
    """
    Copy the MS from its original slow directory to the calibration directory
    preserving frequency/date/hour structure.
    """
    # Extract relative path from the original_ms
    relative_path = get_relative_path(original_ms)  # e.g. "13MHz/2024-12-17/00/20241217_001153_13MHz.ms"
    calibration_ms = os.path.join(calibration_base_dir, relative_path)
    os.makedirs(os.path.dirname(calibration_ms), exist_ok=True)
    shutil.copytree(original_ms, calibration_ms)
    return calibration_ms



def get_utc_hour_from_path(ms_path: str) -> int:
    """Extract the UTC hour from the measurement set path (assumes the hour is the second-to-last path component)"""
    parts = ms_path.split('/')
    hour_str = parts[-2]
    return int(hour_str)


@app.task(bind=True,autoretry_for=(Exception,),retry_kwargs={"max_retries": 3, "countdown": 10},)
def run_pipeline_slow_on_one_cpu_nvme(self, vis: str, start: int = 1, end: int = 14, chanbin: int = 4) -> str:
    """
    A pipeline that:
    - Checks if MS is a calibrator; if yes, copy to calibration directory without removing original.
    - If MS UTC hour is in [start..end], process it on NVMe: copy to NVMe, flag, save metadata, average.
      After averaging, move results to /lustre/pipeline/slow-averaged/.
    - Do not remove the original MS from /lustre/pipeline/slow/.
    - Remove the NVMe copy after processing.
    """

    # Check if calibrator
    #sources_in_window = is_within_transit_window(vis, window_minutes=4)
    #if sources_in_window:
        # Copy to calibration directory
    #    copy_ms_to_calibration_task(vis)

    # Check the UTC hour
    utc_hour = get_utc_hour_from_path(vis)

    # Process if hour in [start..end]
    if start <= utc_hour <= end:
        # Copy to NVMe
        nvme_ms = copy_ms_to_nvme_task(vis)

        # Flag on NVMe
        strategy = get_aoflagger_strategy("LWA_opt_GH1.lua")
        logging.info(f"[NVMe pipeline] Flagging {nvme_ms} with strategy {strategy}")
        flagged_ms = flag_with_aoflagger_task.run(nvme_ms, strategy=strategy, in_memory=False, n_threads=1)

        # Save flag metadata on NVMe
        flagged_ms = save_flag_metadata_nvme_task.run(flagged_ms)

        # Average frequency on NVMe
        ms_tuple = average_frequency_nvme_task(flagged_ms, chanbin=chanbin)
        # ms_tuple = (original_nvme_ms, averaged_ms_on_nvme)

        # Remove the original NVMe MS
        averaged_ms_on_nvme = remove_ms_task(ms_tuple)
        # Now we have the averaged MS on NVMe and the original NVMe MS is removed

        # Move the final averaged MS and flag metadata from NVMe back to Lustre (slow-averaged)
        final_output_dir, ms_base = build_output_paths(vis, base_output_dir='/lustre/pipeline/slow-averaged/')
        final_averaged_ms = os.path.join(final_output_dir, f"{ms_base}_averaged.ms")

        os.makedirs(os.path.dirname(final_averaged_ms), exist_ok=True)
        shutil.move(averaged_ms_on_nvme, final_averaged_ms)

        # Move the flag metadata file from NVMe to slow-averaged
        ms_base_no_averaged = os.path.splitext(os.path.basename(vis))[0]
        nvme_meta_file = os.path.join('/fast/pipeline', f"{ms_base_no_averaged}_flagmeta.bin")
        final_flag_meta = os.path.join(final_output_dir, f"{ms_base_no_averaged}_flagmeta.bin")

        if os.path.exists(nvme_meta_file):
            shutil.move(nvme_meta_file, final_flag_meta)

        return final_averaged_ms
    else:
        # Hour not in [start..end], do nothing special, just return the original vis path
        return vis


@app.task
def split_2pol_task(
    ms_input: str,
    ms_output: str = None,
    correlation: str = "XX,YY",
    datacolumn: str = "all",
    remove_original: bool = True
) -> str:
    """
    Celery task to split an MS down to the specified polarizations (default "XX,YY")
    using CASA's split task. By default, removes the original MS after splitting.

    :param ms_input: Path to the input measurement set (e.g., 
                     "/lustre/pipeline/cosmology/41MHz/2025-01-01/12/xyz.ms/")
    :param ms_output: Path to the output measurement set. If None, we auto-generate
                      by stripping trailing slashes, removing '.ms' extension, and
                      appending '_2pol.ms'
    :param correlation: Correlations to keep (e.g., "XX,YY")
    :param datacolumn: Data column to copy (default "all")
    :param remove_original: If True, remove the original MS after splitting
    :return: The path to the newly created 2-pol measurement set
    """
    from casatasks import split as casatask_split  # Import CASA tools inside the task

    # Strip trailing slash if present
    ms_input_stripped = ms_input.rstrip('/')

    if ms_output is None:
        # Example:
        #   ms_input_stripped = "/path/to/20250101_120802_41MHz.ms"
        #   base_name         = "/path/to/20250101_120802_41MHz"
        #   ms_output         = "/path/to/20250101_120802_41MHz_2pol.ms"
        base_name, _ = os.path.splitext(ms_input_stripped)
        ms_output = base_name + "_2pol.ms"

    # Run CASA split
    casatask_split(
        vis         = ms_input_stripped,
        outputvis   = ms_output,
        correlation = correlation,
        datacolumn  = datacolumn
    )

    print(f"[split_2pol_task] Created 2-pol MS -> {ms_output}")

    if remove_original:
        # Remove the input MS directory after the split is done
        # Usually safe with .rmtree, but 'rm -rf' also works
        if os.path.exists(ms_input_stripped):
            shutil.rmtree(ms_input_stripped, ignore_errors=True)
            print(f"[split_2pol_task] Removed original MS: {ms_input_stripped}")

    return ms_output


@app.task
def bandpass_nvme_task(ms_list, delay_table, obs_date, nvme_root="/fast/pipeline") -> str:
    return run_bandpass_calibration(ms_list, delay_table, obs_date, nvme_root=nvme_root)

#for Xander's data
import uuid

def _copy_ms_to_nvme_unique(original_ms: str, nvme_base_dir: str, task_id: str) -> str:
    """
    Copy the MS into a *unique* NVMe staging dir: <nvme_base_dir>/<task_id>/<ms_name>
    This avoids collisions even if multiple workers process the same MS.
    """
    ms_name = os.path.basename(original_ms)
    nvme_staging_root = os.path.join(nvme_base_dir, task_id)
    nvme_ms = os.path.join(nvme_staging_root, ms_name)

    os.makedirs(nvme_staging_root, exist_ok=True)
    if not os.path.isdir(nvme_ms):
        shutil.copytree(original_ms, nvme_ms)
    return nvme_ms



@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    retry_kwargs={'max_retries': 4},
    acks_late=True,
)
def run_nvme_reduce_all_unconditional(
    self,
    vis: str,
    *,
    chanbin: int = 4,
    nvme_base_dir: str = '/fast/pipeline',
    cleanup_nvme: bool = True,
) -> str:
    """
    Process one MS on NVMe (no LST/transit gating), never touch/delete original.
    Output:
      /lustre/pipeline/night-time/averaged/{subband}/{date}/{hour}/{BASE}_averaged.ms
      /lustre/pipeline/night-time/averaged/{subband}/{date}/{hour}/{BASE}_flagmeta.bin
    """

    # Final Lustre destinations
    final_output_dir, ms_base = build_output_paths(
        vis, base_output_dir='/lustre/pipeline/night-time/averaged/'
    )
    final_averaged_ms = os.path.join(final_output_dir, f"{ms_base}_averaged.ms")
    orig_base_noext = os.path.splitext(os.path.basename(vis))[0]
    final_flag_meta  = os.path.join(final_output_dir, f"{orig_base_noext}_flagmeta.bin")

    # Idempotent skip
    if os.path.isdir(final_averaged_ms) and os.path.isfile(final_flag_meta):
        logging.info(f"[NVMe unconditional] Already exists, skipping: {final_averaged_ms}")
        return final_averaged_ms

    # Unique NVMe staging (avoid clashes across Calum servers)
    task_id = getattr(self, 'request', None) and self.request.id or str(uuid.uuid4())
    nvme_ms = _copy_ms_to_nvme_unique(vis, nvme_base_dir, task_id)
    nvme_staging_root = os.path.dirname(nvme_ms)  # /fast/pipeline/<task_id>

    try:
        # 1) Flag on NVMe
        strategy = get_aoflagger_strategy("LWA_opt_GH1.lua")
        logging.info(f"[NVMe unconditional] Flagging {nvme_ms} with {strategy}")
        flagged_ms = flag_with_aoflagger(ms=nvme_ms, strategy=strategy, in_memory=False, n_threads=1)

        # 2) Save flag metadata (wherever your helper writes it)
        try:
            from orca.transform.flagging import save_flag_metadata as _save_flagmeta
            _save_flagmeta(flagged_ms)
        except Exception as meta_e:
            logging.warning(f"[NVMe unconditional] save_flag_metadata error: {meta_e}")

        # 3) Average on NVMe — **pass output_vis** (fixes the TypeError)
        nvme_averaged_out = os.path.join(nvme_staging_root, f"{orig_base_noext}_averaged.ms")
        logging.info(f"[NVMe unconditional] Averaging -> {nvme_averaged_out} (chanbin={chanbin})")
        average_frequency(flagged_ms, nvme_averaged_out, chanbin=chanbin)  # <-- corrected call
        averaged_ms_on_nvme = nvme_averaged_out

        # 4) Move outputs to final Lustre folder
        os.makedirs(final_output_dir, exist_ok=True)

        if os.path.isdir(final_averaged_ms):
            shutil.rmtree(final_averaged_ms, ignore_errors=True)
        logging.info(f"[NVMe unconditional] Moving averaged MS -> {final_averaged_ms}")
        shutil.move(averaged_ms_on_nvme, final_averaged_ms)

        # Flagmeta may be on NVMe or (by your helper) in slow-averaged; move it next to the MS
        nvme_flagmeta  = os.path.join(nvme_staging_root, f"{orig_base_noext}_flagmeta.bin")
        slow_flagmeta  = os.path.join('/lustre/pipeline/slow-averaged', f"{orig_base_noext}_flagmeta.bin")

        src_flagmeta = None
        if os.path.exists(nvme_flagmeta):
            src_flagmeta = nvme_flagmeta
        elif os.path.exists(slow_flagmeta):
            src_flagmeta = slow_flagmeta

        if src_flagmeta:
            if os.path.exists(final_flag_meta):
                os.remove(final_flag_meta)
            logging.info(f"[NVMe unconditional] Moving flagmeta -> {final_flag_meta}")
            shutil.move(src_flagmeta, final_flag_meta)
        else:
            logging.warning(f"[NVMe unconditional] Flagmeta not found in NVMe or slow-averaged for base={orig_base_noext}")

        return final_averaged_ms

    except Exception as e:
        logging.exception(f"[NVMe unconditional] Failed for {vis}: {e}")
        raise

    finally:
        if cleanup_nvme:
            try:
                shutil.rmtree(nvme_staging_root, ignore_errors=True)
            except Exception as ce:
                logging.warning(f"[NVMe unconditional] NVMe cleanup warning ({nvme_staging_root}): {ce}")

# --- Zesting: stage -> applycal -> TTCal zest -> move; params passed by submitter ---
def _zesting_stage_to_nvme(ms_path: str, nvme_root: str, unique: bool = False) -> Tuple[str, str]:
    """
    Copy MS directory to NVMe staging.
    If unique=True, stage under <nvme_root>/<ms_basename>__<ts>_<pid>/<ms_basename>
    Else, stage under <nvme_root>/<ms_basename>/

    Returns (staged_ms, staging_root).
    """
    ms_base = os.path.basename(ms_path.rstrip("/"))

    if unique:
        ts = str(int(time.time()))         # uses existing imports: time, os
        suffix = f"__{ts}_{os.getpid()}"
        staging_root = os.path.join(nvme_root, ms_base + suffix)
        staged_ms    = os.path.join(staging_root, ms_base)
    else:
        staging_root = os.path.join(nvme_root, ms_base)
        staged_ms    = staging_root  # MS lives directly at this path

    os.makedirs(staging_root, exist_ok=True)

    if os.path.exists(staged_ms):
        shutil.rmtree(staged_ms, ignore_errors=True)

    shutil.copytree(ms_path, staged_ms)
    return staged_ms, staging_root


def _zesting_dest_path(ms_src: str, subband: str, obs_date: str, hour: str, dest_root: str) -> str:
    """
    Destination: <dest_root>/<subband>/<date>/<hour>/<ms_basename>/
    """
    ms_name = os.path.basename(ms_src.rstrip("/"))
    dest_dir = os.path.join(dest_root, subband, obs_date, hour)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, ms_name)

def _cleanup_flagversions(staged_ms: str) -> None:
    """
    Remove CASA '<ms>.flagversions' directory created next to the staged MS.
    Works for both layouts:
      - unique=True:  <nvme_root>/<ms_base>__ts_pid/<ms_base>  -> sibling: .../<ms_base>.flagversions
      - unique=False: <nvme_root>/<ms_base>                    -> sibling: <nvme_root>/<ms_base>.flagversions
    """
    parent = os.path.dirname(staged_ms)
    ms_base = os.path.basename(staged_ms.rstrip("/"))
    fv = os.path.join(parent, f"{ms_base}.flagversions")
    if os.path.isdir(fv):
        try:
            shutil.rmtree(fv, ignore_errors=True)
            logging.info(f"[ZEST] Removed flagversions: {fv}")
        except Exception as ce:
            logging.warning(f"[ZEST] Could not remove flagversions {fv}: {ce}")


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
)
def zesting_one_ms_task(
    self,
    ms_path: str,
    subband: str,
    obs_date: str,
    hour: str,
    *,
    bandpass_table: str,
    sources_json: str,
    nvme_root: str = "/fast/pipeline/peel",
    dest_root: str = "/lustre/pipeline/peel_test",
    nvme_unique: bool = False,   # set True if same-basename tasks might run concurrently
) -> str:
    """
    Stage MS to NVMe, CASA applycal(bandpass_table), TTCal zest(sources_json),
    then move to <dest_root>/<sb>/<date>/<hour>. Cleans NVMe on failure. Original untouched.
    """
    from casatasks import applycal
    from orca.wrapper.ttcal import zest_with_ttcal

    staged_ms = None
    staging_root = None

    try:
        # 1) Stage
        staged_ms, staging_root = _zesting_stage_to_nvme(ms_path, nvme_root, unique=nvme_unique)
        logging.info(f"[ZEST] Staged: {staged_ms}")

        # 2) Apply calibration
        logging.info(f"[ZEST] applycal -> {staged_ms} using {bandpass_table}")
        applycal(
            vis=staged_ms,
            gaintable=[bandpass_table],
            calwt=[False],
            flagbackup=True,
        )

        # 3) TTCal zest
        logging.info(f"[ZEST] TTCal zest -> {staged_ms} (sources={sources_json})")
        zest_with_ttcal(
            ms=staged_ms,
            sources=sources_json,
            beam="constant",
            minuvw=10,
            maxiter=30,
            tolerance="1e-4",
        )

        # 4) Move to destination
        dest_ms = _zesting_dest_path(ms_path, subband, obs_date, hour, dest_root)
        if os.path.exists(dest_ms):
            shutil.rmtree(dest_ms, ignore_errors=True)
        logging.info(f"[ZEST] Moving -> {dest_ms}")
        shutil.move(staged_ms, dest_ms)
        
        # Remove any leftover '<ms>.flagversions' beside the staged MS
        _cleanup_flagversions(staged_ms)
        
        # best-effort cleanup of an empty staging root (when unique=True, root != staged_ms)
        if staging_root and os.path.isdir(staging_root):
            shutil.rmtree(staging_root, ignore_errors=True)

        return dest_ms

    except Exception as e:
        logging.exception(f"[ZEST] Failed for {ms_path}: {e}")
        # Clean staged copy only
        try:
            if staged_ms and os.path.exists(staged_ms):
                shutil.rmtree(staged_ms, ignore_errors=True)
                _cleanup_flagversions(staged_ms)
            if staging_root and os.path.isdir(staging_root):
                shutil.rmtree(staging_root, ignore_errors=True)
        except Exception as ce:
            logging.warning(f"[ZEST] Cleanup warning ({staging_root}): {ce}")
        raise

