# orca/tasks/pipeline_tasks.py

import os
from orca.celery import app
from orca.transform.flagging import flag_ants as original_flag_ants
#from orca.transform.flagging import flag_with_aoflagger as original_flag_with_aoflagger
from orca.transform.flagging import flag_with_aoflagger, save_flag_metadata, flag_ants
from orca.transform.calibration import applycal_data_col as original_applycal_data_col
from orca.wrapper.wsclean import wsclean as original_wsclean
from orca.wrapper.ttcal import peel_with_ttcal 
from orca.transform.averagems import average_frequency
from orca.wrapper import change_phase_centre
from orca.utils.calibrationutils import build_output_paths
from typing import List
import shutil
from orca.utils.calibrationutils import is_within_transit_window, get_lst_from_filename, get_relative_path

import logging
from casatasks import concat, clearcal, ft, bandpass
from orca.utils.calibratormodel import model_generation
from orca.utils.msfix import concat_issue_fieldid
#from orca.wrapper.change_phase_centre import get_phase_center
from orca.utils.flagutils import get_bad_antenna_numbers
from orca.utils.calibrationutils import parse_filename
from orca.transform.qa_plotting import plot_bandpass_to_pdf_amp_phase
from orca.utils.paths import get_aoflagger_strategy


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
    flagged_ms = flag_with_aoflagger(ms=nvme_ms, strategy='/opt/share/aoflagger/strategies/nenufar-lite.lua', in_memory=False, n_threads=1)

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


@app.task
def run_pipeline_slow_on_one_cpu_nvme(vis: str, start: int = 1, end: int = 14, chanbin: int = 4) -> str:
    """
    A pipeline that:
    - Checks if MS is a calibrator; if yes, copy to calibration directory without removing original.
    - If MS UTC hour is in [start..end], process it on NVMe: copy to NVMe, flag, save metadata, average.
      After averaging, move results to /lustre/pipeline/slow-averaged/.
    - Do not remove the original MS from /lustre/pipeline/slow/.
    - Remove the NVMe copy after processing.
    """

    # Check if calibrator
    sources_in_window = is_within_transit_window(vis, window_minutes=4)
    if sources_in_window:
        # Copy to calibration directory
        copy_ms_to_calibration_task(vis)

    # Check the UTC hour
    utc_hour = get_utc_hour_from_path(vis)

    # Process if hour in [start..end]
    if start <= utc_hour <= end:
        # Copy to NVMe
        nvme_ms = copy_ms_to_nvme_task(vis)

        # Flag on NVMe
        flagged_ms = flag_with_aoflagger_task.run(nvme_ms, strategy='/opt/share/aoflagger/strategies/nenufar-lite.lua', in_memory=False, n_threads=1)

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
def bandpass_nvme_task(ms_list: list, delay_table: str, obs_date: str,
                       nvme_root: str = "/fast/pipeline") -> str:
    """
    Run bandpass calibration for a frequency-hour block using NVMe.

    This function performs:
        1. Copies the 24 MS files to NVMe (e.g., /fast/pipeline/).
        2. Generates and applies a full-sky beam-based model to each MS.
        3. Concatenates the MS files and fixes FIELD_ID/OBSERVATION_ID.
        4. Sets the phase center to that of the central MS.
        5. Flags RFI using AOFlagger and known bad antennas.
        6. Performs CASA bandpass calibration using the provided delay table.
        7. Generates a QA PDF showing per-antenna bandpass amplitude/phase.
        8. Moves output to:
               /lustre/pipeline/calibration/bandpass/<FREQ>/<DATE>/<HOUR>/
               with filename:
               bandpass_concat.<FREQ>_<HOUR>.bandpass
        9. Cleans up NVMe temporary space.

    Args:
        ms_list (list): List of 24 Measurement Set (MS) paths (same freq/LST).
        delay_table (str): Path to the CASA delay calibration table.
        obs_date (str): Observation date in format 'YYYY-MM-DD'.
        nvme_root (str): Root scratch directory on fast storage (/fast/pipeline by default).

    Returns:
        str: Full path to the bandpass table saved on Lustre.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    first_ms = os.path.basename(ms_list[0])
    freq_tag = first_ms.split("_")[-1].replace(".ms", "")  # e.g. 73MHz
    hour_tag = first_ms.split("_")[1][:2]                  # e.g. 22 from HHMMSS
    scratch = os.path.join(nvme_root, obs_date, f"{freq_tag}_{hour_tag}_bandpass_tmp")
    os.makedirs(scratch, exist_ok=True)
    logging.info(f"Step 0: Created NVMe scratch: {scratch}")

    # Step 1: Copy and add model to each MS

    logging.info("Step 1: Copying MS files and generating models")
    local = []
    for vis in ms_list:
        basename = os.path.basename(vis)
        dst = os.path.join(scratch, basename)
        shutil.copytree(vis, dst)
        local.append(dst)

        logging.info(f"Generating model for {basename}")
        ms_model = model_generation(dst)
        ms_model.primary_beam_model = "/lustre/ai/beam_testing/OVRO-LWA_soil_pt.h5"
        cl_path, ft_needed = ms_model.gen_model_cl()
        logging.info(f"Model component list generated: {cl_path} (FT needed: {ft_needed})")
        if ft_needed:
            clearcal(vis=dst, addmodel=True)
            ft(vis=dst, complist=cl_path, usescratch=True)
            logging.info(f"Applied model to {dst} using FT")
    logging.info(f"Copied and applied models to {len(local)} MS files inside: {scratch}")
        
    

    # Step 2: Concatenate and fix field IDs

    logging.info("Step 2: Concatenating MS files")
    concat_ms = os.path.join(scratch, "bandpass_concat.ms")
    concat(vis=local, concatvis=concat_ms, timesort=True)
    concat_issue_fieldid(concat_ms, obsid=True)
    logging.info(f"Concatenated {len(local)} MS files into: {concat_ms}")

    # Step 3: Change phase center

    center_index = len(local) // 2
    center_coord = change_phase_centre.get_phase_center(local[center_index])
    center_str = center_coord.to_string('hmsdms')
    logging.info(f"Step 3: Changing phase center to: {center_str} (from {local[center_index]})")
    change_phase_centre.change_phase_center(concat_ms, center_str)
    logging.info(f"Changed phase center of {concat_ms} to: {center_str}")

    # Step 4: Flagging

    logging.info("Step 4: Flagging bad antennas and AOFlagger")
    utc_str = parse_filename(first_ms).replace("T", " ")
    ants = get_bad_antenna_numbers(utc_str)

    strategy_path = get_aoflagger_strategy("LWA_opt_GH1.lua") 
    flag_with_aoflagger(concat_ms, strategy=strategy_path)
    flag_ants(concat_ms, ants)
    logging.info(f"Flagged {concat_ms} with AOFlagger and bad antennas: {ants}")

    # Step 5: Bandpass Calibration

    logging.info("Step 5: Running bandpass calibration")
    bp_tab = concat_ms.rstrip(".ms") + f".{freq_tag}_{hour_tag}.bandpass"
    logging.info(f"Bandpass table will be written to: {bp_tab}")
    logging.info("Using delay table: " + delay_table)
    bandpass(
        vis=concat_ms,
        caltable=bp_tab,
        field='', spw='', intent='',
        selectdata=True,
        timerange='', uvrange='>10lambda,<125lambda',
        antenna='', scan='', observation='', msselect='',
        solint='inf', combine='obs,scan,field',
        refant='202', minblperant=4, minsnr=3.0,
        solnorm=False, bandtype='B',
        degamp=3, degphase=3,
        gaintable=delay_table
    )
    logging.info(f"Bandpass table written: {bp_tab}")

    # Step 6: QA Plot

    logging.info("Step 6: Generating QA plot for bandpass")
    qa_pdf = bp_tab.replace(".bandpass", "_bandpass_QA.pdf")
    plot_bandpass_to_pdf_amp_phase(bp_tab, pdf_path=qa_pdf, msfile=concat_ms)
    logging.info(f"Saved QA plot to: {qa_pdf}")


    # Step 7: Move to Lustre

    logging.info("Step 7: Moving results to Lustre")
    dest_dir = os.path.join("/lustre/pipeline/calibration/bandpass", freq_tag, obs_date, hour_tag)
    os.makedirs(dest_dir, exist_ok=True)

    final_tab = os.path.join(dest_dir, os.path.basename(bp_tab))
    final_pdf = os.path.join(dest_dir, os.path.basename(qa_pdf))

    shutil.move(bp_tab, final_tab)
    shutil.move(qa_pdf, final_pdf)
    logging.info(f"Moved bandpass table to: {final_tab}")
    logging.info(f"Moved QA plot to: {final_pdf}")

    # Step 8: Cleanup

    logging.info("Step 8: Cleaning up NVMe scratch directory")
    shutil.rmtree(scratch)
    parent_dir = os.path.dirname(scratch)
    if not os.listdir(parent_dir):
        shutil.rmtree(parent_dir)

    logging.info(f"Bandpass pipeline finished for {freq_tag} at {final_tab}")
    return final_tab

