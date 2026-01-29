"""Bandpass calibration pipeline.

This module provides the bandpass calibration pipeline for OVRO-LWA data.
Processes multiple measurement sets at a given frequency and LST hour to
produce a bandpass calibration table.

The pipeline:
    1. Copies MS files to NVMe for fast processing.
    2. Generates beam-based sky models and applies to each MS.
    3. Concatenates and fixes metadata issues.
    4. Flags RFI using AOFlagger and known bad antennas.
    5. Solves for bandpass using CASA bandpass task.
    6. Generates QA plots and archives results.

Example:
    >>> from orca.calibration.bandpass_pipeline import run_bandpass_calibration
    >>> run_bandpass_calibration(ms_list, delay_table, '2024-01-15')
"""
import os, shutil, logging
from casatasks import concat, clearcal, ft, bandpass
from orca.utils.calibratormodel import model_generation
from orca.utils.msfix import concat_issue_fieldid
from orca.wrapper.change_phase_centre import get_phase_center, change_phase_center
from orca.transform.flagging import flag_with_aoflagger, flag_ants
from orca.utils.flagutils import get_bad_correlator_numbers
from orca.utils.calibrationutils import parse_filename
from orca.transform.qa_plotting import plot_bandpass_to_pdf_amp_phase
from orca.utils.paths import get_aoflagger_strategy




def run_bandpass_calibration(ms_list, delay_table, obs_date, nvme_root="/fast/pipeline") -> str:
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
    center_coord = get_phase_center(local[center_index])
    center_str = center_coord.to_string('hmsdms')
    logging.info(f"Step 3: Changing phase center to: {center_str} (from {local[center_index]})")
    change_phase_center(concat_ms, center_str)
    logging.info(f"Changed phase center of {concat_ms} to: {center_str}")

    # Step 4: Flagging

    logging.info("Step 4: Flagging bad antennas and AOFlagger")
    utc_str = parse_filename(first_ms).replace("T", " ")
    bad_corr_nums = get_bad_correlator_numbers(utc_str)

    strategy_path = get_aoflagger_strategy("LWA_opt_GH1.lua") 
    flag_with_aoflagger(concat_ms, strategy=strategy_path)
    flag_ants(concat_ms, bad_corr_nums)
    logging.info(f"Flagged {concat_ms} with AOFlagger and bad corrs: {bad_corr_nums}")

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
