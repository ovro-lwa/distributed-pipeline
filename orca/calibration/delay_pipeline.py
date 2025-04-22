import argparse, os, shutil, glob, logging
from datetime import timedelta
import re
from datetime import datetime
import numpy as np
import astropy.units as u
from astropy.time import Time
from casatasks import concat, clearcal, ft, gaincal, mstransform
from casatools import msmetadata
from orca.utils.calibratormodel import model_generation
from orca.transform.flagging import flag_with_aoflagger, flag_ants
from orca.utils.flagutils import get_bad_antenna_numbers
from orca.utils.msfix import concat_issue_fieldid
from orca.utils.calibrationutils import get_lst_from_filename
from orca.transform.qa_plotting import plot_delay_vs_antenna
from orca.utils.paths import get_aoflagger_strategy






def closest_ms_by_lst(ms_list, target_lst_rad):
    """Return the MS whose LST is closest to the target LST in radians."""
    best_ms, best_diff = None, float("inf")
    for vis in ms_list:
        try:
            lst = get_lst_from_filename(vis)
            lst_rad = lst.to(u.rad).value
            diff = abs((lst_rad - target_lst_rad + np.pi) % (2 * np.pi) - np.pi)  # angular diff
            if diff < best_diff:
                best_ms, best_diff = vis, diff
        except Exception as e:
            print(f"Warning: failed to parse LST from {vis}: {e}")
            continue
    return best_ms

def copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def run_delay_pipeline(obs_date, ref_lst=20.00554072/24*2*3.14159265359, tol_min=1):
    """
    Run delay calibration for all frequencies >= 41 MHz for a given date.

    This function:
        1. Selects the best MS for each frequency based on proximity to a reference LST.
        2. Copies selected MS files to NVMe.
        3. Concatenates and combines SPWs using mstransform.
        4. Flags RFI using AOFlagger and known bad antennas.
        5. Generates a model (Cyg A only) and applies it.
        6. Runs gaincal to solve for delays.
        7. Outputs:
              - CASA delay calibration table at /lustre/pipeline/calibration/delay/<DATE>/
              - A QA PDF showing per-antenna delays.
        8. Cleans up NVMe temporary files.

    Args:
        obs_date (str): Date in 'YYYY-MM-DD' format.
        ref_lst (float): Reference LST (in radians). Default is Cyg A transit.
        tol_min (int): Reserved for future LST filtering tolerance (minutes).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    base = f"/lustre/pipeline/calibration"
    out_delay_dir = f"{base}/delay/{obs_date}"
    os.makedirs(out_delay_dir, exist_ok=True)

    # 1. gather candidate MSes 
    freqs = [d for d in os.listdir(base) if d.endswith("MHz") and
             int(d.rstrip("MHz")) >= 41]
    logging.info(f"Processing frequencies: {', '.join(freqs)}")

    picked = []
    for f in freqs:
        day_dir = os.path.join(base, f, obs_date)
        if not os.path.isdir(day_dir):
            logging.warning(f"no {day_dir}; skipping")
            continue
        hour_dirs = sorted(glob.glob(os.path.join(day_dir, "*")))
        ms_per_hour = [glob.glob(os.path.join(h, "*.ms")) for h in hour_dirs]
        ms_flat = [m for sub in ms_per_hour for m in sub]
        if not ms_flat:
            continue
        best = closest_ms_by_lst(ms_flat, ref_lst)
        picked.append(best)
        logging.info(f"{f}: picked {os.path.basename(best)}")
    
    # Convert tolerance (in minutes) to radians
    tol_rad = (tol_min / 60) * (2 * np.pi / 24)

    filtered = []
    for vis in picked:
        try:
            lst = get_lst_from_filename(vis).to(u.rad).value
            diff = abs((lst - ref_lst + np.pi) % (2 * np.pi) - np.pi)
            if diff <= tol_rad:
                filtered.append(vis)
            else:
                logging.warning(f"{os.path.basename(vis)} excluded — LST diff {diff * 24 / (2 * np.pi):.3f} min exceeds tolerance of {tol_min} min")
        except Exception as e:
            logging.warning(f"Failed to get LST for {vis}: {e}")

    picked = filtered

    if not picked:
        raise RuntimeError("No MSes matched the requested date/LST")

    # 2. copy to NVMe 
    scratch = f"/fast/pipeline/{obs_date}/delay_tmp"
    os.makedirs(scratch, exist_ok=True)
    local_ms = []
    for vis in picked:
        dst = os.path.join(scratch, os.path.basename(vis))
        copytree(vis, dst)
        local_ms.append(dst)
    # Extract datetime from first MS filename
    basename = os.path.basename(local_ms[0])
    match = re.search(r"(\d{8}_\d{6})", basename)
    utc_str = match.group(1) if match else None
    utc_time = datetime.strptime(utc_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")



    # 3. concat + mstransform 
    concat_ms = os.path.join(scratch, "delay_concat.ms")
    concat(vis=local_ms, concatvis=concat_ms, timesort=True)
    logging.info(f"Concatenated {len(local_ms)} MSes into {concat_ms}") 

    # combine SPWs so delay solve sees a wide band
    concat_tf = os.path.join(scratch, "delay_concat_tf.ms")
    mstransform(vis=concat_ms, outputvis=concat_tf,
                datacolumn='data', combinespws=True)
    shutil.rmtree(concat_ms)
    logging.info(f"Transformed to TF: {concat_tf}")


    # 4. flagging 
    ants = get_bad_antenna_numbers(utc_time) 
    strategy_path = get_aoflagger_strategy("LWA_opt_GH1.lua")

    logging.info(f"Bad antennas: {ants}")
    logging.info(f"Using AOFlagger strategy: {strategy_path}")
    flag_with_aoflagger(concat_tf, strategy=strategy_path)
    flag_ants(concat_tf, ants)
    logging.info(f"Flagged {concat_tf} with AOFlagger and bad antennas")

    # 5. model (Cyg A only) + gaincal 
    ms_model = model_generation(concat_tf)
    ms_model.primary_beam_model = "/lustre/ai/beam_testing/OVRO-LWA_soil_pt.h5"
    cl_path, _ = ms_model.gen_model_cl(included_sources=["CygA"])
    logging.info(f"Generated model component list: {cl_path}")


    clearcal(vis=concat_tf, addmodel=True)
    ft(vis=concat_tf, complist=cl_path, usescratch=True)
    logging.info(f"Applied model to {concat_tf}")

    delay_tab = os.path.join(out_delay_dir,
                             f"{obs_date.replace('-','')}_delay.delay")
    gaincal(vis=concat_tf,
            caltable=delay_tab,
            uvrange='>10lambda,<125lambda',
            solint='inf',
            refant='202',
            minblperant=4,
            minsnr=3.0,
            gaintype='K', calmode='ap',
            normtype='mean', solnorm=False,
            parang=False)
    logging.info(f"Delay table written to {delay_tab}")

    # 6. generate QA plot and clean NVMe scratch 
    qa_pdf = delay_tab.replace(".delay", "_vs_antenna.pdf")
    plot_delay_vs_antenna(delay_tab, output_pdf=qa_pdf)
    logging.info(f"Saved QA plot to {qa_pdf}")

    # Remove scratch dir
    shutil.rmtree(scratch)
    parent_dir = os.path.dirname(scratch)
    if not os.listdir(parent_dir):
        shutil.rmtree(parent_dir)
        logging.info(f"Removed empty parent directory: {parent_dir}")

    return delay_tab
        


