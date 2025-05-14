#!/usr/bin/env python
"""
Generate delay and band‑pass calibration products for a given date.

Assumes all MS files are already present on Lustre.
"""

import os, glob, time, shutil, argparse
from collections import defaultdict
from orca.calibration.delay_pipeline import run_delay_pipeline
from orca.tasks.pipeline_tasks import bandpass_nvme_task

FAST_ROOT = "/fast/pipeline"
CAL_ROOT  = "/lustre/pipeline/calibration"

def _group_24(ms_paths):
    """Return lists of exactly 24 MS paths (may span two consecutive hours)."""
    by_hour = defaultdict(list)
    for p in ms_paths:
        hour_dir = os.path.basename(os.path.dirname(p)).zfill(2)  # zero-pad to 2 digits
        by_hour[hour_dir].append(p)
        #by_hour[os.path.basename(os.path.dirname(p))].append(p)
    
    hours = sorted(by_hour)
    blocks, i = [], 0
    while i < len(hours):
        cur = by_hour[hours[i]]
        if len(cur) == 24:
            blocks.append(cur); i += 1
        elif len(cur) < 24 and i + 1 < len(hours):
            merged = cur + by_hour[hours[i + 1]]
            if len(merged) == 24:
                blocks.append(merged); i += 2
            else:
                i += 1
        else:
            i += 1
    return blocks

def _clean_tmp(date, freq, hour):
    tmp = os.path.join(FAST_ROOT, date, f"{freq}_{hour}_bandpass_tmp")
    shutil.rmtree(tmp, ignore_errors=True)

def main(obs_date, retries=3):
    delay_tab = run_delay_pipeline(obs_date,  tol_min=0.1)
    print(f"Delay table ready: {delay_tab}")

    for freq in sorted(
    d for d in os.listdir(CAL_ROOT)
    if d.endswith("MHz") and int(d.rstrip("MHz")) >= 41
    ):
    #for freq in sorted(d for d in os.listdir(CAL_ROOT) if d.endswith("MHz")):
        freq_dir = os.path.join(CAL_ROOT, freq, obs_date)
        ms_files = glob.glob(f"{freq_dir}/*/*.ms")
        if not ms_files:
            continue

        blocks = _group_24(ms_files)
        if not blocks:
            print(f"No complete 24‑file sets for {freq}")
            continue
        
        for i, block in enumerate(blocks):
            print(f"\nSelected 24 MS files for {freq} block {i+1}:")
            for ms in block:
                print(f"  {ms}")


        pending = []
        for block in blocks:
            hour_tag = os.path.basename(block[0]).split("_")[1][:2]
            res = bandpass_nvme_task.apply_async(
                args=[block, delay_tab, obs_date], queue="bandpass")
            pending.append((hour_tag, block, res, 0))

        while pending:
            time.sleep(120)
            for hour_tag, block, res, tries in list(pending):
                if res.ready():
                    if res.successful():
                        print(f"{freq} {hour_tag} finished: {res.result}")
                        pending.remove((hour_tag, block, res, tries))
                    else:
                        _clean_tmp(obs_date, freq, hour_tag)
                        if tries < retries:
                            new_res = bandpass_nvme_task.apply_async(
                                args=[block, delay_tab, obs_date], queue="bandpass")
                            pending.remove((hour_tag, block, res, tries))
                            pending.append((hour_tag, block, new_res, tries + 1))
                        else:
                            print(f"{freq} {hour_tag} failed after retry")
                            pending.remove((hour_tag, block, res, tries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Observation date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.date)

