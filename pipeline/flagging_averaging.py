"""Flagging and frequency averaging pipeline.

This script applies RFI flagging with AOFlagger and frequency averaging
to measurement sets. Processes data in Celery task chains for distributed
execution.
"""
from celery import chain, group
from orca.tasks.pipeline_tasks import (
    flag_with_aoflagger_task,
    save_flag_metadata_task,
    average_frequency_task,
    copy_ms_task,
    remove_ms_task,
    extract_original_ms_task
)
import glob
import os
import time
import logging
from orca.utils.calibrationutils import is_within_transit_window

# Setup logging
logging.basicConfig(filename='pipeline_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


root_directory = "/lustre/pipeline/slow/"
freq_dir = "73MHz"  
date_subdir = "2024-12-05/00"  

directory = os.path.join(root_directory, freq_dir, date_subdir)
ms_files = sorted(glob.glob(f"{directory}/*.ms"))

all_chains = []

for vis in ms_files:
    sources_in_window = is_within_transit_window(vis, window_minutes=4)

    # Pipeline steps:
    # 1. Flag original MS in place
    # 2. Save flag metadata to slow-averaged/
    # 3. Frequency average to slow-averaged/
    # 4. If calibrator present: copy original MS to slow-averaged/
    #    If not: remove original MS.

    base_chain = chain(
        # Run AOFlagger on the original MS in slow/
        flag_with_aoflagger_task.s(vis),

        # Save flag metadata to slow-averaged/
        save_flag_metadata_task.s(),

        # Average frequency and put averaged MS in slow-averaged/
        average_frequency_task.s(chanbin=4)
    )

    if sources_in_window:
        # Calibrator present:
        # After averaging, copy the original MS to slow-averaged
        # We need to extract the original ms from the (ms, averaged_ms) tuple first.
        pipeline_chain = base_chain | extract_original_ms_task.s() | copy_ms_task.s()
    else:
        # No calibrator:
        pipeline_chain = base_chain 

    all_chains.append(pipeline_chain)

if __name__ == "__main__":
    if all_chains:
        print("Submitting tasks for all MS files...")
        group_result = group(all_chains)()

        # Wait for all tasks to complete
        print("Waiting for tasks to finish...")
        try:
            group_result.join()
            print("All tasks have been processed successfully!")
        except Exception as e:
            logging.error(f"Error processing tasks: {e}")
            print(f"Error processing tasks: {e}")

            # Check individual task results
            for result in group_result.results:
                if not result.successful():
                    try:
                        exc = result.result
                        task_name = result.task_id
                        logging.error(f"Task {task_name} failed with error: {exc}")
                        print(f"Task {task_name} failed with error: {exc}")
                    except Exception as sub_e:
                        logging.error(f"Unable to retrieve task result: {sub_e}")
                        print(f"Unable to retrieve task result: {sub_e}")

    else:
        print("No tasks to process.")

