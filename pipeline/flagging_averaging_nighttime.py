from celery import chain, group
from orca.tasks.pipeline_tasks import (
    copy_ms_nighttime_task,  
    flag_with_aoflagger_task,
    save_flag_metadata_nighttime_task,
    average_frequency_nighttime_task,
    remove_ms_task
)
import glob
import os
import time
import logging
from orca.utils.calibrationutils import is_within_transit_window, get_lst_from_filename
import random


# Setup logging
logging.basicConfig(filename='pipeline_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def is_within_lst_range(ms: str, start_hour=11, end_hour=14) -> bool:
    """
    Check if the LST time of the measurement set is within the specified inclusive range.
    """
    lst = get_lst_from_filename(ms).hour
    return (start_hour <= lst <= end_hour)


root_directory = "/lustre/pipeline/night-time/"
freq_dirs = [
    "13MHz", "18MHz", "23MHz", "27MHz", "32MHz", "36MHz", 
    "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz", 
    "69MHz", "73MHz", "78MHz", "82MHz"
]

date_hour_subdirs = [f"2023-12-07/{str(hour).zfill(2)}" for hour in range(1, 15)]

if __name__ == "__main__":
    # Loop over each hourly subdirectory
    for idx, date_hour_subdir in enumerate(date_hour_subdirs):
        print(f"Processing {date_hour_subdir}...")

        # List to hold all chains for this hourly batch
        all_chains = []

        for freq in freq_dirs:
            directory = os.path.join(root_directory, freq, date_hour_subdir)
            ms_files = sorted(glob.glob(f"{directory}/*.ms")) 

            for vis in ms_files:
                sources_in_window = is_within_transit_window(vis, window_minutes=4)
                in_lst_range = is_within_lst_range(vis, 11, 14)

                if sources_in_window or in_lst_range:
                    # i) LST range 11-14 or calibrator
                    # - Copy the MS file 
                    # - Flag the copy
                    # - Save flag metadata
                    # - Frequency average to 96 kHz
                    # - Remove the duplicate 
                    pipeline_chain = chain(
                        copy_ms_nighttime_task.s(vis),  # Duplicate the MS 
                        flag_with_aoflagger_task.s(),  # Flag the copy
                        save_flag_metadata_nighttime_task.s(),  # Save flag metadata
                        average_frequency_nighttime_task.s(chanbin=4),  # Average to 96 kHz (chanbin 4)
                        remove_ms_task.s()  # Remove the copy after processing
                    )
                else:
                    # ii) All other LST ranges
                    # - Flag the original MS directly
                    # - Save flag metadata
                    # - Frequency average to 96 kHz
                    # - Remove the original
                    pipeline_chain = chain(
                        flag_with_aoflagger_task.s(vis),  # Flag the original MS
                        save_flag_metadata_nighttime_task.s(),  # Save flag metadata
                        average_frequency_nighttime_task.s(chanbin=4),  # Average to 96 kHz (chanbin 4)
                        remove_ms_task.s()  # Remove the original MS after processing
                    )

                # Add the pipeline chain to the list for this hour
                all_chains.append(pipeline_chain)


        if all_chains:
            random.shuffle(all_chains)

            # Run all chains as a single group for the current hourly subdirectory
            print(f"Submitting tasks for {date_hour_subdir} at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC...")
            group_result = group(all_chains)()

            # Wait for all tasks for this hour to complete
            print(f"Waiting for tasks to finish for {date_hour_subdir}...")

            try:
                group_result.join()  # This will block until all tasks are finished
                print(f"All tasks for {date_hour_subdir} have been processed successfully!")
            except Exception as e:
                logging.error(f"Error processing tasks for {date_hour_subdir}: {e}")
                print(f"Error processing tasks for {date_hour_subdir}: {e}")

                # Check individual results to log which specific tasks failed
                for result in group_result.results:
                    if not result.successful():
                        try:
                            exc = result.result  # The actual exception raised
                            task_name = result.task_id
                            logging.error(f"Task {task_name} failed in {date_hour_subdir} with error: {exc}")
                            print(f"Task {task_name} failed in {date_hour_subdir} with error: {exc}")
                        except Exception as sub_e:
                            logging.error(f"Unable to retrieve task result for {date_hour_subdir}: {sub_e}")
                            print(f"Unable to retrieve task result for {date_hour_subdir}: {sub_e}")

        else:
            print(f"No tasks to process for {date_hour_subdir}. Moving to the next hour.")

        time.sleep(5)  # Small delay before processing the next hour
