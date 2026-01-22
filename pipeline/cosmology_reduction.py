import os
import glob
import random
import time
import logging
from celery import chain, group
from orca.tasks.pipeline_tasks import split_2pol_task

def run_for_date(date_subdir):
    """
    For the given `date_subdir`, gather all hours across all frequencies, then
    process hour by hour. In each hour, tasks from all frequencies are collected
    into one list and shuffled together.
    """
    root_directory = "/lustre/pipeline/cosmology/"
    freq_dirs = [
        "41MHz", "46MHz", "50MHz", "55MHz", "59MHz",
        "64MHz", "69MHz", "73MHz", "78MHz", "82MHz"
    ]
    
    # 1) Figure out all hour directories across all frequencies.
    #    We'll create a union of all hour subdirectories found.
    all_hours = set()
    for freq in freq_dirs:
        freq_date_path = os.path.join(root_directory, freq, date_subdir)
        if not os.path.isdir(freq_date_path):
            continue
        
        possible_hours = [
            d for d in os.listdir(freq_date_path)
            if os.path.isdir(os.path.join(freq_date_path, d))
        ]
        all_hours.update(possible_hours)  # union of hours across freq
    
    # Sort the union so you can process in ascending order, if desired
    all_hours = sorted(all_hours)

    # 2) Process hour-by-hour, but gather tasks from all frequencies at once
    for hour in all_hours:
        hour_tasks = []
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting tasks at {current_time} for hour {hour} across ALL frequencies...")

        # 3) For each frequency, see if this hour exists and gather .ms files
        for freq in freq_dirs:
            freq_date_path = os.path.join(root_directory, freq, date_subdir)
            hour_dir = os.path.join(freq_date_path, hour)
            
            if not os.path.isdir(hour_dir):
                continue  # hour doesn't exist for this frequency
            
            # Collect all .ms files
            ms_files = sorted(glob.glob(os.path.join(hour_dir, "*.ms")))
            if not ms_files:
                continue

            for vis in ms_files:
                basename = os.path.basename(vis)
                
                # Skip if the file name already indicates it's reduced
                if "_2pol.ms" in basename:
                    #print(f"Skipping {vis} (already appears reduced).")
                    continue

                ms_input_stripped = vis.rstrip('/')
                base_name, _ = os.path.splitext(ms_input_stripped)
                ms_2pol = base_name + "_2pol.ms"

                # Also skip if the output already exists on disk
                if os.path.exists(ms_2pol):
                    #print(f"Skipping {vis} because {ms_2pol} already exists.")
                    continue

                # Create the pipeline task
                pipeline_chain = chain(split_2pol_task.s(vis))
                hour_tasks.append(pipeline_chain)
        
        # 4) Shuffle and run tasks for this hour, across all frequencies
        if hour_tasks:
            random.shuffle(hour_tasks)
            group_result = group(hour_tasks)()
            group_result.join()
            print(f"Completed tasks for hour {hour} across all frequencies.")
        else:
            print(f"No tasks to run for hour {hour}.")

def main():
    """
    Run the processing for a list of dates, hour by hour, but each hour includes
    all frequencies in a single shuffle group.
    """
    date_list = [#"2025-04-01","2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06", "2025-04-07", "2025-04-08", 
            "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-12"]
    #[
     #   "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14", "2025-03-15", "2025-03-16", "2025-03-17", "2025-03-18", "2025-03-19", "2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31",
        #"2025-03-19", "2025-03-20",
        #"2025-03-21", "2025-03-22"
    #]

    for date_subdir in date_list:
        print(f"Starting processing for date {date_subdir}...")
        try:
            run_for_date(date_subdir)
        except Exception as e:
            logging.error(f"Failed processing for {date_subdir}: {e}", exc_info=True)
            print(f"Error during processing {date_subdir}. Skipping to next date...")

if __name__ == "__main__":
    main()

