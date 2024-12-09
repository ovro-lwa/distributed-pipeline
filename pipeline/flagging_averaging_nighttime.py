from celery import chain
from orca.tasks.pipeline_tasks import (
    copy_ms_nighttime_task,  
    flag_with_aoflagger_task,
    save_flag_metadata_nighttime_task,
    average_frequency_nighttime_task,
    remove_ms_task
)
import glob
import os
from orca.utils.calibrationutils import is_within_transit_window, get_lst_from_filename

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


for freq in freq_dirs:
    directory = os.path.join(root_directory, freq, "2023-11-21/06")
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

        # Run the pipeline
        pipeline_chain.apply_async()

