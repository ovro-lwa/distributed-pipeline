from celery import chain
from orca.tasks.pipeline_tasks import (
    copy_ms_task,
    flag_with_aoflagger_task,
    save_flag_metadata_task,
    average_frequency_task,
    remove_ms_task
)
import glob
import os
from orca.utils.calibrationutils import is_within_transit_window
import shutil


# Directory with MS files
directory = "/lustre/pipeline/slow/73MHz/2024-11-29/00/"
ms_files = sorted(glob.glob(f"{directory}*.ms"))


# Parameters
strategy = '/opt/share/aoflagger/strategies/nenufar-lite.lua'
in_memory = False
n_threads = 5

for vis in ms_files:
    sources_in_window = is_within_transit_window(vis, window_minutes=4)

    # Create the pipeline: first copy, then process
    pipeline_chain = chain(
        copy_ms_task.s(vis),                            # Copies the MS asynchronously
        flag_with_aoflagger_task.s(),                   # Flags the newly copied MS
        save_flag_metadata_task.s(),                    # Saves flag metadata
        average_frequency_task.s(chanbin=4)             # Averages frequency
    )

    # If not in transit, remove the MS after processing
    if not sources_in_window:
        pipeline_chain = pipeline_chain | remove_ms_task.s()

    # Run the pipeline
    pipeline_chain.apply_async()
