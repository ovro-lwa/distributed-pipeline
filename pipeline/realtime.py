from celery import chain
from orca.tasks.pipeline_tasks import (
    flag_ants_task,
    flag_with_aoflagger_task,
    applycal_data_col_task,
    wsclean_task,
    peel_with_ttcal_task 
)


import glob

# Specify the directory with ms files
directory = "/lustre/nkosogor/distributed_pipeline_test_data/5hour_data/"

# Collect all files with the .ms extension
ms_files = sorted(glob.glob(f"{directory}*.ms"))

# Define constants and parameters
bad_antennas = '51,79,80,117,137,193,150,178,201,208,224,183,261,211,215, 230,236,239,242,246,294,301,307,289,33,3,41,42,44,92,12,14,17,21,154,56,57,28,127,126'
bad_antennas_list = [int(ant.strip()) for ant in bad_antennas.split(',')]
bcal_file = '/lustre/nkosogor/distributed_pipeline_test_data/5hour_data/73.bandpass'
strategy = '/opt/share/aoflagger/strategies/nenufar-lite.lua'
in_memory = False
n_threads = 5
out_dir = '/lustre/nkosogor/distributed_pipeline_test_data/5hour_data/'
extra_args = [
    '-multiscale',
    '-multiscale-scale-bias', '0.8',
    '-pol', 'I',
    '-size', '4096', '4096',
    '-scale', '0.03125',
    '-niter', '0',
    '-casa-mask', '/home/pipeline/cleanmask.mask/',
    '-mgain', '0.85',
    '-weight', 'briggs', '0'
]
num_threads = 1
mem_gb = 50

for vis in ms_files:
    filename_prefix = generate_filename_prefix(vis)

    pipeline_chain = chain(
        flag_ants_task.s(vis, bad_antennas_list),
        flag_with_aoflagger_task.s(strategy=strategy, in_memory=in_memory, n_threads=n_threads),
        applycal_data_col_task.s(gaintable=bcal_file),
        peel_with_ttcal_task.s(sources='/home/pipeline/sources.json'),  # Add peeling step
        wsclean_task.s(out_dir=out_dir, filename_prefix=filename_prefix, extra_args=extra_args,
                       num_threads=num_threads, mem_gb=mem_gb)
    )
    pipeline_chain.apply_async()
