from glob import glob
import uuid, os, shutil
from celery import group
from orca.tasks.imaging_tasks import imaging_shared_pipeline_task   

from orca.utils.calibrationutils import parse_filename
from orca.utils.flagutils import get_bad_correlator_numbers


'''delay    = '/lustre/pipeline/calibration/delay/2025-04-19/20250419_delay.delay'
bandpass = '/lustre/pipeline/calibration/bandpass/55MHz/2025-04-19/06/bandpass_concat.55MHz_06.bandpass'
out_dir  = '/lustre/pipeline/images_test/55MHz/2025-04-19/08'
ms_files = sorted(glob("/lustre/pipeline/night-time/averaged/55MHz/2025-04-19/08/*.ms"))
qa_path  = bandpass + ".qa"
'''
def load_bad_antennas(qa_file: str) -> list:
    bad_ants = []
    if os.path.exists(qa_file):
        with open(qa_file, 'r') as f:
            for line in f:
                if line.strip().startswith("Antennas with Amp outside"):
                    # extract from this same line
                    parts = line.strip().split(":")
                    if len(parts) > 1:
                        bad_ants = [int(x) for x in parts[1].strip(" []\n").split()]
                    break
    return bad_ants

'''
bad_corr_qa = load_bad_antennas(qa_path)
first_cal_ms = sorted(glob("/lustre/pipeline/calibration/55MHz/2025-04-19/06/*.ms"))[0]
utc_str = parse_filename(os.path.basename(first_cal_ms)).replace("T", " ")
bad_corr_nums = get_bad_correlator_numbers(utc_str)+bad_corr_qa
'''

freq = "46MHz"# 59MHz  64MHz  69MHz  73MHz  78MHz  82MHz
obs_date = "2025-04-19"
imaging_hour = "05"
bandpass_hour = "06"  # bandpass is always from hour 06
print(freq,imaging_hour)
delay = f"/lustre/pipeline/calibration/delay/{obs_date}/{obs_date.replace('-', '')}_delay.delay"
bandpass = f"/lustre/pipeline/calibration/bandpass/{freq}/{obs_date}/{bandpass_hour}/bandpass_concat.{freq}_{bandpass_hour}.bandpass"
qa_path = bandpass + ".qa"
out_dir = f"/lustre/pipeline/images_test/{freq}/{obs_date}/{imaging_hour}"
ms_files = sorted(glob(f"/lustre/pipeline/night-time/averaged/{freq}/{obs_date}/{imaging_hour}/*.ms"))
first_cal_ms = sorted(glob(f"/lustre/pipeline/calibration/{freq}/{obs_date}/{bandpass_hour}/*.ms"))[0]



bad_corr_qa = load_bad_antennas(qa_path)
utc_str = parse_filename(os.path.basename(first_cal_ms)).replace("T", " ")
bad_corr_nums = get_bad_correlator_numbers(utc_str) + bad_corr_qa


#ms_files = ms_files[0:1]

batch_root = os.path.join("/fast/pipeline",
                          f"imaging-batch-{uuid.uuid4().hex[:8]}")
os.makedirs(batch_root, exist_ok=True)

jobs = group(
    imaging_shared_pipeline_task.s(
        ms, delay, bandpass, out_dir,
        workdir_root=batch_root,
        keep_full_products=(i == 0),# first MS keeps full set
        bad_corrs = bad_corr_nums
        )
    for i, ms in enumerate(ms_files)
)

result = jobs.apply_async(queue='imaging')

# optional tidy-up after all succeed
def _cleanup(_):
    shutil.rmtree(batch_root, ignore_errors=True)

result.then(_cleanup)



