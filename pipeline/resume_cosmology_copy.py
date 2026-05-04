"""Submit resume jobs for orphaned cosmology *_copy.ms files.

Walks the cosmology source tree, finds every ``*_copy.ms`` directory under
the configured (subband, date, hour) combinations, and submits one
``resume_cosmology_copy_pipeline_on_nvme`` task per file, round-robin to a
list of target Celery queues (calim worker nodes).

Each task moves the ``_copy.ms`` from Lustre to the worker's local
``/fast/pipeline/`` scratch, AOFlags + averages it there, and writes the
final averaged MS plus flag metadata to
``/lustre/pipeline/cosmology/averaged/<subband>/<date>/<hour>/``.
The non-``_copy`` original MS is never touched.

Run from a host with the orca environment + Redis access:

    python pipeline/resume_cosmology_copy.py
"""
import glob
import os
import random

from orca.tasks.pipeline_tasks import resume_cosmology_copy_pipeline_on_nvme


ROOT = "/lustre/pipeline/cosmology"
DATE = "2026-04-19"
SUBBANDS = [
    "13MHz", "18MHz", "23MHz", "27MHz", "32MHz", "36MHz",
    "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz",
    "69MHz", "73MHz", "78MHz", "82MHz",
]
HOURS = ["05", "06", "07", "08", "11"]
TARGET_NODES = ["calim04", "calim05", "calim07"]
CHANBIN = 4


def main() -> None:
    copy_files = []
    for subband in SUBBANDS:
        for hour in HOURS:
            directory = os.path.join(ROOT, subband, DATE, hour)
            matches = sorted(glob.glob(f"{directory}/*_copy.ms"))
            print(f"{subband} {hour}: {len(matches)} _copy.ms files")
            copy_files.extend(matches)

    random.shuffle(copy_files)
    print(f"Submitting {len(copy_files)} resume tasks to nodes: {TARGET_NODES}...")

    for i, copy_ms in enumerate(copy_files):
        node = TARGET_NODES[i % len(TARGET_NODES)]
        resume_cosmology_copy_pipeline_on_nvme.s(
            copy_ms, chanbin=CHANBIN
        ).apply_async(queue=node)

    print("Done.")


if __name__ == "__main__":
    main()
