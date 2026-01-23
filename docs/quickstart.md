# Quick Start

## Starting the Pipeline

### 1. Start Celery Workers

```bash
# Start a worker for the default queue
celery -A orca.celery worker -Q default -c 4 --loglevel=info

# Start a worker for imaging tasks
celery -A orca.celery worker -Q imaging -c 2 --loglevel=info
```

### 2. Monitor with Flower

```bash
celery -A orca.celery flower --port=5555
```

Then open `http://localhost:5555` in your browser.

## Running Pipeline Scripts

### Flagging and Averaging

Process a full day of slow-cadence data:

```bash
python pipeline/flagging_averaging_all_nvme_auto_24h_LST_edges.py 2025-12-25
```

This script:
1. Scans slow data directories for the specified date
2. Filters by LST range to handle edge times properly
3. Submits Celery tasks for AOFlagger RFI flagging
4. Applies frequency averaging (192 → 48 channels)
5. Archives results to the output directory

### Dynamic Spectrum Production

Generate dynamic spectra from averaged data:

```bash
python pipeline/produce_dynspec_updated.py
```

This produces FITS cubes containing dynamic spectra across all subbands.

## Basic Usage in Python

```python
from celery import group
from orca.tasks.pipeline_tasks import copy_ms_task, flag_with_aoflagger_task

# Chain tasks
result = (
    copy_ms_task.s('/path/to/input.ms', '/output/dir/')
    | flag_with_aoflagger_task.s()
).apply_async()

# Wait for result
output_ms = result.get()
```
