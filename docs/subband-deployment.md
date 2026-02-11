# Subband Pipeline — Deployment & Testing Guide

> Step-by-step instructions for deploying and testing the Celery-based subband
> pipeline on the OVRO-LWA calim cluster.
>
> For architecture and file reference see [docs/subband-pipeline.md](subband-pipeline.md).

---

## Table of Contents

- [1. Prerequisites (one-time setup)](#1-prerequisites-one-time-setup)
- [2. Verify input data](#2-verify-input-data)
- [3. Start a Celery worker](#3-start-a-celery-worker)
- [4. Dry run](#4-dry-run)
- [5. Run for real (single subband, single LST hour)](#5-run-for-real-single-subband-single-lst-hour)
- [6. Monitor progress](#6-monitor-progress)
- [7. Verify output](#7-verify-output)
- [8. Clean up NVMe](#8-clean-up-nvme)
- [9. Scaling up](#9-scaling-up)
- [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites (one-time setup)

**a) Install the package on the worker node(s) in editable mode:**

```bash
ssh lwacalim08                          # or whichever node you're testing on
conda activate py38_orca_nkosogor       # the pipeline conda env
cd /path/to/distributed-pipeline
git checkout feature/celery-subband-integration
pip install -e .
```

**b) Ensure `~/orca-conf.yml` exists** on every node that will run a worker.
It must contain valid RabbitMQ and Redis URIs:

```yaml
queue:
  prefix: default
  broker_uri: pyamqp://user:pass@rabbitmq.calim.mcs.pvt:5672/vhost
  result_backend_uri: redis://10.41.0.85:6379/0
```

If one already exists from a previous orca setup, it will work — just make sure
the `broker_uri` and `result_backend_uri` are correct. If you want to use a
separate vhost for testing, create one in RabbitMQ first.

**c) Verify external tools are available** on the worker node:

```bash
which wsclean        # should be /opt/bin/wsclean or similar
which aoflagger      # should be /opt/bin/aoflagger
which chgcentre      # should be /opt/bin/chgcentre
```

TTCal conda envs (`julia060`, `ttcal_dev`) should already be set up on the
calim nodes. Verify with:

```bash
conda env list | grep -E 'julia060|ttcal_dev'
```

---

## 2. Verify input data

73 MHz archive data lives at `/lustre/pipeline/night-time/averaged/73MHz/<date>/<hour>/`.

```bash
# Pick a date that has data:
ls /lustre/pipeline/night-time/averaged/73MHz/
# e.g. 2025-06-15

# Check which hours have data:
ls /lustre/pipeline/night-time/averaged/73MHz/2025-06-15/
# e.g. 03/ 04/ 05/ ... 14/ 15/ ...

# Count MS files in one hour:
ls /lustre/pipeline/night-time/averaged/73MHz/2025-06-15/14/*.ms | wc -l
# Typically ~60 files at 1-min cadence
```

Verify calibration tables exist:

```bash
ls /lustre/gh/calibration/pipeline/bandpass/73MHz/latest.bandpass
ls /lustre/gh/calibration/pipeline/xy/73MHz/latest.X
```

> **Tip:** Adjust paths to match your actual calibration table locations.
> The `--bp_table` and `--xy_table` arguments accept any valid path.

---

## 3. Start a Celery worker

SSH into the node that owns the subband you want to test. For 73 MHz that's
**lwacalim08**:

```bash
ssh lwacalim08
conda activate py38_orca_nkosogor
cd /path/to/distributed-pipeline

celery -A orca.celery worker \
    -Q calim08 \
    --hostname=calim08@lwacalim08 \
    -c 4 \
    --loglevel=INFO
```

- `-Q calim08` — listen only on the calim08 queue (73 MHz lives here)
- `-c 4` — 4 concurrent Phase 1 tasks (4 MS files processed in parallel)
- Use `-c 2` for a lighter test, or `-c 8` if you want more parallelism

**Leave this terminal running.** The worker will print task logs in real time.

---

## 4. Dry run

In a **second terminal** (any calim node or lwacalimhead — the submission
script only talks to RabbitMQ, it doesn't need to be on calim08):

```bash
conda activate py38_orca_nkosogor
cd /path/to/distributed-pipeline

python pipeline/subband_celery.py \
    --range 14-15 \
    --date 2025-06-15 \
    --bp_table /lustre/gh/calibration/pipeline/bandpass/73MHz/latest.bandpass \
    --xy_table /lustre/gh/calibration/pipeline/xy/73MHz/latest.X \
    --subbands 73MHz \
    --peel_sky --peel_rfi \
    --dry_run
```

This **does not submit any tasks**. It will print:

- How many MS files it found
- The LST segment label (e.g. `14h`)
- The target queue (`calim08`)
- Each MS file path it would process

**Check that:**
- ✅ The file count looks right (~60 for one hour)
- ✅ The queue is `calim08`
- ✅ The LST label matches your expectation
- ✅ No `"No files for 73MHz"` warnings

---

## 5. Run for real (single subband, single LST hour)

```bash
python pipeline/subband_celery.py \
    --range 14-15 \
    --date 2025-06-15 \
    --bp_table /lustre/gh/calibration/pipeline/bandpass/73MHz/latest.bandpass \
    --xy_table /lustre/gh/calibration/pipeline/xy/73MHz/latest.X \
    --subbands 73MHz \
    --peel_sky --peel_rfi \
    --skip_cleanup \
    --run_label test_73MHz_1hr
```

Key flags explained:

| Flag | Why |
|------|-----|
| `--subbands 73MHz` | Only process this one subband |
| `--range 14-15` | One LST hour (adjust to match your data) |
| `--skip_cleanup` | **Keep intermediate files on NVMe** for inspection |
| `--run_label test_73MHz_1hr` | Human-readable label for finding output |
| `--peel_sky --peel_rfi` | Enable source and RFI peeling |

Without `--skip_cleanup`, intermediate MS files are deleted from NVMe after
concatenation to save space. Use it for debugging.

To also run hot-baseline diagnostics, add `--hot_baselines`.

---

## 6. Monitor progress

### Option A — Watch the worker terminal

The worker terminal (step 3) shows real-time logs:

```
Phase 1 START on lwacalim08: 20250615_140000_73MHz_averaged.ms
Phase 1 DONE: 20250615_140000_73MHz_averaged.ms
...
Phase 2 START on lwacalim08: 73MHz (58 files)
Concatenating MS files...
Running AOFlagger...
Starting Science Imaging for 73MHz...
Phase 2 COMPLETE: 73MHz → /lustre/pipeline/14h/2025-06-15/test_73MHz_1hr/73MHz
```

### Option B — Flower web dashboard

Start Flower on any node:

```bash
celery -A orca.celery flower --port=5555
```

SSH tunnel from your laptop:

```bash
ssh -L 5555:localhost:5555 you@lwacalim10
```

Open http://localhost:5555 — see every task, its state, timing, and any errors.

### Option C — Check from Python

```python
from orca.celery import app

# Use the AsyncResult ID from the submission output or from Flower
from celery.result import AsyncResult
result = AsyncResult('task-id-here', app=app)

print(result.status)   # PENDING, STARTED, SUCCESS, FAILURE
print(result.result)   # Return value (archive path) on SUCCESS
```

---

## 7. Verify output

After the pipeline completes:

```bash
# NVMe working directory (kept because --skip_cleanup was used):
ls /fast/pipeline/14h/2025-06-15/test_73MHz_1hr/73MHz/

# Should contain:
#   73MHz_concat.ms     ← concatenated MS
#   I/deep/             ← 4 deep Stokes I image sets
#   I/10min/            ← 10-min interval Stokes I images
#   V/deep/             ← Stokes V deep image
#   V/10min/            ← Stokes V 10-min interval images
#   snapshots/          ← pilot snapshot images
#   QA/                 ← diagnostic plots (hot baselines, snapshot stats)
#   logs/               ← CASA log

# Lustre archive (always written):
ls /lustre/pipeline/14h/2025-06-15/test_73MHz_1hr/73MHz/

# Check a specific FITS image:
ls /lustre/pipeline/14h/2025-06-15/test_73MHz_1hr/73MHz/I/deep/*fits | head

# Quick sanity check — does the image have the right size?
python -c "
from astropy.io import fits
hdu = fits.open('/lustre/pipeline/14h/2025-06-15/test_73MHz_1hr/73MHz/I/deep/73MHz-I-Deep-Taper-Robust-0-image.fits')
print(hdu[0].data.shape)   # expect (1, 1, 4096, 4096) or similar
print(hdu[0].header['CRVAL1'], hdu[0].header['CRVAL2'])  # RA, Dec of phase centre
"
```

---

## 8. Clean up NVMe

Once you're satisfied the output looks correct:

```bash
rm -rf /fast/pipeline/14h/2025-06-15/test_73MHz_1hr/
```

NVMe space is limited — always clean up after testing.

---

## 9. Scaling up

### Multiple LST hours (same subband)

```bash
python pipeline/subband_celery.py \
    --range 13-19 --date 2025-06-15 \
    --bp_table ... --xy_table ... \
    --subbands 73MHz \
    --peel_sky --peel_rfi
```

This submits one chord per LST hour (13h, 14h, 15h, 16h, 17h, 18h). They
queue up on `calim08` and execute sequentially (one chord at a time since
Phase 2 is resource-intensive).

### Multiple subbands

Start workers on each calim node, then submit without `--subbands`:

```bash
# On each calimNN node, start a worker:
# (or use the systemd service if configured)
ssh lwacalim07 "cd /path/to/repo && conda activate py38_orca_nkosogor && \
    celery -A orca.celery worker -Q calim07 --hostname=calim07@lwacalim07 -c 4 &"

ssh lwacalim08 "..."
ssh lwacalim09 "..."

# Submit all subbands:
python pipeline/subband_celery.py \
    --range 14-15 --date 2025-06-15 \
    --bp_table ... --xy_table ... \
    --peel_sky --peel_rfi
```

Each subband gets routed to the correct node automatically.

### Full observation

```bash
python pipeline/subband_celery.py \
    --range 13-19 --date 2025-06-15 \
    --bp_table /lustre/gh/calibration/pipeline/bandpass/latest.bandpass \
    --xy_table /lustre/gh/calibration/pipeline/xy/latest.X \
    --peel_sky --peel_rfi --hot_baselines
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Worker never picks up tasks | Wrong queue name | Verify worker listens on `calim08` (check `-Q` flag) |
| `FileNotFoundError: orca-conf.yml` | Missing config on worker node | Copy `~/orca-conf.yml` to the worker's home dir |
| `No files for 73MHz in 14h` | No data for that date/hour | Check `/lustre/pipeline/night-time/averaged/73MHz/<date>/<hour>/` exists |
| Calibration fails | Bad cal table path or SPW mismatch | Check paths; inspect `logs/casa_pipeline.log` on NVMe |
| TTCal / peeling fails | Conda env missing | Run `conda env list` on worker — need `julia060` and `ttcal_dev` |
| `wsclean: command not found` | WSClean not on PATH | Set `export WSCLEAN_BIN=/opt/bin/wsclean` or check install |
| Worker OOM killed | Too much concurrency | Reduce `-c` (e.g. `-c 2`), or check `mem` in imaging config |
| `Connection refused` on broker | RabbitMQ down or wrong URI | Check `~/orca-conf.yml` broker_uri; verify RabbitMQ is running |
| Task stuck in PENDING | Worker not running or queue mismatch | Start worker; confirm queue matches `get_queue_for_subband()` |
| Phase 2 never starts | A Phase 1 task failed all retries | Check Flower for failed tasks; fix and resubmit |

### Useful debug commands

```bash
# Check RabbitMQ queues:
rabbitmqctl list_queues name messages consumers

# Check Celery cluster status:
celery -A orca.celery inspect active

# Check registered tasks:
celery -A orca.celery inspect registered

# Purge a queue (careful!):
celery -A orca.celery purge -Q calim08

# Check NVMe usage:
df -h /fast/
```
