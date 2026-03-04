# QA-Light Pipeline

A quick single-timestep imaging pipeline for daily quality assurance.
Takes **one 10-second MS file per subband**, applies calibration, and produces
MFS images — no concatenation, no science phases, no archival pipeline overhead.

> **This is a standalone QA tool, not part of the main Celery pipeline.**

## Quick Start

```bash
# 1. Edit config.sh — set OBS_DATE, UTC_HOUR, BP_TABLE
vi config.sh

# 2. Run everything (fetch → flag → cal → image → cleanup)
./run_qa_light.sh

# 3. Output appears at:
#    /lustre/pipeline/qa_light_images/<DATE>/<HH>/lower-band-QA-MFS-MFS-image.fits
#    /lustre/pipeline/qa_light_images/<DATE>/<HH>/upper-band-QA-MFS-MFS-image.fits
```

## Configuration (`config.sh`)

| Variable | Description | Example |
|----------|-------------|---------|
| `OBS_DATE` | Slow-data date | `2026-02-26` |
| `UTC_HOUR` | UTC hour directory | `03` |
| `BP_TABLE` | Full path to `.B.flagged` bandpass table | `/lustre/pipeline/calibration/results/2026-02-20/05h/successful/20260226_172229/tables/calibration_2026-02-20_05h.B.flagged` |
| `FLAG_BAD_ANTS` | Flag bad antennas via mnc_python | `true` / `false` |
| `RUN_AOFLAGGER` | Run AOFlagger RFI flagging | `true` / `false` |
| `TIMESTEP` | Specific MS to pick (empty = first available) | `20260226_030500` |

## Partial Runs

```bash
./run_qa_light.sh --lower-only     # Only lower band (18–41 MHz)
./run_qa_light.sh --fetch-only     # Fetch data only, no cal/imaging
./run_qa_light.sh --skip-fetch     # Skip fetch (data already on NVMe)
./run_qa_light.sh --skip-imaging   # Fetch + cal only, no imaging
```

## Pipeline Steps

| Script | What it does |
|--------|-------------|
| `01_fetch_data.sh` | Copy one MS per subband from Lustre slow-data to NVMe |
| `02_apply_cal.py` | Flag bad antennas + apply bandpass calibration (CASA) |
| `03_image_lower_band.sh` | WSClean MFS imaging of lower subbands (18–41 MHz) |
| `04_image_upper_band.sh` | WSClean MFS imaging of upper subbands (46–82 MHz) |

## Notes

- Missing subbands (e.g. 27, 50, 64 MHz) are skipped automatically.
- NVMe working copy (`/fast/pipeline/qa_light/...`) is deleted after successful imaging.
- `BP_TABLE` must be a full path — the run-timestamp directory varies per calibration run.
- The `02_apply_cal.py` step uses the `py38_orca_nkosogor` conda env; bad-antenna flagging uses the `development` env.
