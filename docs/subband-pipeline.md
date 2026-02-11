# Subband Processing Pipeline — Architecture & File Reference

> **Branch:** `feature/celery-subband-integration`
>
> This document covers the Celery-based subband processing pipeline for OVRO-LWA.
> It processes per-subband data through calibration, flagging, peeling, imaging,
> and archival — all orchestrated by Celery with NVMe-local execution on the calim
> cluster.

---

## Overview

The pipeline replaces the previous Slurm + ThreadPoolExecutor approach with a
**Celery chord pattern**:

```
Phase 1 (parallel, one Celery task per MS file):
    copy to NVMe → flag bad antennas → apply calibration → peel sky → peel RFI

Phase 2 (sequential, runs once all Phase 1 tasks finish):
    concatenate → fix field IDs → change phase centre → AOFlagger →
    pilot snapshots → snapshot QA → hot-baseline removal → science imaging →
    PB correction → archive to Lustre
```

Both phases are routed to the **same per-node queue** (e.g. `calim08`) so that
all I/O stays on the node's local NVMe.

---

## File Map

### Submission & Orchestration

| File | Purpose |
|------|---------|
| `pipeline/subband_celery.py` | **CLI entry point.** Discovers MS files, computes LST segments, submits one chord per (subband, LST-hour) to the correct Celery queue. |
| `orca/tasks/subband_tasks.py` | **Celery task definitions.** Contains `prepare_one_ms_task` (Phase 1), `process_subband_task` (Phase 2), and `submit_subband_pipeline()` which wires them into a chord. |
| `orca/celery.py` | **Celery app configuration.** Defines broker/backend, all queues (`default`, `cosmology`, `bandpass`, `imaging`, `calim00`–`calim10`), and task include list. |

### Processing Logic (no Celery decorators — pure functions, testable locally)

| File | Purpose |
|------|---------|
| `orca/transform/subband_processing.py` | **All subband processing steps.** File discovery, NVMe copy, SPW map calculation, calibration application, bad-antenna flagging (via MNC subprocess), concatenation, field ID fix, snapshot QA, scan-based integration flagging, timestamp injection, PB correction dispatch, archival, subprocess helpers. |
| `orca/transform/hot_baselines.py` | **Hot-baseline & amplitude-vs-UV diagnostics.** Identifies bad antennas/baselines from a concatenated MS, produces heatmap and UV diagnostic plots, optionally flags in-place. |
| `orca/transform/pb_correction.py` | **Primary beam correction.** Applies the OVRO-LWA beam model to FITS images using `extractor_pb_75`. |

### Configuration

| File | Purpose |
|------|---------|
| `orca/resources/subband_config.py` | **All pipeline configuration in one place.** Node↔subband mapping (`NODE_SUBBAND_MAP`), NVMe/Lustre directory layout, peeling parameters (conda envs, models, args), AOFlagger strategy path, hot-baseline params, snapshot params, all 7 imaging steps (5× Stokes I + 2× Stokes V), resource allocation per node. |
| `orca/resources/system_config.py` | **Hardware mapping.** 353-entry `SYSTEM_CONFIG` dict mapping LWA antenna numbers to correlator numbers, ARX boards, SNAP2 boards, and channel assignments. Used by `hot_baselines.py`. |
| `orca/configmanager.py` | **Orca-wide config singleton.** Reads `~/orca-conf.yml` (or `default-orca-conf.yml`) for broker URI, backend URI, telescope params, executable paths. |
| `orca/default-orca-conf.yml` | Default config template. Copy to `~/orca-conf.yml` and fill in credentials. |

### Utilities & Wrappers

| File | Purpose |
|------|---------|
| `orca/utils/mnc_antennas.py` | **Standalone script** for querying bad antennas from MNC antenna health. Runs as a subprocess in the `development` conda env (because `mnc_python` is not in the pipeline env). Outputs JSON to stdout. |
| `orca/wrapper/ttcal.py` | Wrapper around TTCal.jl for peeling (both `peel` and `zest` modes). |
| `orca/wrapper/change_phase_centre.py` | Wrapper around `chgcentre` for re-phasing visibilities. |

---

## Node ↔ Subband ↔ Queue Mapping

Each subband is pinned to a specific calim node. The node's local NVMe
(`/fast/pipeline/`) is used for all intermediate I/O. Final products are
archived to shared Lustre (`/lustre/pipeline/`).

| Subband(s) | Node | Celery Queue | NVMe |
|------------|------|-------------|------|
| 18 MHz, 23 MHz | lwacalim00 | `calim00` | Shared (half resources each) |
| 27 MHz, 32 MHz | lwacalim01 | `calim01` | Shared |
| 36 MHz, 41 MHz | lwacalim02 | `calim02` | Shared |
| 46 MHz, 50 MHz | lwacalim03 | `calim03` | Shared |
| 55 MHz | lwacalim04 | `calim04` | Full node |
| 59 MHz | lwacalim05 | `calim05` | Full node |
| 64 MHz | lwacalim06 | `calim06` | Full node |
| 69 MHz | lwacalim07 | `calim07` | Full node |
| **73 MHz** | lwacalim08 | `calim08` | Full node |
| 78 MHz | lwacalim09 | `calim09` | Full node |
| 82 MHz | lwacalim10 | `calim10` | Full node |

Dual-subband nodes get 16 CPUs / 60 GB / 12 wsclean threads each.
Single-subband nodes get 32 CPUs / 120 GB / 24 wsclean threads.

---

## Directory Layout

```
NVMe (per-node, not shared):
/fast/pipeline/<lst>/<date>/<run_label>/<subband>/
    ├── *.ms                 # Individual MS files (Phase 1)
    ├── <subband>_concat.ms  # Concatenated MS (Phase 2)
    ├── I/deep/              # Stokes I deep images
    ├── I/10min/             # Stokes I 10-min interval images
    ├── V/deep/              # Stokes V deep images
    ├── V/10min/             # Stokes V 10-min interval images
    ├── snapshots/           # Pilot snapshot images + QA
    ├── QA/                  # Hot-baseline plots, diagnostics
    └── logs/                # CASA log, subprocess logs

Lustre (shared, archived products):
/lustre/pipeline/<lst>/<date>/<run_label>/<subband>/
    └── (same structure as above, minus intermediate MS files)
```

---

## Imaging Steps

The pipeline produces 7 image products per subband-hour:

| # | Stokes | Category | Suffix | Key Features |
|---|--------|----------|--------|-------------|
| 1 | I | deep | `I-Deep-Taper-Robust-0.75` | Multiscale, inner taper, robust −0.75 |
| 2 | I | deep | `I-Deep-Taper-Robust-0` | Multiscale, inner taper, robust 0 |
| 3 | I | deep | `I-Deep-NoTaper-Robust-0.75` | Multiscale, no taper, robust −0.75 |
| 4 | I | deep | `I-Deep-NoTaper-Robust-0` | Multiscale, no taper, robust 0 |
| 5 | I | 10min | `I-Taper-10min` | Multiscale, taper, 6 intervals |
| 6 | V | deep | `V-Taper-Deep` | Dirty image, taper |
| 7 | V | 10min | `V-Taper-10min` | Dirty image, taper, 6 intervals |

All images are 4096×4096 at 0.03125° scale with primary beam correction applied.

---

## External Dependencies

These must be available on the calim worker nodes:

| Tool | Purpose | Default Path |
|------|---------|-------------|
| **WSClean** | Imaging | `/opt/bin/wsclean` (or `$WSCLEAN_BIN`) |
| **AOFlagger** | RFI flagging | `/opt/bin/aoflagger` |
| **chgcentre** | Phase centre rotation | `/opt/bin/chgcentre` |
| **TTCal.jl** | Source peeling | Via conda envs `julia060` / `ttcal_dev` |
| **mnc_python** | Antenna health queries | In `development` conda env |
| **extractor_pb_75** | Primary beam model | Cluster-installed |
| **RabbitMQ** | Celery broker | `rabbitmq.calim.mcs.pvt:5672` |
| **Redis** | Celery result backend | `10.41.0.85:6379` |

---

## How It Connects

```
pipeline/subband_celery.py          # User runs this
    │
    ├── orca.resources.subband_config   # Reads NODE_SUBBAND_MAP, queue routing
    ├── orca.transform.subband_processing.find_archive_files_for_subband()
    │
    └── orca.tasks.subband_tasks.submit_subband_pipeline()
            │
            ├── chord([prepare_one_ms_task.s(...) × N])   ← Phase 1 (parallel)
            │       │
            │       ├── subband_processing.copy_ms_to_nvme()
            │       ├── subband_processing.flag_bad_antennas()
            │       │       └── subprocess → orca/utils/mnc_antennas.py
            │       ├── subband_processing.apply_calibration()
            │       └── orca.wrapper.ttcal.zest_with_ttcal()
            │
            └── process_subband_task.s(...)                ← Phase 2 (callback)
                    │
                    ├── subband_processing.concatenate_ms()
                    ├── subband_processing.fix_field_id()
                    ├── orca.wrapper.change_phase_centre.change_phase_center()
                    ├── AOFlagger (subprocess)
                    ├── WSClean pilot snapshots (subprocess)
                    ├── subband_processing.analyze_snapshot_quality()
                    ├── subband_processing.flag_bad_integrations()
                    ├── hot_baselines.run_diagnostics()      (optional)
                    ├── WSClean science imaging ×7 (subprocess)
                    ├── pb_correction.apply_pb_correction()
                    └── subband_processing.archive_results()
```
