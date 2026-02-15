# Subband Processing Pipeline — Architecture & File Reference

> This document covers the Celery-based subband processing pipeline for OVRO-LWA.
> It processes per-subband data through calibration, flagging, peeling, imaging,
> science extraction (dewarping, photometry, transient search, flux check),
> and archival — all orchestrated by Celery with NVMe-local execution on the calim
> cluster.

---

## Overview

The pipeline uses the following
**Celery chord pattern**:

```
Phase 1 (parallel, one Celery task per MS file):
    copy to NVMe → flag bad antennas → apply calibration → peel sky → peel RFI

Phase 2 (sequential, runs once all Phase 1 tasks finish):
    concatenate → fix field IDs → change phase centre → AOFlagger →
    pilot snapshots → snapshot QA → hot-baseline removal → science imaging →
    PB correction → ionospheric dewarping → target photometry →
    solar system photometry → transient search → flux scale check →
    archive to Lustre (per-run + centralized samples/detections)
```

Both phases are routed to the **same per-node queue** (e.g. `calim08`) so that
all I/O stays on the node's local NVMe.

Every step emits `[TIMER]` log lines (e.g. `[TIMER] aoflagger: 42.3s`)
for post-run performance analysis. Grep with `grep '\[TIMER\]' worker.log`.

---

## File Map

### Submission & Orchestration

| File | Purpose |
|------|---------|
| `pipeline/subband_celery.py` | **CLI entry point.** Discovers MS files, computes LST segments, submits one chord per (subband, LST-hour) to the correct Celery queue. Flags: `--targets`, `--catalog`, `--snapshot_clean`, `--remap SUBBAND=NODE`. |
| `orca/tasks/subband_tasks.py` | **Celery task definitions.** Contains `prepare_one_ms_task` (Phase 1), `process_subband_task` (Phase 2 including science phases A–D), and `submit_subband_pipeline()` which wires them into a chord. All steps have `[TIMER]` instrumentation. |
| `orca/celery.py` | **Celery app configuration.** Defines broker/backend, all queues (`default`, `cosmology`, `bandpass`, `imaging`, `calim00`–`calim10`), and task include list. |

### Processing Logic (no Celery decorators — pure functions, testable locally)

| File | Purpose |
|------|---------|
| `orca/transform/subband_processing.py` | **Core subband steps.** File discovery, NVMe copy, SPW map calculation, calibration application, bad-antenna flagging (via MNC subprocess), concatenation, field ID fix, snapshot QA, scan-based integration flagging, timestamp injection, PB correction dispatch, `find_deep_image()`, `extract_sources_to_df()` (BDSF), archival (including centralized `samples/` and `detections/` trees), subprocess helpers. |
| `orca/transform/hot_baselines.py` | **Hot-baseline & amplitude-vs-UV diagnostics.** Identifies bad antennas/baselines from a concatenated MS, produces heatmap and UV diagnostic plots, optionally flags in-place. |
| `orca/transform/pb_correction.py` | **Primary beam correction.** Applies the OVRO-LWA beam model to FITS images using `extractor_pb_75`. |

### Science Extraction Modules (Phase 2, steps 7b A–D)

| File | Purpose |
|------|---------|
| `orca/transform/ionospheric_dewarping.py` | **Ionospheric dewarping.** Parses VLSSr catalog, cross-matches extracted sources, builds 2-D pixel warp screens via `griddata`, applies `map_coordinates` correction to all FITS images. Produces diagnostic quiver/distortion plots in `Dewarp_Diagnostics/`. |
| `orca/transform/cutout.py` | **Target photometry.** Loads CSV target lists, extracts Stokes I+V cutouts (deep + 10-min difference), measures flux with ionospheric-aware search radius + confusing-source masking, optionally fits with CASA `imfit`. Outputs per-target CSVs, diagnostic PNGs, and detection flags. |
| `orca/transform/solar_system_cutout.py` | **Solar system photometry.** Computes ephemerides for Moon, Mercury–Neptune via `astropy.coordinates.get_body`, extracts I+V cutouts from deep and 10-min images, writes per-body CSVs with angular diameters and distances. |
| `orca/transform/transient_search.py` | **Transient search.** Stokes V blind search + Stokes I deep-subtracted search. Adaptive bright-source masking, local RMS maps, bi-lobed artifact rejection, catalog cross-match, cutout generation. Quality gate: >10 candidates triggers a warning flag. |
| `orca/transform/flux_check_cutout.py` | **Flux scale validation.** Fits calibrators (3C48, 3C147, 3C196, 3C286, 3C295, 3C380, 3C123) with CASA `imfit`, compares to Scaife & Heald 2012 / Perley-Butler 2017 models. Outputs `flux_check_hybrid.csv` and diagnostic ratio plot in `QA/`. |

### Configuration

| File | Purpose |
|------|---------|
| `orca/resources/subband_config.py` | **All pipeline configuration in one place.** Node↔subband mapping (`NODE_SUBBAND_MAP`), NVMe/Lustre directory layout, peeling parameters, AOFlagger strategy path, hot-baseline params (with `uv_window_size`), `SNAPSHOT_PARAMS` (dirty) and `SNAPSHOT_CLEAN_PARAMS` (niter=50 000), all 7 imaging steps, resource allocation per node, `CALIB_DATA` (SH12/PB17 flux models for 7 calibrators), `VLSSR_CATALOG` and `BEAM_MODEL_H5` paths. |
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
    ├── *.ms                     # Individual MS files (Phase 1)
    ├── <subband>_concat.ms      # Concatenated MS (Phase 2)
    ├── I/deep/                  # Stokes I deep images (+pbcorr, +dewarped)
    ├── I/10min/                 # Stokes I 10-min interval images
    ├── V/deep/                  # Stokes V deep images
    ├── V/10min/                 # Stokes V 10-min interval images
    ├── snapshots/               # Pilot snapshot images + QA
    ├── QA/                      # Hot-baseline plots, flux check CSV+plot
    ├── samples/<sample>/<target>/ # Target photometry cutouts + CSVs
    ├── detections/              # Transient cutouts, solar system detections
    ├── Dewarp_Diagnostics/      # Warp screens, quiver plots
    └── logs/                    # CASA log, subprocess logs

Lustre (shared, archived products):
/lustre/pipeline/images/<lst>/<date>/<run_label>/<subband>/
    └── (same structure as above, minus intermediate MS files)

Lustre (centralized cross-run aggregation):
/lustre/pipeline/images/samples/<sample>/<target>/<subband>/
/lustre/pipeline/images/detections/transients/{I,V}/<J-name>/<subband>/
/lustre/pipeline/images/detections/SolarSystem/<Body>/<subband>/
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
    │  Flags: --targets, --catalog, --snapshot_clean, --remap SUBBAND=NODE
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
            │       └── orca.wrapper.ttcal.zest_with_ttcal()  (sky + RFI)
            │
            └── process_subband_task.s(...)                ← Phase 2 (callback)
                    │
                    ├── subband_processing.concatenate_ms()
                    ├── subband_processing.fix_field_id()
                    ├── orca.wrapper.change_phase_centre.change_phase_center()
                    ├── AOFlagger (subprocess)
                    ├── WSClean pilot snapshots (dirty or --snapshot_clean)
                    ├── subband_processing.analyze_snapshot_quality()
                    ├── subband_processing.flag_bad_integrations()
                    ├── hot_baselines.run_diagnostics()      (optional)
                    ├── WSClean science imaging ×7 (subprocess)
                    ├── pb_correction.apply_pb_correction()
                    │
                    │  --- Science Phases (all on NVMe) ---
                    ├── A. ionospheric_dewarping (VLSSr cross-match)
                    ├── B. cutout.process_target()            (--targets)
                    ├── B2. solar_system_cutout.process_solar_system()
                    ├── C. transient_search.run_test()        (--catalog)
                    ├── D. flux_check_cutout.run_flux_check()
                    │
                    └── subband_processing.archive_results()
                            ├── per-run archive to Lustre
                            ├── centralized samples/ tree
                            └── centralized detections/ tree
```

---

## Timing Instrumentation

Every step logs `[TIMER] <tag>: <seconds>s`. Extract with:

```bash
grep '\[TIMER\]' /path/to/worker.log
```

### Phase 1 tags (per MS)

| Tag | Step |
|-----|------|
| `copy_to_nvme` | rsync MS to NVMe |
| `flag_bad_antennas` | MNC antenna flagging |
| `apply_calibration` | bandpass + XY table |
| `peel_sky` | TTCal sky model |
| `peel_rfi` | TTCal RFI model |
| `phase1_total` | Total per-MS wall time |

### Phase 2 tags (per subband)

| Tag | Step |
|-----|------|
| `concatenation` | virtualconcat / concat |
| `fix_field_id` | FIELD table fix |
| `chgcentre` | Phase centre rotation |
| `aoflagger` | Post-concat flagging |
| `pilot_snapshots_qa` | Snapshot imaging + QA |
| `hot_baselines` | Hot baseline removal |
| `imaging_<suffix>` | Each of 7 imaging steps |
| `imaging_all` | All imaging combined |
| `science_dewarping` | Ionospheric dewarping |
| `science_target_photometry` | Target cutouts |
| `science_solar_system` | Solar system photometry |
| `science_transient_search` | Transient detection |
| `science_flux_check` | Flux scale validation |
| `archive_to_lustre` | Copy to Lustre |
| `phase2_total` | Total Phase 2 wall time |

---

## CLI Examples

Run 55 MHz on calim08 (dirty snapshots):

```bash
python pipeline/subband_celery.py \
    --range 14-15 --date 2024-12-18 \
    --bp_table /lustre/gh/bandpass_tables/2024-12-18/calibration_2024-12-18_01h.B.flagged \
    --xy_table /lustre/gh/polcal/xyphase_delay_pos_3.8643ns.Xf \
    --subbands 55MHz \
    --remap 55MHz=calim08 \
    --peel_sky --peel_rfi --hot_baselines
```

Same, with CLEAN snapshots (niter=50 000):

```bash
python pipeline/subband_celery.py \
    --range 14-15 --date 2024-12-18 \
    --bp_table /lustre/gh/bandpass_tables/2024-12-18/calibration_2024-12-18_01h.B.flagged \
    --xy_table /lustre/gh/polcal/xyphase_delay_pos_3.8643ns.Xf \
    --subbands 55MHz \
    --remap 55MHz=calim08 \
    --peel_sky --peel_rfi --hot_baselines \
    --snapshot_clean
```

Full observation with science extraction:

```bash
python pipeline/subband_celery.py \
    --range 13-19 --date 2025-06-15 \
    --bp_table /lustre/gh/calibration/pipeline/bandpass/latest.bandpass \
    --xy_table /lustre/gh/calibration/pipeline/xy/latest.X \
    --peel_sky --peel_rfi --hot_baselines \
    --targets /lustre/gh/targets/exoplanets.csv /lustre/gh/targets/pulsars.csv \
    --catalog /lustre/gh/catalogs/bdsf_73MHz.csv \
    --snapshot_clean
```

Add `--dry_run` to any command to preview without submitting.
