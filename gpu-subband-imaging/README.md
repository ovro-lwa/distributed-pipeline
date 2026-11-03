# GPU Subband Imaging

Distributed OVRO-LWA snapshot imaging across a small GPU cluster. One control
process schedules subband batches over SSH and records progress in SQLite, so
interrupted runs resume. Per snapshot: stage, flag + calibrate + average, GPU
peeling (TTCalX), GPU CLEAN imaging (TTCalX `ttcalx_imager`), compressed FITS;
per band and LST hour: 1-h and 10-min images, QA and a movie.

## Versions

| | tag | what it is |
|---|---|---|
| **v1** | `gsi-v1` (= `gsi-production-2026-10-25`) | First GPU pipeline (until 2026-10-25): GPU peeling, WSClean dirty I+V snapshots by date, hourly movies. |
| **v2** | `gsi-v2` (13405ce; TTCalX `gsi-v2` = 748884a) | Production since 2026-10-26: LST-hour layout as the main pipeline, CLEAN snapshots on the GPU imager, GPU 1-h / 10-min images, QA guards, nightly LST runs. Each snapshot goes through disk several times; the hour job re-reads all kept MSes. |
| **v3** | opt-in (`v3:` config; TTCalX `v0.3.0`) | One band-hour in one process, data kept in RAM: numpy prep instead of CASA, peel and imaging through shared memory, hour images made from memory. Same images as v2, ~2.5x faster per band-hour. |

With no `v3:` block and no new `peel:` options the code runs as v2
(`docs/default_path_vs_13405ce.md`). Roll back: `git checkout gsi-v2` here and in TTCalX.

## Requirements

Python 3.8+ with `python-casacore`, CASA tasks, NumPy, Astropy and PyYAML; Julia, CUDA and
TTCalX; CFITSIO `fpack` and FFmpeg (WSClean and AOFlagger only for the v1 paths); passwordless
SSH from the control host to the workers; shared input, calibration, state and output paths.

```bash
python -m pip install -e .
export GSI_WORKER_SH="$PWD/workers/run_worker.sh"
```

## Configure and run

```bash
cp -R examples/config config-local     # cluster.yaml (nodes, GPUs, scratch), subbands.yaml (image
                                       # geometry), pipeline.yaml (date, inputs, calibration, peel, imaging)
gsi dry-run --config "$PWD/config-local"
gsi run     --config "$PWD/config-local"     # also: gsi status / resume / hour-image / cleanup-nodes
```

Lustre writes are only allowed under `/lustre/pipeline/snapshots_gpu_pipeline` unless
`GSI_ALLOWED_LUSTRE_WRITE_ROOT` (`:`-separated) says otherwise. `GSI_TASKSET_CPUS`, `GSI_NICE`
and `GSI_IONICE` (e.g. `0-23`, `10`, `-c2 -n7`) pin and deprioritise the workers. Every run
writes `<output_root>/_metadata/<date>/run_*.json` (code revisions, frozen configuration,
calibration, parameters, ledger summary) and a `provenance.json` per band-hour.

## Options (`pipeline.yaml`)

Defaults keep the original v1 behaviour; the v2 production values are listed at the end of
`examples/config/pipeline.yaml`.

| option | default | what it does |
|---|---|---|
| `peel.solver`, `model`, `uvw` | `damped`, `kernels`, `ms` | `rk4`, `exact`, `geometric`: CPU TTCal's solver, sky model and precessed baselines |
| `peel.maxiter`, `tolerance`, `minuvw`, `peeliter` | 30, 1e-2, 10, 3 | CPU TTCal: 5, 1e-4, 5, 3 |
| `peel.min_source_elevation_deg`, `require_convergence`, `max_gain_amplitude` | 15, true, 100 | sources below the limit stay in; bad solutions are not subtracted |
| `peel.flag_model`, `resubtract`, `residual_guard`, `flag_zero` | false | flagged data out of both sides of the solve; keep the last good solution; keep an update only if it lowers the residual; flag all-zero samples |
| `peel.rfi`, `cal.rfi_sources`, `peel.rfi_mode` | false, –, `after` | zest the near-field RFI emitters after the sky (`joint`: with it). The main pipeline does this on every band; `peel_overrides` sets options per band |
| `peel.reg_max_offset_deg` (+ `reg_taper_deg`, `reg_lambda`) | 0 (off) | Tukey phase prior on the gains |
| `peel.device` | `gpu` | `cpu`: TTCalX on CPU threads |
| `fix_uvw` | `centroid` | recompute J2000 UVW (raw MSes store (u, v) rotated by ~0.34°, which rotates the images); `time` = as chgcentre; `off` = stored UVW |
| `prep` | `casa` | `numpy`: flags + bandpass + XY in one casacore pass (`stages/fastcal.py`), ~2x faster |
| `imaging.imager`, `imaging.mode` | `wsclean`, `dirty` | `ttcalx`, `clean`: GPU CLEAN I (orca `SNAPSHOT_CLEAN_I_PARAMS`) + dirty V |
| `keep_ms_root`, `keep_ms_local` | off | keep the peeled MSes (Lustre, or node-local with fallback) for the hour images |
| `hour_imaging.*` | – | `ttcalx-image hour` products (orca names), QA, `delete_kept_ms` |
| `stream_hour_images`, `wideband_per_hour` | false | start each band-hour's hour job as soon as its snapshots are done; wideband summary per LST hour |

The zest daemon logs every solve decision and `ZEST_TIMING read=… peel=… write=…` per snapshot.
`uvw.fix_uvw(ms, dry_run=True)` reports how far an MS's UVW are from J2000.

## LST-hour layout and nightly runs (`layout: lst`)

Snapshots are grouped by LST hour (the main pipeline's Sun cut) into the main pipeline's tree:

```
<output_root>/<LST>h/<date>/<run_label>/<band>MHz/
    snapshots/ (dirty V)  snapshots_clean/ (CLEAN I)  I/deep  I/10min  V/deep  V/10min
    QA/  Movies/  provenance.json
```

After its snapshots, each band-hour gets one hour job (`hour_lst.py`): orca's pilot-V QA
(median + 3 MAD / 0.5 x median), the 1-h and 10-min products without the rejected snapshots,
movie, provenance. With `snapshot_phase_centre: lst` snapshots are imaged at the LST-hour centre.
Worker logs: `<output_root>/_logs/<date>/<run_label>/<band>MHz/`.

```bash
source /opt/devel/pipeline/gsi-v2/gsi_env.sh
python $GSI_HOME/scripts/run_nightly_lst.py --date 2026-10-26 \
    --template /opt/devel/pipeline/gsi-v2/config-lst-template \
    --output-config /opt/devel/pipeline/gsi-v2/config-lst-auto
python $GSI_HOME/scripts/night_report.py /lustre/pipeline/snapshots_gpu_pipeline 2026-10-26 <run_label>
```

`run_nightly_lst.py` waits for the night's calibration, checks the new bandpass against the
previous night's (falling back to the backup), runs as the data arrive and writes the night
report. `run_daily_cal_lst.py` runs the daily calibration; `run_daily_allsubbands.py` and
`run_cosmology_backfill.py` are the older date-based launchers. All share one cluster lock.
Deployment: `deploy/README-deploy.md`; scheduler design: `docs/DESIGN.md`.

## v3: in-memory band-hour (opt-in)

Needs TTCalX `v0.3.0` (`main`). The validated production-like setup (images equal to v2 at the
GPU floor on all subbands of 2026-10-31 LST 05h):

```yaml
peel: {engine: sky_rfi, precision: mixed, peeliter: 3, workspace: true, fast_kernels: true}
v3:   {untar_to: /dev/shm/gsi3/untar, prep_inmem: on, peel_io: shm, snapshot_io: shm,
       hour_source: memory, write_kept_ms: false, archive: none, store: auto, checkpoint: local}
```

New options (all off / v2 behaviour by default):

| option | default | what it does |
|---|---|---|
| `v3.*` | off | in-memory band-hour; full list and checks in `docs/v3_design.md` §6 |
| `v3.max_given_up_fraction` | 0.1 | hour fails if more inputs than this fail permanently |
| `v3.stale_checkpoint_hours` | 48 | delete local checkpoints older than this at batch start |
| `peel.engine` | `daemon` | `sky_rfi`: TTCalX sky + RFI engine |
| `peel.precision` | `double` | `mixed`: ~4x faster peel, same images |
| `peel.workspace` | false | reuse GPU peel buffers across snapshots (no GC stalls) |
| `peel.fast_kernels` | false | newer TTCalX GPU kernels (same maths, other rounding) |
| `peel.smooth_flux`, `peel.restore`, ... | off | sponge smoothing and Jupiter restore, not production yet (`docs/peel_config.md`) |
| `GSI_SNAP_PIPELINE=1` | off | pipelined snapshot-imager daemon (v2 path) |
| `GSI_SNAP_REPLY_TIMEOUT_S` | none (900 s when pipelined) | kill the snapshot-imager daemon after this many seconds without a reply |

The imager's own switches (`TTCALX_CHUNK_CACHE`, `TTCALX_CACHE_RESERVE`, ...) default to
748884a behaviour; v3 turns on `ttcalx_imager.FAST_ENV` (see the TTCalX imager docs).
