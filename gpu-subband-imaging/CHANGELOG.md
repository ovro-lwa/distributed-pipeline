# Changelog

Earlier versions (v1-v4): the "Versions" table in [README.md](README.md).

## v5.1 (`release/gsi-v5.1`, 2026-11-25; on `gsi-v5` e3bf59a; TTCalX v0.5.0 unchanged)

Per-batch overheads: Julia JIT, prep CPU, feed starvation of the first peel pass, the idle GPU during the hour
stage. Every option is off by default: a v5 config runs as v5. The v5.1 nightly template
([deploy/config-lst-v5.1-template](deploy/config-lst-v5.1-template), the v5 template plus exactly these keys,
checked by `tests/test_deploy_v5_1_template.py`) turns them on.

| option | default | v5.1 template | what it does |
|---|---|---|---|
| `v3.prep_block_rows` | 0 (whole arrays) | 2048 | the prep processes calibrate and average 2048 rows at a time (cache-sized passes) instead of the whole snapshot. Every value comes from the same per-row operations, so the slot is bitwise the same (`test_blocked_cal_average_is_bitwise_the_whole_array_prep`, also with `prep_gain_cache`). Prep 3.2 -> 2.0 s per snapshot. |
| `v3.zest_persistent` (+ `zest_persistent_idle_s` 1800) | false | true | one zest daemon per GPU that outlives the batch (`v3/zest_server.py`: a small server per GPU and node, its own session, UNIX socket in `/tmp/gsi_zest_<uid>/`). A batch attaches at its start (per-GPU flock) and detaches at the end of its snapshot loop (aoflag=hour: after pass 1) with `RELEASE`, which frees the daemon's workspace, slot buffers and warm-start gains and keeps only the CUDA context (~0.27 GB of GPU memory, also through the hour stage; destroying the context crashes the next TTCalX v0.5.0 peel). Only the first batch on a GPU pays the JIT: the first peel of a batch 45 s -> 0.4 s. A server is reused only with the same key (daemon command, ZEST_* / JULIA_* / CUDA_* environment, size + mtime of the daemon script, sky and RFI models, TTCalX sources, `zest_server.py` and `stages/peel.py`), else replaced. A second batch on the same GPU, or a server that cannot start, gets a private daemon (v5 behaviour); a server lost mid-batch fails the snapshot in flight and the rest of the batch peels with a private daemon. Idle servers exit after `zest_persistent_idle_s`; `gsi cleanup-nodes` stops them. Needs `peel_io: shm` and `cluster.yaml zest_recycle_files: 0` (checked at load). |
| `v3.aoflag_async_io` (+ `aoflag_load_threads` 2) | false | true | `aoflag: hour` only (checked at load): pass 1 writes the peeled slot copies in a background thread (the next peel does not wait; at most one more copy queued, each holds its ring slot until written; a failed copy aborts the batch as before) and pass 2 reads them back in 2 threads ahead of the engine, handed on in order. The same bytes. |
| `v3.prefetch` | false | true | streaming LST runs with `hour_source: memory` (needs it; a no-op warning otherwise): a batch writes `<state_dir>/prefetch/<run_key>/<band>MHz_b<batch>.hour` when its hour stage starts, and the orchestrator gives its slot the next batch at once (never a third). Every batch takes a per-GPU flock (`/tmp/gsi_v3gpu_<uid>_gpu<n>.lock`); the prefetched one does its setup (store plan, calibration tables, bad antennas) and stages + preps its first `lookahead` + 1 snapshots into its ring (the snapshot loop's first top-up: same calls, same slots) while the GPU is busy, and starts the zest daemon, the engine and the peels when the previous batch has exited. No early staging for a retry that already has FITS (its work list needs the store) or when MemAvailable is below `ram_reserve_gb` + the prep processes + ring + raw MSes (`V3_PREFETCH_TIGHT`). A prefetched batch is dispatched with 2 x `batch_timeout_seconds` and restarts its budget when it gets the GPU, so the previous batch's hour stage is not taken from its own budget. |
| `v3.lookahead` | 8 | 16 | snapshots staged + prepped ahead of the peel |
| `cluster.yaml max_feeders` | 8 (v5 template) | 16 | feeder threads (Lustre read + untar) per batch; with lookahead 16 pass 1 is no longer feed-bound (~200 -> 126-157 s per band-hour). Watch the Lustre read load. |
| `fpack_deterministic` | false | false | `fpack -qt 4 -v` instead of `fpack -v` for the snapshot FITS: the same quantization (level 4, SUBTRACTIVE_DITHER_1) with the dither seed from the first tile's checksum instead of the clock, so the same image gives the same compressed pixels (two v5 runs of the same data differ in every .fz by the dither). Off in the template: turning it on changes the archived snapshot pixels at the dither level. |

Measured (2026-10-24 test input, LST 04h + 05h, 23-82 MHz, 5 nodes x 2 GPUs, the template settings; results in
`/lustre/pipeline/snapshots_gpu_pipeline/_test_v5/results`):

- Wall: 1050 s and 1141 s with everything on (two runs), against 1265 / 1276 s for v5 (v5g, v5f) and
  1130-1272 s with everything but `prefetch`; GPU slot time 9790 vs 10500 s and peel GPU time -28 % without
  `prefetch`, pass 1 -23 %. `prefetch` alone: -31 / -32 s per band-hour after the first one on a GPU.
- Identity: peel lines identical to v5 (4247 / 4247 `PEEL_*` lines), flags and QA decisions identical in every
  band-hour (24 / 24), images within the rerun spread (deep median 0.0012 sigma between the two prefetch runs, as
  v5f vs v5; snapshot I / V within the floor). Two v5 runs of the same data are not bitwise equal either:
  fpack's clock dither seed (see `fpack_deterministic`) and a 1e-8 m UVW jitter (casacore measures cache).
- No `zest_persistent` history effect: a reused daemon gives the same peel lines as a fresh one.

Review fixes before the release (`git log e3bf59a..release/gsi-v5.1`): a stale hour marker of an earlier attempt
is removed at dispatch (it could prefetch onto a slot whose batch had only just started); a prefetched batch's wait
for the GPU no longer counts against its budget and timeout; the zest server key covers the files ZEST_* settings
name (an RFI model rewritten in place is not served by a daemon that loaded the old one).

Deploy: `deploy/README-deploy.md`, "Switching to v5.1" (a new `/opt/devel/pipeline/gsi-v5.1` next to gsi-v5,
sharing gsi-v5's TTCalX tree, venv and CuPy cache). Tests: `tests/test_v3_overheads.py`,
`tests/test_v3_prefetch.py`, `tests/test_fpack_deterministic.py`, `tests/test_deploy_v5_1_template.py`.

## v5 (`release/gsi-v5`, 2026-11-24; TTCalX v0.5.0)

Every new switch is off by default: a v4 config runs as v4. The v5 nightly template
([deploy/config-lst-v5-template](deploy/config-lst-v5-template), a diff against the deployed gsi-v4 template)
turns them on.

| switch | default | v5 template | what it does |
|---|---|---|---|
| `peel.peel_fast` | false | true | TTCalX `zest_sky_rfi!(...; peel_fast=true)`: the zest daemon keeps the visibilities in ComplexF32 and runs the fast peel, ~2.6x faster per snapshot (0.38x the peel time); another rounding of the v4 peel (float32 level at 55 / 82 MHz, ~1e-3 relative at 23 MHz where an RFI guard decision occasionally flips). Needs the v4 peel keys (`engine: sky_rfi`, `precision: mixed`, `workspace`, `fast_kernels`, `solver: rk4`, `maxiter` <= 63, `rfi_mode: after`, no warm start / smoothing / restore / phase prior), checked at config load. |
| `imaging.gpu_env: {TTCALX_V5: "1"}` | unset | set | the v3 engine's `use_fast_settings()` adds the imager's v5 set (`ttcalx_imager.v5_env()`): zenith-frame snapshot gridding (`TTCALX_SNAP_ZEN`, ~2.2-2.5x faster snapshots at 55-82 MHz) and the hour-stage switches (`HOUR_V5_ENV`, ~1.14-1.40x faster hour stage). The engine hands the imager the stored UVW (`small["uvw_stored"]`) that the zenith path needs; pinned H2D (`TTCALX_PINNED_H2D`) stays opt-in. |
| `v3.aoflag_procs` | 10 | 16 | more AOFlagger processes per band-hour (with the skip below: 1.65-2.1x faster flagging). |
| `v3.retry_on_checkpoint_node` | false | true | with `hour_source: memory`, `checkpoint: local`, `archive: none`: a band-hour's first retry waits for the node holding its local checkpoint; not for a node that does not answer (`node_free_cores` None) or left the cluster, and at most `v3.retry_pin_max_wait_s` from the first slot elsewhere it was kept from. Later retries go anywhere. Set without those three settings it logs a warning and does nothing. |
| `v3.retry_pin_max_wait_s` | 1800 | 1800 | the pin timeout above (a node that keeps failing the free-core check no longer holds the band-hour, nor the night with `nightly_wait_all`). |
| `max_retries` | 1 | 2 | (v4 template: 1) the first retry is the pinned one, the second goes anywhere, so a node-caused failure does not lose the band-hour. A band-hour can now take up to 3 x `batch_timeout_seconds`; with `v3.max_input_failures` 0 (= `max_retries` + 1) an input is given up after failing in 3 attempts (was 2), i.e. still in the last attempt. |
| `dispatch_band_order` | `[]` | slowest first | band order inside each LST hour in streaming dispatch; a list of distinct configured bands (checked at load); not in the run metadata while empty. |

Always on (no switch; same results):

- `v3.aoflag: hour` skips the strategy run on fully flagged baselines (flags bitwise identical, 1.15x).
- `_plan_shm` counts min(lookahead + 1, max_feeders) raw MSes in tmpfs (was lookahead), so tmpfs untar is not
  turned off when it fits. No effect at the production settings (lookahead 8, max_feeders 8: min(9, 8) = 8, as
  before); it only matters after the memory guard has lowered the lookahead.
- The v3 engine drops the zenith gridder's tables at a band change, and with the CuPy pools before its
  out-of-memory retry (`GpuBackend.free`).
- Router audit (v5 imaging): each snapshot's zenith-router decision (`K_h`, `K_z`, `routed`, `used`, `zen_far`,
  `zen_bad_R`, `zen_oom`; TTCalX `info["zen_router"]`) is in the engine's image reply and in the pilot QA row
  (`QA/.snapshot_stats/*.jsonl`, key `zen`); one INFO line per batch counts them ("zenith router: N snapshot(s),
  Z zenith path, P production path (F fallback: ...)"). Rows and replies of snapshots without the router (v5
  imaging off, an older imager) are unchanged.
- `deploy/warm_cupy_cache.py` (`setup_node.sh` with `GSI_WARM_CUPY=1`): pre-warms a node's CuPy kernel cache on
  synthetic data, so the first snapshot of a batch does not compile ~220 kernels (~36 s in the v5 e2e).

Not in v5 (next release): two batches per GPU (`lanes_per_gpu`, `v3.gpu_share`, `v3/gpu_arbiter.py`).

Tests: `tests/test_v3_dispatch_order.py`, `tests/test_deploy_v5_template.py`, the `peel_fast` cases in
`tests/test_peel_params.py` / `tests/test_v3_daemon_julia.py`, the engine `uvw_stored` / `zen_summary` / `free` cases in
`tests/test_v3_engine.py`, the router QA rows in `tests/test_v3_batch.py`. `test_hour_op_sets_the_cache_reserve_and_reports_the_fallback` (failing on main since
b93d4e5) fixed on the test side.

TTCalX: deploy the tag `v0.5.0` (branch `release/v0.5.0`, reviewed at 83f56b7, final f363db6 = 83f56b7 + the zenith
router's SVD prefetch fix and `zen_router` info; to be pushed and tagged on GitHub
first), see `deploy/README-deploy.md`, "Switching to v5", which also has the nightly switch
(`deploy/nightly/`: the gsi-v4 `daily_guarded` / `night_guarded` drivers with the deploy dir as a variable) and
the rollback.
