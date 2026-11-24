# Changelog

Earlier versions (v1-v4): the "Versions" table in [README.md](README.md).

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
- The v3 engine drops the zenith gridder's tables at a band change.

Not in v5 (next release): two batches per GPU (`lanes_per_gpu`, `v3.gpu_share`, `v3/gpu_arbiter.py`).

Tests: `tests/test_v3_dispatch_order.py`, `tests/test_deploy_v5_template.py`, the `peel_fast` cases in
`tests/test_peel_params.py` / `tests/test_v3_daemon_julia.py`, the engine `uvw_stored` cases in
`tests/test_v3_engine.py`. `test_hour_op_sets_the_cache_reserve_and_reports_the_fallback` (failing on main since
b93d4e5) fixed on the test side.

TTCalX: deploy `release/v0.5.0` @7a4b931 (see `deploy/README-deploy.md`, "Switching to v5").
