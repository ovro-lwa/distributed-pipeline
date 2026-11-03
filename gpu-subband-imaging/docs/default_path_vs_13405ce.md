# Default path vs production 13405ce

**What is verified (short):** the GSI code diff below is reviewed line by line and pinned by tests; the
default peel with TTCalX `exp/ttcalx-v3-all` gives bitwise the same visibilities as 748884a on 4 production
55 MHz snapshots (GPU); the default imager gives the same images as 748884a to the GPU floor (cuFINUFFT
float atomics, as two runs of 748884a), not bit for bit.  "Equivalent" below means exactly that.

What `git diff 13405ce..HEAD` changes in code that runs with **no new switch set** (`v3:` absent or all
defaults, `peel.engine: daemon` and the other new `PeelParams` fields at their defaults, no
`GSI_SNAP_PIPELINE`).  "v3 mode" = `V3Params.active` (any of `untar_to`, `prep_inmem`, `peel_io`,
`snapshot_io`, `hour_source`, `archive` set).  Everything else in the diff (`v3/`, `workers/gsi_v3_engine.py`,
`scripts/v3_*`, tests, docs) is not reached with defaults.

Tests that pin this: `tests/test_defaults_13405ce.py`, `tests/test_gpu_image_client.py`,
`test_peel_params.py::test_default_env_and_provenance_equal_13405ce`,
`test_v3_batch.py::test_read_stats_default_is_13405ce`, `test_v3_stage_ms.py::test_v2_stage_copies_once_by_default`,
`test_v3_hourstore.py::test_defaults_delete_exactly_what_13405ce_deleted` (against the verbatim
`tests/data/hour_lst_13405ce.py`).

| File | Remaining difference on the default path | Why it is behaviour-neutral |
|---|---|---|
| `config.py` | `PipelineConfig.v3 = V3Params()`; new `PeelParams` fields (incl. `fast_kernels`); v3 checks; `v3:` null allowed | Defaults pass every check; `run_key` is unchanged (the `_v3mem` suffix needs `hour_source=memory`). `PeelParams.daemon_env` and `as_dict` leave out all new fields while they are at their defaults, so the `ZEST_*` environment and the provenance `peel` block equal 13405ce's (keys, values, order). |
| `metadata.py` | `pipeline_dict` for `run_*.json` | At defaults the `pipeline` block has 13405ce's keys: no `v3`, `peel` = `PeelParams.as_dict` (`test_metadata.py`). |
| `orchestrator.py` | `hour_in_batch` property; v3 env keys | `hour_in_batch` is false, so planning and the wideband trigger take the old branches. `GSI_PREP_LOOKAHEAD`, `XDG_CACHE_HOME` and `GSI_V3_ALLOW_PROFILE` are forwarded only in v3 mode. |
| `worklist.py` | one batch per band-hour | Only with `hour_source=memory`; otherwise `batch_size` as before. |
| `hour_image.py` | archive source in `run_band` / `plan` | Only with `hour_source=archive`. |
| `hour_lst.py` | split into `qa_step` / `movie_step` / `write_provenance`; archive source; strict checks | Same calls in the same order with the same arguments. `read_stats` default is 13405ce's "last file by name wins" (newest-row only with `newest=True`, v3 mode). The summary is parsed as before (`ValueError` only; `qa[].stem` is read only in archive mode). `rec["movie"]` is set only when the stitch did not raise, as before. Deletion takes the unchanged `elif ms and h.delete_kept_ms` branch. The hour log line text is the same. |
| `worker.py` | v3 prep, tmpfs untar, slots, archive hooks; `_image_archive` split into `_post_image` | Each hook is a no-op with defaults (`_untar_root` None, `_prep_inmem` "off", `peel_io=ms`, `archive=none`, `write_kept_ms` True). Frame, QA row, fpack and move, keep_ms and rmtree run in the old order. The QA row has the old keys (no `t`) and the rows are written once at batch end. The FITS move goes straight to the final names, I then V. `GSI_PREP_LOOKAHEAD` is read only in v3 mode, so all files are submitted to the feeders at once as before. The zest env gets nothing new. |
| `stages/stage_ms.py` | `untar_to` stream path; `attempts` | `attempts=1` (default `v3.stage_attempts`) runs the 13405ce copy + extract code: one copy, no size check, no `STAGE_RETRY`. |
| `stages/fastcal.py`, `quality.py`, `uvw.py` | `*_arrays` helpers | Pure refactors: the MS functions call the helpers with the same arrays (review: checked line by line). |
| `stages/peel.py` | start deadline, slot commands, kill/restart | `start_timeout=None`: the old blocking `_await`. `zest()` reads the same replies; the extra logged prefix `SKY_RFI` is printed only by `engine=sky_rfi`. `close()` waits up to 30 s more after a kill (only when the daemon did not exit). |
| `stages/gpu_image.py` | client rewritten with two modes | Default = 13405ce: one request at a time under the lock, the same request JSON (no `id`), no reply deadline. `GSI_SNAP_REPLY_TIMEOUT_S`, if set, kills the daemon on a missed deadline (never a late reply read as another request's). The pipelined mode with ids needs `GSI_SNAP_PIPELINE=1`, and its daemon must announce it. |
| `workers/ttcalx_snapshot_daemon.py` | `GSI_SNAP_PIPELINE=1` branch | The default loop is 13405ce's verbatim, including the per-snapshot `gridder._SVD_CACHE.clear()`. It imports only `bands, gridder, snapshot`, so it runs with the production imager (748884a). |
| `workers/zest_daemon.jl` | `peel_snapshot!` refactor; SHM / `RESET_WARM` commands; sky_rfi engine; new `ZEST_*` reads | The `engine=daemon` body is the old loop body moved into a function (same calls, same order). An MS path never starts with `SHM ` / `RESET_WARM`. New variables are unset, so their defaults apply and the new checks pass (`ZEST_FAST_KERNELS` unset: `fast_kernels!` is not called). `zest_smooth.jl` is included only if present, and `JSON` is loaded only if the project has it (both are needed only by new switches). The one difference: a `FAILED` reply is now one line (`" \| "` joins the lines). That only matters on a failure; `ZestDaemon.zest` raises on the `FAILED` line as before. Checked: loads and runs the slot I/O test with production TTCalX 748884a and with c78f0b5. |
| `workers/zest_smooth.jl` | new file | Not needed by the default path. |
| backlog port (`feature/gsi-backlog`) | `cal.badants`, `chanbin: 1`, `peel.save_solutions`, `hour_imaging.qa_exclude`, `description`, `check_protected`; `stages/peel_solutions.py`, `scripts/run_backlog.py` | `badants` unset: the live anthealth lookup as before. `chanbin` 4: the average call as before. `save_solutions` empty: no `ZEST_SAVE_SOLUTIONS`, no provenance key, the daemon writes nothing (`sol_dir_ref` stays ""). `qa_exclude` true: the same exclusions. Provenance gets the backlog keys only when one of these is set (`hour_lst.backlog_keys`). `check_protected` refuses only a model with a Scaife & Heald calibrator (none of the production models has one). An unpacked `.ms` input is copied with `cp -r` instead of `copytree` (production inputs are `.ms.tar`). |
| failure paths (backlog port) | zest "out of GPU memory": one retry after 5 s (MS path); snapshot imager GPU allocation error (`OutOfMemory`, `non-uniform points`, `cudaErrorMemoryAllocation`): the one-at-a-time client restarts the daemon before its one retry | Only after such an error (before, the OutOfMemory retry ran in the same process and the others failed the snapshot). A successful run is unchanged. |

**Deploy requirements of the default path:** none beyond 13405ce's.  The pipeline runs with the
production TTCalX (748884a) and the production imager venv (748884a).

**TTCalX (one branch, `exp/ttcalx-v3-all` = `exp/ttcalx-v3` + `exp/imager-v3-engine`):**
- **Peeler:** the 0.3.0 GPU kernels (tiled normal equations, device RFI fringes, fused corrupt-subtract) are off
  by default (`TTCalX.fast_kernels!`, gsi `peel.fast_kernels`); v3 prodlike sets `fast_kernels: true`.  With the
  defaults the production peel (`engine: daemon`, residual guard, RFI) gave **bitwise** the visibilities of
  748884a (all main-table columns of 4 kept MSes, 55 MHz, GPU); 748884a run twice was also bitwise in DATA/FLAG.
- **Imager:** the MS hour job and snapshot defaults are 748884a's settings (no chunk cache, cuFFT plans
  cleared per interval, LAPACK SVD, FFT local RMS, full-size scale FFTs, FFT residual updates).  The v3 entry
  points (`hour --archive`, `run_hour_snaps`, the gsi v3 engine) set all of them on with reserve auto and the OOM
  fallback (`ttcalx_imager.FAST_ENV`).  Always on: compact w-term factors (float32 interpolation) and the new
  Hogbom kernels: images within 748884a's run-to-run GPU floor (82 MHz hour job check), 2026-10-27 regression set PASS.  The strict beam
  fit stays opt-in (`TTCALX_STRICT_BEAM=1`) everywhere.  The snapshot `DegeneratePSFError` replaces the old
  empty-selection crash for the same inputs.

**Smoke check (2026-11-02, lwacalim00 GPU 1, 2026-10-31 LST 05h 55 MHz, first 25 snapshots, production template).**
Compared with the production night (Run_20261030_232620). Scripts and results are in
`/lustre/nkosogor/peel_exp/defaults/{scripts,results}`.

| Run | Code | CLEAN I diff rms / max (σ) | V diff rms / max (σ) | QA rows | provenance keys | log line types |
|---|---|---|---|---|---|---|
| A | this branch, production TTCalX + imager (748884a copies) | 0.0067 / 0.34 (median 0.0053 / 0.017) | 0.0083 / 0.024 | same keys, rms/peak ≤ 1.3e-7 rel | equal* | no new ones |
| B | this branch, TTCalX c78f0b5, imager a496d28 | 0.0067 / 0.34 | 0.0083 / 0.024 | same keys, ≤ 1.8e-7 rel | equal* | no new ones |
| C | v3 prodlike (config F), same code as B | 0.0054 / 0.017 | 0.0083 / 0.024 | + `t` (v3) | + `v3`, new peel keys (set) | v3 lines |

\* Except harness artefacts: `software.*.git` details are missing because the code copies are not git trees, and
`imaging.gpu_env.PYTHONPATH` points at the copies.

**Blockers check (2026-11-02, lwacalim00 GPU 1, 2026-10-31 LST 05h, production template, gsi 2b601c4, TTCalX
`exp/ttcalx-v3-all`).** Scripts and results: `/lustre/nkosogor/peel_exp/blockers/{scripts,results}`.

| Check | Result |
|---|---|
| Peel, 55 MHz, 4 snapshots: 748884a vs `exp/ttcalx-v3-all` defaults (zest daemon) | every main-table column of the 4 kept MSes **bitwise equal** (DATA, FLAG, UVW, ...) |
| Peel: 748884a run twice | DATA / FLAG bitwise equal; UVW differs by ≤ 9e-9 m (the prep's fix_uvw, not the peel) |
| Hour job, 82 MHz, 90 kept MSes (15 min), standalone, 1 s sampling | peak GPU 9292 MiB (748884a) vs 8794 MiB (v3-all defaults); wall 192 s vs 165 s |
| Hour images, 82 MHz, 748884a vs v3-all defaults | dirty / PSF ≤ 3e-7 σ rms; deep I ≤ 0.0003 σ rms / 0.05 σ max; 10-min I ≤ 0.021 σ rms / 0.86 σ max (interval t0002), equal to 748884a vs 748884a (0.023 σ / 0.85 σ there) |
| Snapshot + hour images, 55 MHz, 4 snapshots | snapshot I/V 0.0054 / 0.0082 σ rms as 748884a vs itself; hour I ≤ 0.003 σ rms (748884a vs itself 0.03 σ) |
| Smoke C2 = v3 prodlike + `fast_kernels: true`, 25 snapshots vs production P | I 0.0054 / 0.017 σ, V 0.0083 / 0.024 σ, QA ≤ 9.4e-8 rel (C: 0.0054 / 0.017, 0.0083 / 0.024); hour products C vs C2 ≤ 0.0067 σ rms; 157 s (C 144 s) |
