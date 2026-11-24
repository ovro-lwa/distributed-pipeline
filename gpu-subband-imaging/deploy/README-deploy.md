# Deployment

A deployment is one checkout of this repository plus the TTCalX tree and the imager venv its
config points at. Every node must see it at the same absolute path (shared mount), together
with the config snapshot, manifests, inputs, calibration tables and the output tree. The
control host needs passwordless SSH to every worker in `cluster.yaml`.

| piece | where it is set | v2 production |
|---|---|---|
| gsi code | `GSI_HOME` (in the env script) | `<deploy>/distributed-pipeline/gpu-subband-imaging` at tag `gsi-v2` |
| worker launcher | `GSI_WORKER_SH` (default: `workers/run_worker.sh` beside the config dir) | `$GSI_HOME/workers/run_worker.sh` |
| pipeline conda env | `GSI_RUNTIME_ENV` (`run_worker.sh`, `setup_node.sh`), `GSI_CONDA_SH` | python-casacore, CASA, PyYAML |
| TTCalX (Julia) | `env.ttcalx`, `env.julia_bin`, `env.shared_depot` in `pipeline.yaml` (`GSI_TTCALX_DIR`, `GSI_JULIA_BIN`, `GSI_SHARED_DEPOT` override) | TTCalX `gsi-v2` (748884a) |
| imager venv | `hour_imaging.ttcalx_image` (`<venv>/bin/ttcalx-image`); its `python` runs the snapshot imager unless `imaging.gpu_python` is set | `ttcalx_imager` of TTCalX `gsi-v2` |
| fpack | `env.dev_env` (`<dev_env>/bin/fpack`) | |

The env script (`/opt/devel/pipeline/gsi-v2/gsi_env.sh` in production) activates the conda env and
exports `GSI_HOME`, `PYTHONPATH=$GSI_HOME/src`, `GSI_WORKER_SH`, the worker pinning
(`GSI_TASKSET_CPUS`, `GSI_NICE`, `GSI_IONICE`), `NUMPY_MADVISE_HUGEPAGE=0` and `CASARCFILES`
(casacore measures data for `fix_uvw`). Source it before `gsi` or `run_nightly_lst.py` (README).

The imager venv: `pip install -e '<TTCalX tree>/python/ttcalx_imager[gpu]'`
(see the [TTCalX install notes](https://nkosogor.github.io/TTCalX/)).

## Node setup

Run once per GPU node (precompiles TTCalX into the shared depot, probes each GPU):

```bash
ssh <node> 'bash <GSI_HOME>/deploy/setup_node.sh'
```

It writes `/tmp/gsi_caps_$USER.txt`; put failed GPUs in `cluster.yaml` `disabled_gpus`.

## Updating a copy

`deploy/sync_to_server.sh <host> <dest>` rsyncs the committed subproject (not `config*/`) and
writes `<dest>/.gsi-version`; each run records that revision in its metadata. A git checkout at
a tag works as well, but then remove any old `.gsi-version`: the marker takes precedence over git
in the recorded revision.

## Switching to v3

1. Deploy TTCalX `v0.3.0` as a separate tree next to the `gsi-v2` one, and precompile it
   (`GSI_TTCALX_DIR=<new TTCalX> setup_node.sh` on each node).
2. Build a new imager venv from that tree's `python/ttcalx_imager`.
3. Check out gsi `main` as a separate copy with its own env script (`GSI_HOME`, `GSI_WORKER_SH`).
4. Copy the production config and point it at the new pieces: `env.ttcalx`,
   `hour_imaging.ttcalx_image`; add the `peel:` and `v3:` lines from the README ("v3: in-memory
   band-hour"). With `hour_source: memory`, `batch_timeout_seconds` must be at least
   `hour_imaging.timeout_seconds` + 1800 + `v3.budget_margin_s` (`gsi dry-run` checks it).
5. Shadow run: the new copy on one night into a separate `output_root` (and `state_dir`), compared
   with the v2 products, before the nightly job is switched. Lustre writes outside
   `/lustre/pipeline/snapshots_gpu_pipeline` need `GSI_ALLOWED_LUSTRE_WRITE_ROOT`.
6. Switch the nightly job to the new env script and config.

Rollback: point the nightly job back at the `gsi-v2` deploy (env script, config, TTCalX tree and
venv are unchanged).

## Switching to v5

As v4: a new `/opt/devel/pipeline/gsi-v5` (`$G` below) next to `gsi-v4`, which stays the rollback. The nightly is
`deploy/nightly/daily_guarded.sh` + `night_guarded.sh` (gsi-v4's `daily_guarded_v4.sh` / `night_guarded_v4.sh`
with the deploy dir as `G`; `bp_ramp.py` and `etcd_ok.sh` beside them). They run on calim02 as `pipeline`, source
`$G/gsi_env.sh`, read the template `$G/config-lst-v5-template` and write `$G/config-lst-v5-auto`, the logs
(`night_guard_<date>.log`, `nightly_<date>.log`), `run_label_<date>.txt` and `last_good_bp.txt` in `$G`.

Run every git step in the deploy trees as the `pipeline` user (`sudo -iu pipeline`): the trees belong to it, and
git run by another user stops with "detected dubious ownership" (do not add a global `safe.directory`).

1. TTCalX. The tag `v0.5.0` must be pushed to https://github.com/nkosogor/TTCalX.git first (the release branch
   `release/v0.5.0` is not on GitHub); the deploy trees have no git remote, so fetch the tag by URL:

   ```bash
   G=/opt/devel/pipeline/gsi-v5
   git clone https://github.com/nkosogor/TTCalX.git $G/TTCalX      # or copy the gsi-v4 tree, then:
   git -C $G/TTCalX fetch https://github.com/nkosogor/TTCalX.git tag v0.5.0
   git -C $G/TTCalX checkout --detach v0.5.0
   git -C $G/TTCalX log --oneline -1          # the reviewed v0.5.0 commit (README "Versions" row)
   ```

   Copy `Manifest.toml` from the v4 tree (no new Julia dependencies) and precompile
   (`GSI_TTCALX_DIR=$G/TTCalX setup_node.sh` on each node).
2. Imager venv, built as v4's (`/opt/devel/pipeline/gsi-v4/venv`: a plain venv of `py310_orca`, the imager
   installed editable from its TTCalX tree, the other packages at v4's versions):

   ```bash
   /opt/devel/pipeline/envs/py310_orca/bin/python -m venv $G/venv
   /opt/devel/pipeline/gsi-v4/venv/bin/pip freeze --exclude-editable > $G/venv_pins.txt
   $G/venv/bin/pip install -c $G/venv_pins.txt -e "$G/TTCalX/python/ttcalx_imager[gpu]"
   $G/venv/bin/python -c 'import ttcalx_imager as t; print(t.__version__, t.__file__)'   # 0.5.0, under $G/TTCalX
   ```
3. gsi: `release/gsi-v5` (or its tag) checked out as `$G/distributed-pipeline`, and `$G/gsi_env.sh` =
   `deploy/nightly/gsi_env.sh` (gsi-v4's with `gsi-v4` -> `gsi-v5`).
4. Template: `cp -r $G/distributed-pipeline/gpu-subband-imaging/deploy/config-lst-v5-template
   $G/config-lst-v5-template` (the v4 nightly template + the v5 keys; `git log -p` of the directory shows the
   change). Its paths assume `$G` and `/fast/pipeline/gsi_v5`.
5. Shadow / comparison run on hours v4 already made, never overlapping a nightly (a test's `cleanup-nodes`
   deletes the nightly's node caches).
6. Switch the nightly (calim02, as `pipeline`), between two nights: never run the v4 and v5 loops at once. Both
   use the same `state_dir` and `output_root`, and each keeps its own run labels in its `$G`, so two loops would
   run the same date twice into one tree under two run labels.

   ```bash
   N=/opt/devel/pipeline/gsi-v5/distributed-pipeline/gpu-subband-imaging/deploy/nightly
   # a. wait until v4's current night has finished: its night_guard_<date>.log ends with "night <date> finished",
   #    and nothing of it runs any more
   pgrep -af 'run_nightly_lst|gpu_subband_imaging.cli'          # empty (a test run's --config is not the nightly)
   # b. stop the v4 loop (it is then only waiting for the next date's calibration)
   screen -ls                                                   # daily_v4
   screen -S daily_v4 -X quit
   pgrep -af daily_guarded_v4                                   # empty
   # c. the reference bandpass table carries over
   cp /opt/devel/pipeline/gsi-v4/last_good_bp.txt /opt/devel/pipeline/gsi-v5/last_good_bp.txt
   # d. the first v5 date: the next date v4 has not started (no "starting night" in gsi-v4/night_guard_<date>.log)
   D=2026-MM-DD
   CHECK_ONLY=1 G=/opt/devel/pipeline/gsi-v5 bash $N/night_guarded.sh $D     # optional: calibration check only
   screen -dmS daily_v5 bash -c "G=/opt/devel/pipeline/gsi-v5 bash $N/daily_guarded.sh $D 2>&1 | tee -a /opt/devel/pipeline/gsi-v5/daily_v5.log; exec bash"
   ```

   The loop logs "starting night <date> on gsi-v5" to `$G/night_guard_<date>.log`.

Rollback to v4 (its deploy, template, TTCalX tree and venv are unchanged): the same steps the other way round.
Wait for (or stop) the running v5 night, `screen -S daily_v5 -X quit`, check `pgrep -af daily_guarded` is empty,
`cp /opt/devel/pipeline/gsi-v5/last_good_bp.txt /opt/devel/pipeline/gsi-v4/` (the tables v5 accepted), and start
`screen -dmS daily_v4 bash -c "bash /opt/devel/pipeline/gsi-v4/daily_guarded_v4.sh <next date> 2>&1 | tee -a
/opt/devel/pipeline/gsi-v4/daily_v4.log; exec bash"`. A night v5 stopped in the middle is redone on v4 with
`night_guarded_v4.sh <date>`: v4 writes it under its own run label (v4 has no `run_label_<date>.txt` for it); the
partial v5 run directory stays under the v5 label.
