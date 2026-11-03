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
