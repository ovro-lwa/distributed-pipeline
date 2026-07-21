# Deployment

Install the repository at the same absolute path on every node, or use a shared
mount. Site paths can be set with the `GSI_CONDA_SH`, `GSI_RUNTIME_ENV`,
`GSI_DEV_ENV`, `GSI_JULIA_BIN`, `GSI_SHARED_DEPOT`, and `GSI_TTCALX_DIR`
environment variables. `setup_node.sh` also has generic `/opt` defaults.

Run the setup once on each GPU node:

```bash
ssh <node> 'bash <repo>/deploy/setup_node.sh'
```

The script precompiles TTCalX and writes `/tmp/gsi_caps_$USER.txt` with the GPU
probe result. Match failed or unavailable GPUs in `cluster.yaml` with
`disabled_gpus`.

From the control host:

```bash
export GSI_WORKER_SH=<repo>/workers/run_worker.sh
gsi dry-run --config <config-dir>
gsi run --config <config-dir>
```

To update the shared production copy from a clean commit:

```bash
gpu-subband-imaging/deploy/sync_to_server.sh calim0
```

The deployment writes `.gsi-version`; each run records that parent-repository
revision in its metadata.

The control host needs passwordless SSH to every configured worker. All nodes
must see the config snapshot, manifests, inputs, calibrations, and output tree.
