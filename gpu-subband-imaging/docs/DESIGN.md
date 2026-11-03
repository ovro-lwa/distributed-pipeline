# Design

The control process (`gsi run`) splits each subband into batches and assigns them to free
`(node, GPU)` slots. Workers run over SSH (`workers/run_worker.sh`). A SQLite ledger in
`cluster.state_dir` tracks queued, running, done, failed and skipped batches, so `gsi resume`
continues an interrupted run.

## Batch flow (v1/v2: one snapshot MS at a time on disk)

```text
copy and extract the MS tar to node-local scratch (work_root)
  -> flag bad antennas, optional AOFlagger, bandpass + XY, channel average   prep: casa | numpy
  -> fix_uvw (J2000 UVW)
  -> GPU peel (TTCalX zest daemon, workers/zest_daemon.jl)
  -> optional chgcentre copy to the LST-hour centre                  snapshot_phase_centre: lst
  -> snapshot images: WSClean dirty I+V, or GPU CLEAN I + dirty V    imaging.imager: wsclean | ttcalx
  -> movie frame, pilot V QA row, fpack, move to the output tree
  -> keep the peeled MS for the hour job (keep_ms_root) or delete it
```

CPU feeders prepare snapshots in parallel. One persistent Julia process peels one snapshot at a
time while `archive_workers` threads image and archive the previous ones. With `layout: lst`
each band-hour then gets one hour job (`hour_lst.py`: QA, 1-h and 10-min images from the kept
MSes, movie, provenance); `stream_hour_images` starts it as soon as that band-hour's batches are done.

v3 (`snapshot_io: shm`, `hour_source: memory`) runs the same steps for a whole band-hour in one
batch with the data in RAM; see [v3_design.md](v3_design.md).

## Calibration

Calibration tables are matched by `CHAN_FREQ` overlap. This avoids relying on CASA SPW indices,
which may differ between the measurement set, bandpass table, and XY-phase table.

## Recovery

Workers skip snapshots whose FITS products already exist. Failed batches retry up to
`max_retries`; exhausted batches are skipped so other bands can finish. Temporary manifests are
removed at the end of the run, and the node calibration caches too unless
`calcache_cleanup: none` (then `gsi cleanup-nodes` removes them).

## Outputs

- `layout: lst` (v2 production): `<output_root>/<LST>h/<date>/<run_label>/<band>MHz/` (see the README).
- `layout: utc` (v1): `<output_root>/<band>MHz/<date>/<UTC hour>/`, Stokes I/V FITS and one MP4 per hour.

`_metadata/<date>/run.json` (`run_<run_key>.json` with `layout: lst`) records the code revision,
configuration, calibration paths, parameters and ledger summary.
