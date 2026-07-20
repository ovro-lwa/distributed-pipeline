# Design

The control process splits each subband into batches and assigns them to free
`(node, GPU)` slots. Workers run over SSH. A SQLite ledger tracks queued,
running, completed, failed, and skipped batches.

## Batch flow

```text
copy and extract MS tar to local NVMe
  -> flag bad antennas
  -> apply frequency-matched bandpass and XY calibration
  -> optional AOFlagger
  -> channel average
  -> GPU peel with TTCalX
  -> dirty Stokes I/V imaging with WSClean
  -> render the movie frame
  -> compress and archive FITS
  -> delete local working data
```

CPU feeders prepare snapshots in parallel. A persistent Julia process peels one
prepared snapshot at a time while one archive thread images and compresses the
previous snapshot.

## Calibration

Calibration tables are matched by `CHAN_FREQ` overlap. This avoids relying on
CASA SPW indices, which may differ between the measurement set, bandpass table,
and XY-phase table.

## Recovery

Workers skip snapshots whose Stokes I FITS product already exists. Failed
batches retry up to `max_retries`; exhausted batches are skipped so other bands
can finish. Temporary manifests and node-local calibration caches are removed
at the end of the run.

## Outputs

Each UTC hour has compressed Stokes I/V FITS files and one MP4. PNG frames can
be deleted after a successful stitch. Worker logs are stored beside the date's
products.
