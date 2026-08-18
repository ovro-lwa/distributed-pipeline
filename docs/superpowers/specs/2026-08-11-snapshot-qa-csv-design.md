# Snapshot QA CSV Design

## Goal

For every future main-pipeline Phase 2 subband reduction, archive the data used
to create the snapshot RMS diagnostic plot as a machine-readable CSV. This is
part of normal processing and does not backfill previous runs.

## Output

Write the following file beside the existing snapshot QA PNG:

```text
QA/snapshot_qa_<subband>.csv
```

The CSV contains one row for every successfully analyzed pilot snapshot, in
snapshot-index order, with these columns:

```text
snapshot_index,scan_number,timestamp_utc,rms_jy_per_beam,peak_jy_per_beam,flagged,filename
```

- `snapshot_index` is the index assigned by the existing RMS analysis.
- `scan_number` is the corresponding sorted unique `SCAN_NUMBER` from the
  concatenated measurement set.
- `timestamp_utc` is parsed from the timestamped pilot snapshot filename and
  written in ISO 8601 UTC form.
- `rms_jy_per_beam` and `peak_jy_per_beam` are the existing computed values.
- `flagged` records whether the existing RMS thresholds selected the snapshot.
- `filename` preserves the archived pilot snapshot filename.

## Design

Add a focused CSV-writing helper to `orca/transform/subband_processing.py`.
The helper receives the existing `stats` and `bad_indices`, reads scan numbers
from the concatenated MS, and writes the CSV beneath the work directory's `QA`
directory. Phase 2 invokes it immediately after RMS analysis and before the
existing plot and flag-application steps.

The existing RMS calculation, thresholds, plot, and flag application remain
unchanged. A CSV reporting failure is logged and does not fail an otherwise
valid reduction. If a snapshot index cannot be mapped to a scan number, its
`scan_number` field is left empty and a warning is logged.

## Testing

Focused unit tests will verify:

1. Every stats row is written in snapshot-index order.
2. Snapshot indices map to the correct measurement-set scan numbers.
3. RMS, peak, timestamp, filename, and flagged status are preserved.
4. An unmappable scan produces an empty field without dropping the snapshot.
5. Phase 2 calls the writer with the same QA results used by plotting and
   flagging.
