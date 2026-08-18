"""Lightweight visibility checks before GPU peeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VisibilityStats:
    sampled_values: int
    usable_values: int
    nonzero_values: int
    nonzero_fraction: float
    median_amplitude: float
    max_amplitude: float


def visibility_stats(ms: Path, column: str, sample_rows: int,
                     min_amplitude: float) -> VisibilityStats:
    """Sample the start, middle, and end of an MS visibility column."""
    from casacore.tables import table

    with table(str(ms), ack=False) as measurement_set:
        nrows = len(measurement_set)
        if nrows == 0:
            return VisibilityStats(0, 0, 0, 0.0, 0.0, 0.0)

        rows_per_chunk = max(1, sample_rows // 3)
        starts = sorted({
            0,
            max(0, (nrows - rows_per_chunk) // 2),
            max(0, nrows - rows_per_chunk),
        })
        amplitudes = []
        sampled_values = 0
        for start in starts:
            count = min(rows_per_chunk, nrows - start)
            data = measurement_set.getcol(column, startrow=start, nrow=count)
            flags = measurement_set.getcol("FLAG", startrow=start, nrow=count)
            sampled_values += data.size
            usable = (~flags) & np.isfinite(data)
            amplitudes.append(np.abs(data[usable]))

    values = np.concatenate(amplitudes) if amplitudes else np.array([], dtype=float)
    usable_values = int(values.size)
    if not usable_values:
        return VisibilityStats(sampled_values, 0, 0, 0.0, 0.0, 0.0)
    nonzero_values = int(np.count_nonzero(values > min_amplitude))
    return VisibilityStats(
        sampled_values=sampled_values,
        usable_values=usable_values,
        nonzero_values=nonzero_values,
        nonzero_fraction=nonzero_values / usable_values,
        median_amplitude=float(np.median(values)),
        max_amplitude=float(np.max(values)),
    )
