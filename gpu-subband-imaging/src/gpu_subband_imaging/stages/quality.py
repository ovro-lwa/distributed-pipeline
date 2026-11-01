"""Lightweight visibility checks before GPU peeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class VisibilityStats:
    sampled_values: int
    usable_values: int
    nonzero_values: int
    nonzero_fraction: float
    median_amplitude: float
    max_amplitude: float


def _sample_starts(nrows: int, sample_rows: int) -> Tuple[int, List[int]]:
    rows_per_chunk = max(1, sample_rows // 3)
    starts = sorted({
        0,
        max(0, (nrows - rows_per_chunk) // 2),
        max(0, nrows - rows_per_chunk),
    })
    return rows_per_chunk, starts


def _stats(amplitudes: List[np.ndarray], sampled_values: int,
           min_amplitude: float) -> VisibilityStats:
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


def visibility_stats(ms: Path, column: str, sample_rows: int,
                     min_amplitude: float) -> VisibilityStats:
    """Sample the start, middle, and end of an MS visibility column."""
    from casacore.tables import table

    with table(str(ms), ack=False) as measurement_set:
        nrows = len(measurement_set)
        if nrows == 0:
            return VisibilityStats(0, 0, 0, 0.0, 0.0, 0.0)

        rows_per_chunk, starts = _sample_starts(nrows, sample_rows)
        amplitudes = []
        sampled_values = 0
        for start in starts:
            count = min(rows_per_chunk, nrows - start)
            data = measurement_set.getcol(column, startrow=start, nrow=count)
            flags = measurement_set.getcol("FLAG", startrow=start, nrow=count)
            sampled_values += data.size
            usable = (~flags) & np.isfinite(data)
            amplitudes.append(np.abs(data[usable]))

    return _stats(amplitudes, sampled_values, min_amplitude)


def visibility_stats_arrays(data: np.ndarray, flag: np.ndarray, sample_rows: int,
                            min_amplitude: float) -> VisibilityStats:
    """visibility_stats on in-memory (nrow, nchan, ncorr) DATA and FLAG (gsi v3)."""
    nrows = int(data.shape[0])
    if nrows == 0:
        return VisibilityStats(0, 0, 0, 0.0, 0.0, 0.0)
    rows_per_chunk, starts = _sample_starts(nrows, sample_rows)
    amplitudes = []
    sampled_values = 0
    for start in starts:
        count = min(rows_per_chunk, nrows - start)
        d = data[start:start + count]
        sampled_values += d.size
        usable = (~flag[start:start + count]) & np.isfinite(d)
        amplitudes.append(np.abs(d[usable]))
    return _stats(amplitudes, sampled_values, min_amplitude)
