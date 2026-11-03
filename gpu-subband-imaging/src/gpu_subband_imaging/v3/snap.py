"""One averaged, calibrated snapshot held in memory (gsi v3).

This is the hand-off between the in-memory prep (v3/fastprep.py), the shared-memory
peel slots (v3/slots.py) and the hour store (v3/hourstore.py).  The arrays are what the
v2 averaged MS holds in its main table (all rows, cross-correlations and autos, in MS
row order), so an MS written from a SnapArrays and read back gives the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SnapArrays:
    stem: str                 # e.g. 20261031_101709_23MHz
    antenna1: np.ndarray      # (nrow,) int32
    antenna2: np.ndarray      # (nrow,) int32
    uvw: np.ndarray           # (nrow, 3) float64, after fix_uvw
    data: np.ndarray          # (nrow, nchan, ncorr) complex64, ncorr = 4 (XX, XY, YX, YY)
    flag: np.ndarray          # (nrow, nchan, ncorr) bool
    weight: np.ndarray        # (nrow, ncorr) float32
    sigma: np.ndarray         # (nrow, ncorr) float32
    flag_row: np.ndarray      # (nrow,) bool
    time: np.ndarray          # (nrow,) float64, MS TIME (s, MJD seconds)
    time_centroid: np.ndarray  # (nrow,) float64
    exposure: np.ndarray      # (nrow,) float64
    interval: np.ndarray      # (nrow,) float64
    # Subtable values the consumers need: freqs (nchan,) Hz, chan_width (nchan,) Hz,
    # phase_dir (2,) rad, ant_pos (nant, 3) ITRF m, telescope, field_name, and anything
    # else a writer needs to rebuild the MS (kept as plain python / numpy values).
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def nrow(self) -> int:
        return int(self.antenna1.shape[0])

    @property
    def nchan(self) -> int:
        return int(self.data.shape[1])

    def nbytes(self) -> int:
        return int(sum(getattr(self, k).nbytes for k in
                       ("antenna1", "antenna2", "uvw", "data", "flag", "weight", "sigma",
                        "flag_row", "time", "time_centroid", "exposure", "interval")))
