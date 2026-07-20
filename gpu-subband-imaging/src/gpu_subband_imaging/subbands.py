"""Match measurement-set and calibration SPWs by frequency overlap."""
from __future__ import annotations

from typing import List

import numpy as np


def _chan_freqs(caltable_or_ms: str) -> List[np.ndarray]:
    from casacore.tables import table  # runtime dep (worker/host only)
    with table(caltable_or_ms + "/SPECTRAL_WINDOW", ack=False) as t:
        return [t.getcell("CHAN_FREQ", i) for i in range(t.nrows())]


def spwmap(ms_path: str, caltable_path: str) -> List[int]:
    """One entry per MS SPW: the caltable SPW with max frequency overlap."""
    ms_freqs = _chan_freqs(ms_path)
    cal_freqs = _chan_freqs(caltable_path)
    out = []
    for mf in ms_freqs:
        ms_min, ms_max = float(np.min(mf)), float(np.max(mf))
        best, best_overlap = -1, 0.0
        for i, cf in enumerate(cal_freqs):
            lo, hi = max(ms_min, float(np.min(cf))), min(ms_max, float(np.max(cf)))
            if hi > lo and (hi - lo) > best_overlap:
                best_overlap, best = hi - lo, i
        if best < 0:  # no overlap: nearest centre
            c = float(np.mean(mf))
            best = int(np.argmin([abs(float(np.mean(cf)) - c) for cf in cal_freqs]))
        out.append(int(best))
    return out
