"""Peel an averaged MS through a shared-memory slot (gsi v3 step 3, `v3.peel_io: shm`).

Until the in-memory prep (v3/fastprep.py) hands over SnapArrays directly, the worker and
the equivalence tests go MS -> SnapArrays -> slot -> "SHM <slot>" -> slot -> MS:
`snap_from_ms` reads what the peel needs from the averaged MS, the daemon peels the slot
in place, and `write_peeled` puts DATA and FLAG back into the MS.  The MS then holds the
bytes the v2 path (`ZestDaemon.zest(ms)`: read_ms_fast / write_ms_fast!) writes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from . import slots as _slots
from .snap import SnapArrays

PathLike = Union[str, Path]
_MAIN = {"antenna1": "ANTENNA1", "antenna2": "ANTENNA2", "uvw": "UVW", "data": "DATA",
         "flag": "FLAG", "weight": "WEIGHT", "sigma": "SIGMA", "flag_row": "FLAG_ROW",
         "time": "TIME", "time_centroid": "TIME_CENTROID", "exposure": "EXPOSURE",
         "interval": "INTERVAL"}


def snap_from_ms(ms: PathLike, column: str = "DATA") -> SnapArrays:
    """SnapArrays of an averaged single-SPW, single-field MS (main rows in MS order)."""
    from casacore.tables import table
    ms = str(ms)
    with table(ms, ack=False) as t:
        cols = {k: t.getcol(column if c == "DATA" else c) for k, c in _MAIN.items()}
    with table(ms + "/SPECTRAL_WINDOW", ack=False) as s:
        if s.nrows() != 1:
            raise ValueError(f"{ms}: expected one SPW, found {s.nrows()}")
        freqs = np.asarray(s.getcol("CHAN_FREQ")[0], dtype=np.float64)
        chan_width = np.asarray(s.getcol("CHAN_WIDTH")[0], dtype=np.float64)
    with table(ms + "/FIELD", ack=False) as f:
        phase_dir = np.asarray(f.getcol("PHASE_DIR")[0, 0, :2], dtype=np.float64)
        field_name = str(f.getcol("NAME")[0])
    with table(ms + "/ANTENNA", ack=False) as a:
        ant_pos = np.asarray(a.getcol("POSITION"), dtype=np.float64)
    meta: Dict[str, object] = dict(freqs=freqs, chan_width=chan_width, phase_dir=phase_dir,
                                   ant_pos=ant_pos, field_name=field_name, source_ms=ms)
    try:
        with table(ms + "/OBSERVATION", ack=False) as o:
            meta["telescope"] = str(o.getcol("TELESCOPE_NAME")[0])
            meta["observer"] = str(o.getcol("OBSERVER")[0])
    except Exception:
        pass
    stem = Path(ms).name
    for suffix in ("_avg.ms", ".ms"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return SnapArrays(stem=stem, meta=meta, **cols)


def write_peeled(ms: PathLike, snap: SnapArrays, column: str = "DATA") -> None:
    """Put the peeled DATA (as `column`) and FLAG of `snap` into `ms` (same rows)."""
    from casacore.tables import table
    with table(str(ms), readonly=False, ack=False) as t:
        if t.nrows() != snap.nrow:
            raise ValueError(f"{ms} has {t.nrows()} rows, the slot {snap.nrow}")
        t.putcol(column, np.ascontiguousarray(snap.data))
        t.putcol("FLAG", np.ascontiguousarray(snap.flag))
        t.flush()


class SlotUnavailable(RuntimeError):
    """The snapshot could not be put into a slot (e.g. /dev/shm full: ENOSPC).

    Raised before the daemon saw the snapshot, with the MS untouched, so the caller can
    peel it through the MS instead (v2 path).
    """


def peel_ms_via_slot(daemon, ms: PathLike, ring: "_slots.SlotRing",
                     timeout: Optional[float] = 600.0,
                     timings: Optional[Dict[str, float]] = None) -> float:
    """Peel `ms` in place through one slot of `ring`; returns the daemon's elapsed time.

    `timings` (optional) receives the python side of the hand-off in seconds: ms_read
    (snap_from_ms), slot_write (write_slot), ms_write (slot -> MS write-back).  In the full
    v3 path the prep writes the slot from memory, so only slot_write remains of these.

    Raises SlotUnavailable when the slot cannot be written (MS untouched).  When something
    fails on this side after the daemon peeled the slot (state check, MS write-back), the
    daemon's warm-start gains are reset, as the daemon itself does after FAILED.
    """
    slot = ring.acquire()
    try:
        t0 = time.perf_counter()
        snap = snap_from_ms(ms)
        t1 = time.perf_counter()
        try:
            _slots.write_slot(slot, snap)
        except OSError as e:
            raise SlotUnavailable(f"cannot write slot {slot}: {e}") from e
        del snap
        t2 = time.perf_counter()
        elapsed = daemon.zest_slot(slot, timeout=timeout)
        try:
            state = _slots.get_state(slot)
            if state != _slots.PEELED:
                raise RuntimeError(f"slot {slot} is {_slots.STATE_NAMES.get(state, state)} "
                                   f"after ZESTED")
            t3 = time.perf_counter()
            peeled = _slots.view_slot(slot, writable=False)
            write_peeled(ms, peeled)
            del peeled
            sol = Path(f"{slot}.peelsol")       # peel.save_solutions: they belong to the MS
            if sol.is_dir():
                import shutil
                dst = Path(f"{ms}.peelsol")
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(sol), str(dst))
        except BaseException:
            try:
                daemon.reset_warm()
            except Exception as e:                    # the original error matters more
                import logging
                logging.getLogger("gsi.peel").error("RESET_WARM failed: %s", e)
            raise
        if timings is not None:
            timings.update(ms_read=t1 - t0, slot_write=t2 - t1,
                           ms_write=time.perf_counter() - t3)
        return elapsed
    finally:
        ring.release(slot)
