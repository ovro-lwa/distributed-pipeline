"""AOFlagger on one band-hour of peeled snapshots (gsi v3, ``v3.aoflag: hour``), in memory.

As the main pipeline flags its concatenated band-hour after peeling, but without an MS: the worker copies
every peeled slot (``slots`` layout) to disk, and this module runs the AOFlagger strategy on each
cross-correlation baseline's time x frequency image (all snapshots of the band-hour, 4 correlations as 8
real/imaginary images, the slots' flags as the initial mask) and ORs the result into every correlation's
FLAG in the slot copies, in place.  Autocorrelations are left as they are (not imaged).

Runs in the Python that has AOFlagger's bindings (``import aoflagger``, AOFlagger >= 3) and numpy -- on calim
that is the ``development`` env, Python 3.6, so this module is 3.6-compatible and reads the slot layout itself
(``_view``: the ``slots`` GSISLOT1 header and array table) instead of importing ``slots`` / ``snap``.  The worker
calls it as a subprocess:

    python -m gpu_subband_imaging.v3.aoflag_hour <slot list file> <strategy.lua> <processes> [<stats.json>]

Every row must be the same baseline in every slot (one band-hour of one array); this is checked.
"""
import json
import mmap
import multiprocessing
import os
import struct
import sys
import time
from typing import List

import numpy as np

MAGIC = b"GSISLOT1"          # as v3/slots.py
JSON_OFFSET = 64

BLOCK = 512          # baselines per read: (n_snap, BLOCK, nchan, 4) complex64 ~ 358 x 512 x 48 x 32 B = 280 MB


class _Slot:
    """The arrays of one slot file (v3/slots.py layout) as numpy views of a shared mmap."""

    def __init__(self, path: str, writable: bool):
        with open(path, "rb") as f:
            pre = f.read(JSON_OFFSET)
            if pre[:8] != MAGIC:
                raise ValueError("%s: not a GSISLOT1 slot" % path)
            n = struct.unpack("<I", pre[8:12])[0]
            h = json.loads(f.read(n).decode())
        fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, os.fstat(fd).st_size, mmap.MAP_SHARED,
                           mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0))
        finally:
            os.close(fd)
        for name in ("antenna1", "antenna2", "data", "flag"):
            e = h["arrays"][name]
            setattr(self, name, np.ndarray(tuple(e["shape"]), dtype=np.dtype(e["dtype"]), buffer=mm,
                                           offset=e["offset"]))


def _rows(paths: List[str]):
    """Cross-correlation row indices and the per-row baseline, checked equal across the slots."""
    s0 = _Slot(paths[0], writable=False)
    a1, a2 = np.asarray(s0.antenna1).copy(), np.asarray(s0.antenna2).copy()
    nchan = s0.data.shape[1]
    for p in paths[1:]:
        s = _Slot(p, writable=False)
        if s.data.shape[1:] != s0.data.shape[1:] or not (np.array_equal(s.antenna1, a1) and
                                                         np.array_equal(s.antenna2, a2)):
            raise ValueError("%s: rows / channels differ from %s; one band-hour of one array expected" % (p, paths[0]))
    return np.flatnonzero(a1 != a2), nchan


def _work(args):
    """Flag the cross rows `rows` (one process): returns (n samples, flagged before, flagged after)."""
    paths, strategy, rows = args
    import aoflagger
    af = aoflagger.AOFlagger()
    st = af.load_strategy_file(strategy)
    snaps = [_Slot(p, writable=True) for p in paths]
    nt, nf = len(snaps), snaps[0].data.shape[1]
    n = before = after = 0
    for b0 in range(0, len(rows), BLOCK):
        r = rows[b0:b0 + BLOCK]
        d = np.stack([s.data[r] for s in snaps])            # (nt, nb, nf, 4)
        f = np.stack([s.flag[r] for s in snaps])            # (nt, nb, nf, 4)
        init = f.any(axis=3)                                # (nt, nb, nf)
        new = np.empty_like(init)
        for k in range(len(r)):
            if init[:, k, :].all():                         # fully flagged: out | init == init, skip the strategy run
                new[:, k, :] = True
                continue
            iset = af.make_image_set(nt, nf, 8)
            for p in range(4):                              # AOFlagger buffers are [channel][time]
                iset.set_image_buffer(2 * p, np.ascontiguousarray(d[:, k, :, p].real.T, dtype=np.float32))
                iset.set_image_buffer(2 * p + 1, np.ascontiguousarray(d[:, k, :, p].imag.T, dtype=np.float32))
            mask = af.make_flag_mask(nt, nf, False)
            mask.set_buffer(np.ascontiguousarray(init[:, k, :].T))
            out = st.run(iset, mask)
            new[:, k, :] = out.get_buffer().T | init[:, k, :]
        n += new.size
        before += int(init.sum())
        after += int(new.sum())
        for t, s in enumerate(snaps):                       # OR into every correlation, in the slot copy
            s.flag[r] = f[t] | new[t][:, :, None]
    return n, before, after


def _check_bindings() -> None:
    """A flag-mask round trip: AOFlagger 3.2.0's Python FlagMask.set_buffer reads the bool buffer as doubles
    (smears every input flag over the next time samples); 3.2.1 is right.  Refuse a binding that changes it."""
    import aoflagger
    af = aoflagger.AOFlagger()
    m = np.random.RandomState(1).rand(7, 13) > 0.7          # (channels, times), as _work passes it
    mask = af.make_flag_mask(13, 7, False)
    mask.set_buffer(np.ascontiguousarray(m))
    if not np.array_equal(np.asarray(mask.get_buffer(), dtype=bool), m):
        raise RuntimeError("this AOFlagger's Python FlagMask does not round-trip a bool mask (AOFlagger 3.2.0 "
                           "binding bug?); use AOFlagger >= 3.2.1 (v3.aoflag_python)")


def flag_hour(paths: List[str], strategy: str, procs: int) -> dict:
    t0 = time.time()
    _check_bindings()
    rows, nchan = _rows(paths)
    parts = [rows[k::procs] for k in range(procs)] if procs > 1 else [rows]
    if procs > 1:
        with multiprocessing.get_context("spawn").Pool(procs) as pool:
            res = pool.map(_work, [(paths, strategy, p) for p in parts])
    else:
        res = [_work((paths, strategy, rows))]
    n, before, after = (sum(x[i] for x in res) for i in range(3))
    return dict(n_snap=len(paths), n_cross=int(len(rows)), nchan=int(nchan), procs=procs, samples=n,
                flag_frac_before=before / max(1, n), flag_frac_after=after / max(1, n),
                elapsed_s=round(time.time() - t0, 2))


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    paths = [p.strip() for p in open(a[0]) if p.strip()]
    stats = flag_hour(paths, a[1], int(a[2]))
    print("AOFLAG_HOUR " + json.dumps(stats), flush=True)
    if len(a) > 3:
        json.dump(stats, open(a[3], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
