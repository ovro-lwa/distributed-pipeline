"""Shared-memory snapshot slots for peeling without MS files (gsi v3 step 3, design 3.2).

A slot is one file in /dev/shm that holds a whole `SnapArrays`:

    bytes 0-7    magic b"GSISLOT1"
    bytes 8-11   u32 little endian: JSON length
    bytes 12-15  u32 little endian: state (FREE=0, PREPPED=1, PEELED=2)
    bytes 16-63  reserved (zero)
    bytes 64-    UTF-8 JSON, then zero padding up to `header_bytes` (a multiple of 4096)
    arrays       C-order, each at a 64-byte aligned offset

The JSON has `nrow, nchan, ncorr, stem, mjd, header_bytes, nbytes, state_offset`, an
`arrays` map `name -> {offset, dtype (numpy .str), shape}`, and `meta`, the JSON-able
values of `SnapArrays.meta`.  The numpy arrays of `meta` (freqs, chan_width, phase_dir,
ant_pos, ...) are stored as arrays named `meta.<key>`, so they are bit exact in both
readers.  `mjd = (min TIME + max TIME) / 2 / 86400`, as TTCalX `ms_mjd`.

The state lives in the fixed header word (not in the JSON) so it can be changed with a
4-byte store.  A writer sets FREE first and PREPPED only after every array is written,
so a crash mid-write never leaves a slot that looks ready.  workers/zest_daemon.jl
(`read_slot` / `write_slot!`) reads the same layout and sets PEELED.

`SlotRing` owns `n` reusable files `gsi3_<pid>_<ring>_<k>` in /dev/shm (`ring` counts the
rings of this process), removes them on exit, and on start removes stale ones left by a pid
that no longer runs.  Slot space is allocated with posix_fallocate, so a full tmpfs gives
OSError(ENOSPC) instead of a SIGBUS on the first write to a sparse page.
"""
from __future__ import annotations

import atexit
import errno
import itertools
import json
import mmap
import os
import re
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .snap import SnapArrays

MAGIC = b"GSISLOT1"
FREE, PREPPED, PEELED = 0, 1, 2
STATE_NAMES = {FREE: "FREE", PREPPED: "PREPPED", PEELED: "PEELED"}
STATE_OFFSET = 12
JSON_OFFSET = 64
ALIGN = 64
HEADER_ALIGN = 4096
SHM_ROOT = "/dev/shm"
# gsi3_<pid>_<ring>_<k>; gsi3_<pid>_<k> (the first layout) is cleaned up as well
_NAME = re.compile(r"gsi3_(\d+)_(\d+)(?:_(\d+))?")
_RING_IDS = itertools.count()

# SnapArrays array fields, in file order (small first, DATA and FLAG last)
FIELDS = ("antenna1", "antenna2", "uvw", "weight", "sigma", "flag_row", "time",
          "time_centroid", "exposure", "interval", "data", "flag")
_DTYPES = {"antenna1": "<i4", "antenna2": "<i4", "uvw": "<f8", "weight": "<f4",
           "sigma": "<f4", "flag_row": "|b1", "time": "<f8", "time_centroid": "<f8",
           "exposure": "<f8", "interval": "<f8", "data": "<c8", "flag": "|b1"}
# meta arrays the peel daemon needs (read_ms_fast takes them from the MS subtables)
REQUIRED_META = ("freqs", "phase_dir", "ant_pos")
PathLike = Union[str, Path]


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def ms_mjd(time: np.ndarray) -> float:
    """TTCalX ms_mjd: (min TIME + max TIME) / 2 / 86400, in float64."""
    return (float(np.min(time)) + float(np.max(time))) / 2 / 86400


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def _layout(snap: SnapArrays) -> Tuple[dict, Dict[str, np.ndarray]]:
    """The slot JSON (without header_bytes / offsets resolved) and the arrays to write."""
    arrays: Dict[str, np.ndarray] = {}
    for name in FIELDS:
        a = np.asarray(getattr(snap, name))
        want = np.dtype(_DTYPES[name])
        if a.dtype != want:
            # a value-preserving cast only (e.g. int64 antenna ids); never silently round data
            if not np.can_cast(a.dtype, want, casting="same_kind") or name in ("data", "uvw"):
                raise TypeError(f"slot field {name}: dtype {a.dtype}, expected {want}")
            b = a.astype(want)
            if not np.array_equal(b, a):
                raise TypeError(f"slot field {name}: {a.dtype} -> {want} changes values")
            a = b
        arrays[name] = a
    nrow, nchan, ncorr = arrays["data"].shape
    if arrays["flag"].shape != (nrow, nchan, ncorr):
        raise ValueError(f"flag shape {arrays['flag'].shape} != data shape {arrays['data'].shape}")
    for name in FIELDS:
        if arrays[name].shape[:1] != (nrow,):
            raise ValueError(f"slot field {name}: shape {arrays[name].shape}, expected nrow={nrow}")
    meta_json = {}
    for k, v in snap.meta.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind not in "biufc":
                raise TypeError(f"meta {k}: numpy dtype {v.dtype} not supported in a slot")
            arrays[f"meta.{k}"] = v
        else:
            meta_json[k] = _jsonable(v)
    for k in REQUIRED_META:
        if f"meta.{k}" not in arrays:
            raise ValueError(f"SnapArrays.meta[{k!r}] must be a numpy array for a slot")
    for k, ok, want in (("freqs", arrays["meta.freqs"].shape == (nchan,), f"({nchan},)"),
                        ("phase_dir", arrays["meta.phase_dir"].shape == (2,), "(2,) [ra, dec] rad"),
                        ("ant_pos", arrays["meta.ant_pos"].ndim == 2 and
                         arrays["meta.ant_pos"].shape[1] == 3, "(nant, 3)")):
        if not ok:
            raise ValueError(f"meta {k} shape {arrays['meta.' + k].shape}, expected {want}")
        if arrays["meta." + k].dtype != np.float64:
            raise TypeError(f"meta {k} must be float64 (the peel reads it bit exact)")
    head = {"format": "GSISLOT1", "nrow": int(nrow), "nchan": int(nchan), "ncorr": int(ncorr),
            "stem": snap.stem, "mjd": ms_mjd(arrays["time"]), "state_offset": STATE_OFFSET,
            "meta": meta_json}
    return head, arrays


def _place(head: dict, arrays: Dict[str, np.ndarray]) -> dict:
    """Fill in offsets, header_bytes and nbytes (header size grows with the JSON)."""
    header_bytes = HEADER_ALIGN
    while True:
        off = header_bytes
        table = {}
        for name, a in arrays.items():
            off = _align(off, ALIGN)
            table[name] = {"offset": off, "dtype": a.dtype.str, "shape": list(a.shape)}
            off += a.nbytes
        h = dict(head, header_bytes=header_bytes, nbytes=_align(off, ALIGN), arrays=table)
        blob = json.dumps(h, sort_keys=True).encode()
        need = _align(JSON_OFFSET + len(blob), HEADER_ALIGN)
        if need <= header_bytes:
            return h
        header_bytes = need


def slot_nbytes(snap: SnapArrays) -> int:
    """Bytes a slot needs for `snap`."""
    head, arrays = _layout(snap)
    return _place(head, arrays)["nbytes"]


def _allocate(fd: int, size: int) -> None:
    """Back the first `size` bytes with real tmpfs pages (OSError ENOSPC when it is full).

    A file that is only ftruncate'd is sparse: writing through an mmap into a page tmpfs
    cannot allocate kills the process with SIGBUS.  macOS (tests only) has no
    posix_fallocate; there the file is just extended.
    """
    if hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(fd, 0, size)
        except OSError as e:
            if e.errno in (errno.EOPNOTSUPP, errno.EINVAL):     # filesystem without fallocate
                if os.fstat(fd).st_size < size:
                    os.ftruncate(fd, size)
                return
            raise
    elif os.fstat(fd).st_size < size:
        os.ftruncate(fd, size)


def _open_map(path: PathLike, size: Optional[int] = None, writable: bool = True):
    flags = os.O_RDWR if writable else os.O_RDONLY
    if size is not None:
        flags |= os.O_CREAT
    fd = os.open(str(path), flags, 0o600)
    try:
        if size is not None:
            _allocate(fd, size)
        length = os.fstat(fd).st_size
        return mmap.mmap(fd, length, mmap.MAP_SHARED,
                         mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0))
    finally:
        os.close(fd)


def write_slot(path: PathLike, snap: SnapArrays) -> dict:
    """Write `snap` into the slot file `path` (created or grown as needed); state PREPPED.

    Returns the slot JSON.  The file is never shrunk, so a ring slot keeps its size.  Raises
    OSError (ENOSPC) before anything is written when /dev/shm has no room.
    """
    head, arrays = _layout(snap)
    h = _place(head, arrays)
    blob = json.dumps(h, sort_keys=True).encode()
    try:
        mm = _open_map(path, size=h["nbytes"])
    except OSError:
        try:                                   # old contents of a reused slot are not valid
            set_state(path, FREE)
        except (OSError, ValueError):
            pass
        raise
    try:
        mm[STATE_OFFSET:STATE_OFFSET + 4] = struct.pack("<I", FREE)     # not ready while writing
        mm[0:8] = MAGIC
        mm[8:12] = struct.pack("<I", len(blob))
        mm[16:JSON_OFFSET] = bytes(JSON_OFFSET - 16)
        mm[JSON_OFFSET:JSON_OFFSET + len(blob)] = blob
        end = JSON_OFFSET + len(blob)
        mm[end:h["header_bytes"]] = bytes(h["header_bytes"] - end)
        for name, a in arrays.items():
            e = h["arrays"][name]
            dst = np.ndarray(a.shape, dtype=a.dtype, buffer=mm, offset=e["offset"])
            dst[...] = a
            del dst
        mm[STATE_OFFSET:STATE_OFFSET + 4] = struct.pack("<I", PREPPED)
    finally:
        mm.close()
    return h


def read_header(path: PathLike) -> dict:
    """The slot JSON plus the current `state` (int) of the header word."""
    with open(path, "rb") as f:
        pre = f.read(JSON_OFFSET)
        if len(pre) < JSON_OFFSET or pre[:8] != MAGIC:
            raise ValueError(f"{path}: not a GSISLOT1 slot")
        n, state = struct.unpack("<II", pre[8:16])
        h = json.loads(f.read(n).decode())
    h["state"] = state
    return h


def get_state(path: PathLike) -> int:
    with open(path, "rb") as f:
        pre = f.read(16)
    if pre[:8] != MAGIC:
        raise ValueError(f"{path}: not a GSISLOT1 slot")
    return struct.unpack("<I", pre[STATE_OFFSET:STATE_OFFSET + 4])[0]


def set_state(path: PathLike, state: int) -> None:
    if state not in STATE_NAMES:
        raise ValueError(f"unknown slot state {state}")
    fd = os.open(str(path), os.O_RDWR)
    try:
        if os.pread(fd, 8, 0) != MAGIC:
            raise ValueError(f"{path}: not a GSISLOT1 slot")
        os.pwrite(fd, struct.pack("<I", state), STATE_OFFSET)
    finally:
        os.close(fd)


def view_slot(path: PathLike, writable: bool = True) -> SnapArrays:
    """Zero-copy SnapArrays over the slot file (numpy views of one shared mmap).

    The mapping stays alive as long as any of the arrays does.  `meta` holds the JSON
    meta values plus the `meta.<key>` arrays (views as well), and `meta["_slot"]` the
    slot JSON with its `state` at the time of the call.
    """
    h = read_header(path)
    mm = _open_map(path, writable=writable)
    if len(mm) < h["nbytes"]:
        mm.close()
        raise ValueError(f"{path}: file is {len(mm)} bytes, header says {h['nbytes']}")

    def arr(name):
        e = h["arrays"][name]
        a = np.ndarray(tuple(e["shape"]), dtype=np.dtype(e["dtype"]), buffer=mm,
                       offset=e["offset"])
        if not writable:
            a.flags.writeable = False
        return a

    meta = dict(h["meta"])
    for name in h["arrays"]:
        if name.startswith("meta."):
            meta[name[5:]] = arr(name)
    meta["_slot"] = h
    snap = SnapArrays(stem=h["stem"], meta=meta, **{f: arr(f) for f in FIELDS})
    del mm   # the arrays hold the buffer
    return snap


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale(root: PathLike = SHM_ROOT) -> List[str]:
    """Remove our gsi3_<pid>_... slot files whose pid is gone; returns the removed paths.

    Only files of this uid, with an exact slot name, whose pid does not run.  Race (accepted):
    if the pid of a crashed worker has been reused by a live process, its stale slots stay
    until that process exits; a live ring is never removed.
    """
    removed = []
    uid = os.getuid()
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return removed
    for name in names:
        m = _NAME.fullmatch(name)
        if not m:
            continue
        p = os.path.join(str(root), name)
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        pid = int(m.group(1))
        if st.st_uid != uid or pid == os.getpid() or _pid_alive(pid):
            continue
        try:
            os.unlink(p)
            removed.append(p)
        except FileNotFoundError:
            pass
    return removed


class SlotRing:
    """`n` reusable slot files /dev/shm/gsi3_<pid>_<ring>_<k>, handed out FREE and given back.

    `acquire()` blocks until a slot is free (this bounds the prep lookahead).  Each file gets
    `slot_bytes` of allocated tmpfs space up front (posix_fallocate: OSError ENOSPC here if
    /dev/shm cannot hold the ring), is grown by `write_slot` if a snapshot needs more, and is
    removed by `close()`, at interpreter exit, or by the next ring's stale cleanup after a crash.
    """

    def __init__(self, n: int, slot_bytes: int = 0, root: PathLike = SHM_ROOT):
        if n < 1:
            raise ValueError("SlotRing needs n >= 1")
        self.root = Path(root)
        self.removed_stale = cleanup_stale(self.root)
        self.pid = os.getpid()
        self.ring_id = next(_RING_IDS)
        self.paths = [self.root / f"gsi3_{self.pid}_{self.ring_id}_{k}" for k in range(n)]
        self._free = list(self.paths)
        self._cv = threading.Condition()
        self._closed = False
        try:
            for p in self.paths:
                fd = os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    _allocate(fd, max(int(slot_bytes), HEADER_ALIGN))
                finally:
                    os.close(fd)
        except BaseException:
            self._closed = True
            self._unlink_all()
            raise
        atexit.register(self.close)

    def __enter__(self) -> "SlotRing":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def n_free(self) -> int:
        with self._cv:
            return len(self._free)

    def acquire(self, timeout: Optional[float] = None) -> Path:
        """A free slot path (the caller writes it with write_slot)."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._free or self._closed, timeout):
                raise TimeoutError("no free shm slot")
            if self._closed:
                raise RuntimeError("SlotRing is closed")
            return self._free.pop(0)

    def release(self, path: PathLike) -> None:
        """Mark the slot FREE and hand it back to the ring."""
        path = Path(path)
        if path not in self.paths:
            raise ValueError(f"{path} is not a slot of this ring")
        with self._cv:
            if path in self._free:
                raise ValueError(f"{path} released twice")
            try:
                set_state(path, FREE)
            except (ValueError, FileNotFoundError):
                pass                       # never written (or already cleaned up)
            self._free.append(path)
            self._cv.notify()

    def _unlink_all(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        self._unlink_all()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
