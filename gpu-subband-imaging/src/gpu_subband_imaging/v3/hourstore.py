"""GSIV v1: one compact record file per band-hour (gsi v3, docs/v3_design.md 3.4 and 4).

A record holds what the hour imager (``ttcalx_imager.hour``) needs of one peeled snapshot, in
the form ``ttcalx_imager.msio.samples_from_msdata(..., compact=True)`` returns it: per kept
sample I and V (complex64); per cross-correlation row uvw (float64), natw (float32) and the
kept mask ``ok`` (bit-packed); and the metadata (R, time, tstart, freqs, chan_width, pd,
zero_frac, flagfrac, stem, ...).  ``ttcalx_imager.gsiv.load_archive`` rebuilds u, v, w, nw with
the same expressions as ``msio.load_samples``, so the hour job gets bit-identical samples.

File layout (all integers little-endian, every block 64-byte aligned)::

    block 0       file header   magic GSIVHDR1   JSON: format, version, mode, band, date, LST,
                                                 phase centre; arrays: a1, a2, ant_pos, freqs, ...
    block 1..n    records       magic GSIVREC1   JSON: scalars + array table; arrays: I, V, uvw, ...

    block = 64-B block header | JSON (padded to 64) | arrays (each padded to 64)
    block header = struct "<8sIIQQII" + 24 zero bytes:
        magic, meta_len, n_arrays, nbytes (whole block), seq (0 = file header, records 1..n),
        crc (crc32 of block bytes [64:nbytes]), hcrc (crc32 of the 36 header bytes before it)
    array table entry (in the JSON "arrays" list): name, dtype (numpy str, e.g. "<c8"), shape,
        offset (from the start of the data region = 64 + meta_len padded), nbytes

Beside it, ``<file>.idx.jsonl`` has one line per record (seq, stem, offset, nbytes, crc, n_kept,
time, tstart, R, zero_frac, flag_frac, peel).  A record is written, flushed and fsynced before
its index line, so the index never points at bytes that are not on disk.  The file is
self-describing: a reader that finds the index short (a crash between the record fsync and the
index line) scans the blocks after the last indexed one and accepts every complete block with a
good crc; whatever follows is a torn tail, which :class:`RecordWriter` truncates on resume (the
index is rewritten atomically).  A bad crc on an *indexed* record is corruption: readers raise
(``strict``) or drop that record; a resuming writer truncates the file there.

The reader is duplicated in ``ttcalx_imager/gsiv.py`` (imager venv); tests/test_v3_hourstore.py
cross-checks the two when the imager is importable.

This module needs numpy only.  Building records (``records_from_ms`` / ``records_from_snap``)
imports ``ttcalx_imager.msio`` (and casacore for R), i.e. runs in the imager venv: the worker
hook (``ArchiveClient``) starts ``python -m gpu_subband_imaging.v3.hourstore serve`` there, so the
records, their R and the hour job's own MS reading use the same code and libraries.
"""
from __future__ import annotations

import json
import os
import select
import struct
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# ============================================================================= GSIV v1 codec
# BEGIN shared section: copied verbatim into ttcalx_imager/gsiv.py (tests check they match)
FORMAT = "GSIV"
VERSION = 1
MAGIC_HDR = b"GSIVHDR1"
MAGIC_REC = b"GSIVREC1"
ALIGN = 64
BLOCK_HEADER = 64
_BH = struct.Struct("<8sIIQQI")          # 36 bytes, then hcrc (u32) and 24 zero bytes
_HCRC = struct.Struct("<I")
MODES = ("iv",)                          # "full" (full-pol + flags + weights) is not implemented yet
# per-record arrays that are left out when they equal the file header's (same for every snapshot)
SHARED_ARRAYS = ("a1", "a2", "ant_pos")
# metadata copied into each index line (the rest is in the record JSON)
INDEX_KEYS = ("stem", "time", "tstart", "zero_frac", "peel")

PathLike = Union[str, Path]


class GSIVError(Exception):
    pass


def _pad(n: int) -> int:
    return (-n) % ALIGN


def idx_path(path: PathLike) -> Path:
    return Path(f"{path}.idx.jsonl")


# ----------------------------------------------------------------------------- block codec
def _le(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return a


def encode_block(magic: bytes, seq: int, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]
                 ) -> Tuple[bytes, List[Any], int, int]:
    """(64-B header, payload pieces, total nbytes, crc) of one block; nothing is copied."""
    table, data = [], []
    off = 0
    for name, a in arrays.items():
        if a is None:
            continue
        a = _le(np.asarray(a))
        if a.dtype == np.bool_:
            raise GSIVError(f"array {name}: store bool masks bit-packed (pack_mask)")
        if a.dtype.hasobject:
            raise GSIVError(f"array {name}: object dtype")
        table.append(dict(name=name, dtype=a.dtype.str, shape=list(a.shape), offset=off, nbytes=int(a.nbytes)))
        data.append(a)
        off += a.nbytes + _pad(a.nbytes)
    m = dict(meta)
    m["arrays"] = table
    js = json.dumps(m, separators=(",", ":")).encode()
    pieces: List[Any] = [js, b"\0" * _pad(len(js))]
    for a in data:
        pieces.append(memoryview(a.reshape(-1).view(np.uint8)))
        pieces.append(b"\0" * _pad(a.nbytes))
    crc = 0
    for p in pieces:
        crc = zlib.crc32(p, crc)
    nbytes = BLOCK_HEADER + len(js) + _pad(len(js)) + off
    h = _BH.pack(magic, len(js), len(table), nbytes, seq, crc)
    header = h + _HCRC.pack(zlib.crc32(h)) + b"\0" * (BLOCK_HEADER - _BH.size - _HCRC.size)
    return header, pieces, nbytes, crc


def decode_header(b: bytes) -> Dict[str, Any]:
    if len(b) < BLOCK_HEADER:
        raise GSIVError("short block header")
    h = bytes(b[:_BH.size])
    (hcrc,) = _HCRC.unpack(bytes(b[_BH.size:_BH.size + _HCRC.size]))
    if zlib.crc32(h) != hcrc:
        raise GSIVError("block header checksum mismatch")
    magic, meta_len, n_arrays, nbytes, seq, crc = _BH.unpack(h)
    if magic not in (MAGIC_HDR, MAGIC_REC):
        raise GSIVError(f"bad block magic {magic!r}")
    if nbytes % ALIGN or nbytes < BLOCK_HEADER + meta_len:
        raise GSIVError("bad block length")
    return dict(magic=magic, meta_len=meta_len, n_arrays=n_arrays, nbytes=nbytes, seq=seq, crc=crc)


def decode_block(buf, off: int = 0, hdr: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """(meta, arrays) of the block at ``off`` of ``buf`` (bytes / numpy uint8 / memmap).
    The arrays are read-only views into ``buf``."""
    mv = memoryview(buf).cast("B") if not isinstance(buf, np.ndarray) else buf
    if hdr is None:
        hdr = decode_header(bytes(mv[off:off + BLOCK_HEADER]))
    js = bytes(mv[off + BLOCK_HEADER: off + BLOCK_HEADER + hdr["meta_len"]])
    meta = json.loads(js)
    d0 = off + BLOCK_HEADER + hdr["meta_len"] + _pad(hdr["meta_len"])
    arrays = {}
    for t in meta.pop("arrays"):
        dt = np.dtype(t["dtype"])
        cnt = int(np.prod(t["shape"], dtype=np.int64)) if t["shape"] else 1
        a = np.frombuffer(buf, dtype=dt, count=cnt, offset=d0 + t["offset"]).reshape(t["shape"])
        a.flags.writeable = False
        arrays[t["name"]] = a
    return meta, arrays


def block_crc(buf, off: int, nbytes: int) -> int:
    mv = memoryview(buf).cast("B") if not isinstance(buf, np.ndarray) else memoryview(buf)
    return zlib.crc32(mv[off + BLOCK_HEADER: off + nbytes])


def pack_mask(ok: np.ndarray) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(ok, dtype=np.bool_).reshape(-1), bitorder="little")


def unpack_mask(bits: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    n = int(np.prod(shape, dtype=np.int64))
    return np.unpackbits(bits, count=n, bitorder="little").reshape(tuple(shape)).astype(np.bool_)


def _pread(fd: int, n: int, off: int) -> bytes:
    out = []
    while n > 0:
        b = os.pread(fd, n, off)
        if not b:
            break
        out.append(b)
        n -= len(b); off += len(b)
    return b"".join(out)


def _preadinto(fd: int, buf: np.ndarray, off: int) -> int:
    mv = memoryview(buf).cast("B")
    got = 0
    while got < len(mv):
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            break
        got += k
    return got


# ----------------------------------------------------------------------------- scan / repair
@dataclass
class Scan:
    """Structure of a GSIV file: the valid records and what is wrong after them."""
    path: str
    size: int
    header_meta: Dict[str, Any]
    header_arrays: Dict[str, np.ndarray]
    header_nbytes: int
    entries: List[Dict[str, Any]]           # index entries of the valid records, file order
    end: int                                # end of the last valid record
    torn_bytes: int = 0                     # bytes after `end` that are not a valid record
    unindexed: int = 0                      # valid records found by scanning (index line missing)
    idx_dropped: int = 0                    # index lines that were unparsable or did not match
    bad_crc: List[int] = field(default_factory=list)   # seq of indexed records with a bad crc

    @property
    def clean(self) -> bool:
        return not (self.torn_bytes or self.unindexed or self.idx_dropped or self.bad_crc)

    @property
    def stems(self) -> List[str]:
        return [e["stem"] for e in self.entries]


def _entry(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], seq: int, off: int, nbytes: int, crc: int
           ) -> Dict[str, Any]:
    e = dict(seq=seq, offset=off, nbytes=nbytes, crc=crc, n_kept=int(meta.get("n", 0)))
    for k in INDEX_KEYS:
        if k in meta:
            e[k] = meta[k]
    e["flag_frac"] = meta.get("flagfrac")
    if "R" in arrays:
        e["R"] = np.asarray(arrays["R"], np.float64).tolist()
    return e


def _read_index(path: PathLike) -> Tuple[List[Dict[str, Any]], int]:
    """Index lines up to the first unparsable one; (entries, number of lines dropped)."""
    p = idx_path(path)
    if not p.exists():
        return [], 0
    lines = p.read_bytes().split(b"\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            return out, sum(1 for x in lines[i:] if x.strip())
    return out, 0


def scan(path: PathLike, verify: bool = False, workers: int = 8) -> Scan:
    """Validate the header, the index and the blocks after it (see the module docstring).

    verify=True also checks the crc of every indexed record (reads the whole file); records found
    by scanning past the index are always crc-checked."""
    path = str(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        hb = _pread(fd, BLOCK_HEADER, 0)
        try:
            h = decode_header(hb)
        except GSIVError as e:
            raise GSIVError(f"{path}: not a GSIV file ({e})") from None
        if h["magic"] != MAGIC_HDR or h["seq"] != 0 or h["nbytes"] > size:
            raise GSIVError(f"{path}: bad file header")
        hbuf = _pread(fd, h["nbytes"], 0)
        if block_crc(hbuf, 0, h["nbytes"]) != h["crc"]:
            raise GSIVError(f"{path}: file header crc mismatch")
        hmeta, harr = decode_block(hbuf, 0, h)
        if hmeta.get("format") != FORMAT or int(hmeta.get("version", 0)) != VERSION:
            raise GSIVError(f"{path}: unsupported format {hmeta.get('format')} v{hmeta.get('version')}")
        sc = Scan(path=path, size=size, header_meta=hmeta, header_arrays=harr, header_nbytes=h["nbytes"],
                  entries=[], end=h["nbytes"])
        idx, sc.idx_dropped = _read_index(path)
        # 1. index entries, while they describe consecutive complete blocks
        for i, e in enumerate(idx):
            ok = (isinstance(e, dict) and e.get("offset") == sc.end and e.get("seq") == len(sc.entries) + 1
                  and sc.end + int(e.get("nbytes", 0)) <= size)
            if ok:
                try:
                    bh = decode_header(_pread(fd, BLOCK_HEADER, sc.end))
                    ok = (bh["magic"] == MAGIC_REC and bh["nbytes"] == e["nbytes"] and bh["crc"] == e["crc"]
                          and bh["seq"] == e["seq"])
                except GSIVError:
                    ok = False
            if not ok:
                sc.idx_dropped += len(idx) - i
                break
            sc.entries.append(e)
            sc.end += int(e["nbytes"])
        # 2. complete blocks after the index (crash between record fsync and index line)
        while sc.end + BLOCK_HEADER <= size:
            try:
                bh = decode_header(_pread(fd, BLOCK_HEADER, sc.end))
            except GSIVError:
                break
            if bh["magic"] != MAGIC_REC or bh["seq"] != len(sc.entries) + 1 or sc.end + bh["nbytes"] > size:
                break
            buf = _pread(fd, bh["nbytes"], sc.end)
            if len(buf) != bh["nbytes"] or block_crc(buf, 0, bh["nbytes"]) != bh["crc"]:
                break
            meta, arrays = decode_block(buf, 0, bh)
            sc.entries.append(_entry(meta, arrays, bh["seq"], sc.end, bh["nbytes"], bh["crc"]))
            sc.unindexed += 1
            sc.end += bh["nbytes"]
        sc.torn_bytes = size - sc.end
        # 3. crc of the indexed records
        if verify:
            n_idx = len(sc.entries) - sc.unindexed

            def check(e):
                b = np.empty(int(e["nbytes"]), np.uint8)
                if _preadinto(fd, b, int(e["offset"])) != len(b):
                    return e["seq"]
                return None if block_crc(b, 0, len(b)) == e["crc"] else e["seq"]
            with ThreadPoolExecutor(max(1, workers)) as ex:
                sc.bad_crc = [s for s in ex.map(check, sc.entries[:n_idx]) if s is not None]
        return sc
    finally:
        os.close(fd)


def _fsync_dir(p: Path) -> None:
    try:
        fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_index(path: PathLike, entries: Sequence[Dict[str, Any]]) -> None:
    p = idx_path(path)
    tmp = Path(f"{p}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(p))
    _fsync_dir(p)


def repair(path: PathLike, log=None) -> Scan:
    """Make the file and its index consistent: truncate at the first bad record (torn tail or
    bad crc), rewrite the index for the valid prefix.  Returns the scan of the repaired file."""
    sc = scan(path, verify=True)
    if sc.bad_crc:
        first = min(sc.bad_crc)
        keep = [e for e in sc.entries if e["seq"] < first]
        sc.end = keep[-1]["offset"] + keep[-1]["nbytes"] if keep else sc.header_nbytes
        sc.entries = keep
    if not sc.clean:
        if log:
            log(f"GSIV repair {path}: {len(sc.entries)} valid records, torn {sc.torn_bytes} B, "
                f"unindexed {sc.unindexed}, index lines dropped {sc.idx_dropped}, bad crc {sc.bad_crc}"
                f" -> truncate to {sc.end}")
        with open(path, "r+b") as fh:
            fh.truncate(sc.end)
            fh.flush()
            os.fsync(fh.fileno())
        _write_index(path, sc.entries)
        sc = scan(path, verify=False)
    return sc


def read_blocks(path: PathLike, backend: str = "ram", verify: bool = True, strict: bool = True,
                workers: int = 8, log=None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray],
                                                     List[Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]]]:
    """(header meta, header arrays, [(meta, arrays, index entry)]) of the valid records of a
    GSIV file.  ram: one read per record (threads) into its own buffer, crc computed on that
    buffer; memmap: views of one read-only np.memmap.  A bad crc raises (strict) or drops the
    record.  Records without a1/a2/ant_pos get the header's."""
    sc = scan(path, verify=False)
    entries = sc.entries
    if backend == "memmap":
        mm = np.memmap(str(path), dtype=np.uint8, mode="r")
        bufs = [(mm, int(e["offset"])) for e in entries]
    elif backend == "ram":
        fd = os.open(str(path), os.O_RDONLY)
        try:
            def rd(e):
                b = np.empty(int(e["nbytes"]), np.uint8)
                if _preadinto(fd, b, int(e["offset"])) != len(b):
                    raise GSIVError(f"{path}: short read of record {e['seq']}")
                return b
            with ThreadPoolExecutor(max(1, workers)) as ex:
                bufs = [(b, 0) for b in ex.map(rd, entries)]
        finally:
            os.close(fd)
    else:
        raise ValueError("backend must be 'ram' or 'memmap'")
    bad = []
    if verify:
        with ThreadPoolExecutor(max(1, workers)) as ex:
            crcs = list(ex.map(lambda be: block_crc(be[0][0], be[0][1], int(be[1]["nbytes"])), zip(bufs, entries)))
        bad = [e["seq"] for e, c in zip(entries, crcs) if c != e["crc"]]
        if bad and strict:
            raise GSIVError(f"{path}: crc mismatch in record(s) {bad}")
        if bad and log:
            log(f"GSIV {path}: dropped record(s) with a bad crc: {bad}")
    out = []
    for (b, off), e in zip(bufs, entries):
        if e["seq"] in bad:
            continue
        meta, arrays = decode_block(b, off)
        for k in SHARED_ARRAYS:
            if k not in arrays and k in sc.header_arrays:
                arrays[k] = sc.header_arrays[k]
        out.append((meta, arrays, e))
    return sc.header_meta, sc.header_arrays, out


# ----------------------------------------------------------------------------- writer
class RecordWriter:
    """Append-only GSIV writer (one process per file).

    A new file gets ``header_meta`` (+ ``header_arrays``); an existing one is resumed: repaired
    (torn tail truncated) and appended to.  ``append`` returns the index entry once the record
    and its index line are on disk."""

    def __init__(self, path: PathLike, header_meta: Optional[Dict[str, Any]] = None,
                 header_arrays: Optional[Dict[str, np.ndarray]] = None, fsync: bool = True,
                 resume: bool = True, log=None):
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._f = None
        self._idx = None
        if self.path.exists() and self.path.stat().st_size > 0:
            if not resume:
                raise GSIVError(f"{self.path} exists (resume=False)")
            sc = repair(self.path, log=log)
            if header_meta is not None:
                for k in ("mode", "phase_centre"):
                    if k in header_meta and sc.header_meta.get(k) != header_meta.get(k):
                        raise GSIVError(f"{self.path}: {k} {sc.header_meta.get(k)!r} != {header_meta.get(k)!r}")
            self.header_meta, self.header_arrays = sc.header_meta, sc.header_arrays
            self.entries = list(sc.entries)
            self._end = sc.end
        else:
            if header_meta is None:
                raise GSIVError(f"{self.path}: new file needs header_meta")
            hm = dict(header_meta)
            hm.setdefault("format", FORMAT); hm.setdefault("version", VERSION); hm.setdefault("mode", "iv")
            if hm["mode"] not in MODES:
                raise GSIVError(f"GSIV mode {hm['mode']!r} not implemented (have {MODES})")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            hdr, pieces, nbytes, _ = encode_block(MAGIC_HDR, 0, hm, dict(header_arrays or {}))
            with open(self.path, "wb") as fh:
                fh.write(hdr)
                for p in pieces:
                    fh.write(p)
                fh.flush()
                os.fsync(fh.fileno())
            _write_index(self.path, [])
            self.header_meta = hm
            self.header_arrays = {k: np.asarray(v) for k, v in (header_arrays or {}).items()}
            self.entries = []
            self._end = nbytes
        self._f = open(self.path, "r+b")
        self._idx = open(idx_path(self.path), "a")
        self.stems = {e.get("stem") for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _strip_shared(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(arrays)
        for k in SHARED_ARRAYS:
            if k in out and k in self.header_arrays:
                a, h = np.asarray(out[k]), self.header_arrays[k]
                if a.shape == h.shape and a.dtype == h.dtype and np.array_equal(a, h):
                    del out[k]
        return out

    def append(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        with self._lock:
            if self._f is None:
                raise GSIVError("writer closed")
            seq = len(self.entries) + 1
            arrays = self._strip_shared(arrays)
            hdr, pieces, nbytes, crc = encode_block(MAGIC_REC, seq, meta, arrays)
            off = self._end
            self._f.seek(off)
            self._f.write(hdr)
            for p in pieces:
                self._f.write(p)
            self._f.flush()
            if self.fsync:
                os.fsync(self._f.fileno())
            e = _entry(meta, arrays, seq, off, nbytes, crc)
            self._idx.write(json.dumps(e, separators=(",", ":")) + "\n")
            self._idx.flush()
            if self.fsync:
                os.fsync(self._idx.fileno())
            self.entries.append(e)
            self.stems.add(e.get("stem"))
            self._end = off + nbytes
            return e

    def close(self) -> None:
        with self._lock:
            for f in (self._f, self._idx):
                if f is not None:
                    f.close()
            self._f = self._idx = None


# END shared section
# ============================================================================= 
# ----------------------------------------------------------------------------- reader / store
@dataclass
class Record:
    meta: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    entry: Dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.meta["stem"]

    @property
    def n(self) -> int:
        return int(self.meta["n"])

    def ok(self) -> np.ndarray:
        return unpack_mask(self.arrays["ok_bits"], self.meta["ok_shape"])

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays.values()))


class HourStore:
    """The compact records of one band-hour, in RAM (``backend="ram"``) or memory-mapped from
    the GSIV file (``"memmap"``); with a path, appends go to the file first (checkpoint).

    Opening an existing file loads its valid records (crc-checked with ``verify``); a store that
    is appended to resumes the file (torn tail truncated).  Records are in file order."""

    def __init__(self, path: Optional[PathLike] = None, backend: str = "ram",
                 header_meta: Optional[Dict[str, Any]] = None,
                 header_arrays: Optional[Dict[str, np.ndarray]] = None,
                 verify: bool = True, strict: bool = True, fsync: bool = True, workers: int = 8, log=None):
        if backend not in ("ram", "memmap"):
            raise ValueError("backend must be 'ram' or 'memmap'")
        if backend == "memmap" and path is None:
            raise ValueError("backend=memmap needs a path")
        self.path = Path(path) if path is not None else None
        self.backend = backend
        self.fsync = fsync
        self.log = log
        self._writer: Optional[RecordWriter] = None
        self._hm = dict(header_meta or {})
        self._ha = dict(header_arrays or {})
        self.records: List[Record] = []
        self.header_meta: Dict[str, Any] = dict(self._hm)
        self.header_arrays: Dict[str, np.ndarray] = dict(self._ha)
        if self.path is not None and self.path.exists() and self.path.stat().st_size > 0:
            self.header_meta, self.header_arrays, self.records = read_records(
                self.path, backend=backend, verify=verify, strict=strict, workers=workers)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Record:
        return self.records[i]

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    @property
    def stems(self) -> List[str]:
        return [r.stem for r in self.records]

    def nbytes(self) -> int:
        return int(sum(r.nbytes() for r in self.records))

    def append(self, rec: Record) -> Record:
        if self.path is not None:
            if self._writer is None:
                hm = self._hm or dict(mode="iv")
                ha = self._ha or {k: rec.arrays[k] for k in SHARED_ARRAYS if k in rec.arrays}
                self._writer = RecordWriter(self.path, hm, ha, fsync=self.fsync, log=self.log)
                self.header_meta, self.header_arrays = self._writer.header_meta, self._writer.header_arrays
                if len(self._writer) != len(self.records):     # file changed under us: reload
                    _, _, self.records = read_records(self.path, backend=self.backend)
            e = self._writer.append(rec.meta, rec.arrays)
            if self.backend == "memmap":
                mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=e["offset"], shape=(e["nbytes"],))
                meta, arrays = decode_block(mm, 0)
                rec = Record(meta, _with_shared(arrays, self.header_arrays), e)
            else:
                rec = Record(rec.meta, rec.arrays, e)
        self.records.append(rec)
        return rec

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _with_shared(arrays: Dict[str, np.ndarray], header_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(arrays)
    for k in SHARED_ARRAYS:
        if k not in out and k in header_arrays:
            out[k] = header_arrays[k]
    return out


def read_records(path: PathLike, backend: str = "ram", verify: bool = True, strict: bool = True,
                 workers: int = 8) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[Record]]:
    """(header meta, header arrays, records) of a GSIV file (see read_blocks)."""
    hm, ha, blocks = read_blocks(path, backend=backend, verify=verify, strict=strict, workers=workers)
    return hm, ha, [Record(m, a, e) for m, a, e in blocks]


# ----------------------------------------------------------------------------- building records
def record_from_compact(c: Dict[str, Any], peel: Optional[Dict[str, Any]] = None) -> Record:
    """Record of one snapshot from ``ttcalx_imager.msio.samples_from_msdata(..., compact=True)``."""
    pd_new = c.get("pd_new")
    meta = dict(kind="iv", stem=c["stem"], path=str(c["path"]), time=float(c["time"]), tstart=float(c["tstart"]),
                tmid=float(c["tmid"]), pd=[float(x) for x in c["pd"]],
                pd_new=None if pd_new is None or pd_new[0] is None else [float(x) for x in pd_new],
                nrow=int(c["nrow"]), nchan=int(c["nchan"]), n=int(c["n"]), flagfrac=float(c["flagfrac"]),
                zero_frac=float(c["zero_frac"]), telescope=str(c["telescope"]), observer=str(c["observer"]),
                field_name=str(c["field_name"]), extra=dict(c.get("extra") or {}),
                ok_shape=[int(x) for x in c["ok"].shape])
    if peel:
        meta["peel"] = peel
    arrays = dict(I=np.asarray(c["I"], np.complex64))
    if c.get("V") is not None:
        arrays["V"] = np.asarray(c["V"], np.complex64)
    arrays.update(uvw=np.asarray(c["uvw"], np.float64), natw=np.asarray(c["natw"], np.float32),
                  ok_bits=pack_mask(c["ok"]), R=np.asarray(c["R"], np.float64),
                  freqs=np.asarray(c["freqs"], np.float64), chan_width=np.asarray(c["chan_width"], np.float64),
                  a1=np.asarray(c["a1"], np.int32), a2=np.asarray(c["a2"], np.int32),
                  ant_pos=np.asarray(c["ant_pos"], np.float64))
    if arrays["I"].shape != (meta["n"],):
        raise GSIVError(f"{meta['stem']}: I has {arrays['I'].shape} samples, n = {meta['n']}")
    return Record(meta, arrays)


def records_from_ms(ms_path: PathLike, ra: Optional[float] = None, dec: Optional[float] = None,
                    want_v: bool = True, peel: Optional[Dict[str, Any]] = None) -> Record:
    """Record of one (peeled) snapshot MS: msio.read_ms + samples_from_msdata(compact=True),
    i.e. exactly what ``msio.load_samples(ms_path, ra, dec)`` computes.  Needs the imager venv."""
    from ttcalx_imager.msio import read_ms, samples_from_msdata
    ms = read_ms(str(ms_path), want_v=want_v)
    return record_from_compact(samples_from_msdata(ms, ra, dec, want_v, compact=True), peel)


def records_from_snap(snap, ra: Optional[float] = None, dec: Optional[float] = None, want_v: bool = True,
                      peel: Optional[Dict[str, Any]] = None) -> Record:
    """Record of one in-memory snapshot (v3.snap.SnapArrays, all rows in MS order) with the
    imager's own expressions (msio.msdata_from_arrays + samples_from_msdata)."""
    from ttcalx_imager.msio import msdata_from_arrays, samples_from_msdata
    m = snap.meta
    ms = msdata_from_arrays(m.get("path", f"{snap.stem}.ms"), snap.antenna1, snap.antenna2, snap.uvw,
                            snap.data, snap.flag, snap.weight, snap.time, m["freqs"], m["chan_width"],
                            m["phase_dir"], m["ant_pos"], telescope=m.get("telescope", "OVRO-LWA"),
                            observer=m.get("observer", ""), field_name=m.get("field_name", ""), want_v=want_v)
    return record_from_compact(samples_from_msdata(ms, ra, dec, want_v, compact=True), peel)


def header_for(band: int, date: str, group: str, phase_centre: Optional[str], run_label: str = "",
               mode: str = "iv", **extra) -> Dict[str, Any]:
    return dict(format=FORMAT, version=VERSION, mode=mode, band=int(band), date=str(date), group=str(group),
                run_label=str(run_label), phase_centre=phase_centre, host=os.uname().nodename,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)


# ----------------------------------------------------------------------------- pipeline paths
def archive_dir(cfg, band: int, group: str) -> Path:
    from ..config import v3_params
    pl = cfg.pipeline
    d = Path(v3_params(pl).archive_root) / f"{band}MHz" / str(pl.date)
    return d / pl.run_label / group if getattr(pl, "layout", "utc") == "lst" and pl.run_label else d / group


def archive_path(cfg, band: int, group: str, batch: int) -> Path:
    """<archive_root>/<band>MHz/<date>/[<run_label>/]<group>/<band>MHz-<date>-<group>-peeled-bNNN.gsiv
    (one file per worker batch: two batches of one band-hour never append to the same file)."""
    return archive_dir(cfg, band, group) / f"{band}MHz-{cfg.pipeline.date}-{group}-peeled-b{int(batch):03d}.gsiv"


def archive_files(cfg, band: int, group: str) -> List[Path]:
    d = archive_dir(cfg, band, group)
    return sorted(d.glob("*.gsiv")) if d.is_dir() else []


def archive_stems(files: Sequence[PathLike]) -> List[str]:
    """Stems of the valid records of the files (structure only, no crc), first occurrence kept,
    sorted -- the snapshots the hour job will load."""
    seen = set()
    for f in files:
        for s in scan(f, verify=False).stems:
            seen.add(s)
    return sorted(seen)


# ----------------------------------------------------------------------------- worker hook
class ArchiveClient:
    """Worker side of the archive writer: a ``hourstore serve`` process in the imager venv
    (same msio / casacore / numpy as the hour job), one per archive file."""

    def __init__(self, python: str, out: PathLike, phase_centre: Optional[str], header: Dict[str, Any],
                 env: Optional[Dict[str, str]] = None, log_path: Optional[PathLike] = None,
                 want_v: bool = True, timeout: float = 600.0):
        self.out = Path(out)
        self.timeout = timeout
        self._lock = threading.Lock()
        cmd = [python, "-m", "gpu_subband_imaging.v3.hourstore", "serve", "--out", str(out),
               "--header", json.dumps(header)]
        if phase_centre:
            cmd += ["--phase-centre", phase_centre]
        if not want_v:
            cmd.append("--no-v")
        self._log = open(log_path, "a") if log_path else None      # None: the worker's stderr
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                  env=env, text=True, bufsize=1)
        r = self._reply(self.timeout)
        if not r.get("ok"):
            raise GSIVError(f"archive writer did not start: {r}")
        self.info = r

    def _reply(self, timeout: float) -> Dict[str, Any]:
        fd = self.p.stdout.fileno()
        t_end = time.monotonic() + timeout
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                raise GSIVError(f"archive writer: no reply in {timeout:.0f} s")
            r, _, _ = select.select([fd], [], [], left)
            if r:
                line = self.p.stdout.readline()
                if not line:
                    raise GSIVError(f"archive writer exited (rc={self.p.poll()})")
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def add(self, ms_path: PathLike, peel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if self.p.poll() is not None:
                raise GSIVError(f"archive writer exited (rc={self.p.returncode})")
            self.p.stdin.write(json.dumps(dict(op="add", ms=str(ms_path), peel=peel)) + "\n")
            self.p.stdin.flush()
            r = self._reply(self.timeout)
        if not r.get("ok"):
            raise GSIVError(f"archive record {ms_path}: {r.get('error')}")
        return r

    def close(self, timeout: float = 60.0) -> None:
        with self._lock:
            try:
                if self.p.poll() is None:
                    self.p.stdin.write(json.dumps(dict(op="close")) + "\n")
                    self.p.stdin.flush()
                    self.p.wait(timeout)
            except Exception:
                pass
            if self.p.poll() is None:
                self.p.kill()
                self.p.wait()
            if self._log is not None:
                self._log.close()


def _serve(a) -> int:
    from ttcalx_imager.rotate import parse_phase_centre
    ra = dec = None
    if a.phase_centre:
        ra, dec = parse_phase_centre(a.phase_centre)
    header = json.loads(a.header) if a.header else {}
    header.setdefault("phase_centre", a.phase_centre)
    header["pd_new"] = None if ra is None else [ra, dec]
    w: Optional[RecordWriter] = None
    if Path(a.out).exists() and Path(a.out).stat().st_size > 0:
        w = RecordWriter(a.out, header, resume=True, fsync=not a.no_fsync,
                         log=lambda m: print(m, file=sys.stderr, flush=True))
    # the protocol owns the original stdout; anything a library prints goes to stderr (the log)
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out.write(json.dumps(dict(ok=True, n=len(w) if w else 0, pid=os.getpid())) + "\n"); out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            q = json.loads(line)
            if q.get("op") == "close":
                break
            t0 = time.perf_counter()
            rec = records_from_ms(q["ms"], ra, dec, want_v=not a.no_v, peel=q.get("peel"))
            t1 = time.perf_counter()
            if w is None:
                w = RecordWriter(a.out, header, {k: rec.arrays[k] for k in SHARED_ARRAYS},
                                 fsync=not a.no_fsync)
            if rec.stem in w.stems:
                r = dict(ok=True, stem=rec.stem, skipped="duplicate")
            else:
                e = w.append(rec.meta, rec.arrays)
                r = dict(ok=True, stem=rec.stem, seq=e["seq"], nbytes=e["nbytes"], n_kept=e["n_kept"],
                         build_s=t1 - t0, write_s=time.perf_counter() - t1)
        except Exception as e:      # report and keep serving
            r = dict(ok=False, error=repr(e))
        out.write(json.dumps(r) + "\n"); out.flush()
    if w is not None:
        w.close()
    out.close()
    return 0


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m gpu_subband_imaging.v3.hourstore", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="worker hook: append records of the MSes named on stdin (JSON lines)")
    s.add_argument("--out", required=True)
    s.add_argument("--phase-centre")
    s.add_argument("--header", default="")
    s.add_argument("--no-v", action="store_true")
    s.add_argument("--no-fsync", action="store_true")
    b = sub.add_parser("build", help="archive from (kept) snapshot MSes")
    b.add_argument("ms", nargs="+", help="MS paths or @listfile")
    b.add_argument("--out", required=True)
    b.add_argument("--phase-centre")
    b.add_argument("--lst-hour", type=int)
    b.add_argument("--band", type=int, default=0)
    b.add_argument("--date", default="")
    b.add_argument("--group", default="")
    b.add_argument("--no-v", action="store_true")
    i = sub.add_parser("info", help="validate a file (crc of every record) and print its summary")
    i.add_argument("file")
    i.add_argument("--repair", action="store_true", help="truncate a torn tail / rewrite the index")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return _serve(a)
    if a.cmd == "build":
        from ttcalx_imager.rotate import parse_phase_centre
        paths = []
        for it in a.ms:
            paths += [l.strip() for l in open(it[1:]) if l.strip()] if it.startswith("@") else [it]
        pc = a.phase_centre or (f"{a.lst_hour:02d}h30m00s 37d12m57.057s" if a.lst_hour is not None else None)
        ra, dec = parse_phase_centre(pc) if pc else (None, None)
        hm = header_for(a.band, a.date, a.group, pc, pd_new=None if ra is None else [ra, dec])
        w = None
        for p in sorted(paths, key=lambda x: os.path.basename(x.rstrip("/"))):
            rec = records_from_ms(p, ra, dec, want_v=not a.no_v)
            if w is None:
                w = RecordWriter(a.out, hm, {k: rec.arrays[k] for k in SHARED_ARRAYS})
            if rec.stem not in w.stems:
                e = w.append(rec.meta, rec.arrays)
                print(json.dumps(dict(stem=rec.stem, seq=e["seq"], nbytes=e["nbytes"], n_kept=e["n_kept"])), flush=True)
        if w:
            w.close()
        return 0
    sc = repair(a.file, log=print) if a.repair else scan(a.file, verify=True)
    print(json.dumps(dict(file=a.file, size=sc.size, records=len(sc.entries), end=sc.end, torn_bytes=sc.torn_bytes,
                          unindexed=sc.unindexed, idx_dropped=sc.idx_dropped, bad_crc=sc.bad_crc,
                          header={k: v for k, v in sc.header_meta.items()},
                          n_kept=sum(e.get("n_kept", 0) for e in sc.entries)), default=str, indent=1))
    return 0 if sc.clean else 1


if __name__ == "__main__":
    sys.exit(_main())
