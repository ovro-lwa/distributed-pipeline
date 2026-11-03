"""GSIV v1: one compact record file per band-hour (gsi v3, docs/v3_design.md 3.4 and 4).

A record holds what the hour imager (``ttcalx_imager.hour``) needs of one peeled snapshot, in
the form ``ttcalx_imager.msio.samples_from_msdata(..., compact=True)`` returns it: per kept
sample I and V (complex64); per cross-correlation row uvw (float64), natw (float32) and the
kept mask ``ok`` (bit-packed); and the metadata (R, time, tstart, freqs, chan_width, pd,
zero_frac, flagfrac, stem, ...).  ``ttcalx_imager.gsiv.load_archive`` rebuilds u, v, w, nw with
the same expressions as ``msio.load_samples``, so u, v, w, nw, I and V are bit-identical to
``load_samples`` of the MS.  R is what ``load_samples`` computes in the writer's process: casacore
measures caches make R depend on the order of the calls in a process (|dR| ~ 1e-10, a phase of
~1e-7 rad), so R equals a sequential ``load_samples`` in the same order (``build`` sorts by
stem) but not a process-pool load -- v2's own hour job (load_all, nproc 8) has that property
too, between any two runs.

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
time, tstart, R, zero_frac, flag_frac, peel, written_ns), and ``<file without .gsiv>.expected.jsonl`` lists the
snapshots the worker archived (the hour job's check that the archive is complete).  A record is written, flushed and fsynced before
its index line, so the index never points at bytes that are not on disk.  The file is
self-describing: a reader that finds the index short (a crash between the record fsync and the
index line) scans the blocks after the last indexed one and accepts every complete block with a
good crc; whatever follows is a torn tail, which :class:`RecordWriter` truncates on resume (the
index is rewritten atomically).  A bad crc on an *indexed* record is corruption.  Readers are
strict by default: torn bytes, dropped index lines, records without an index line, a bad crc
or a record whose freqs / chan_width differ from the header's raise GSIVError (strict=False
drops and reports them); a resuming writer truncates the file at the first defect.

A stem may be appended again (a reprocessed snapshot): the newest record wins on load, by
(written_ns, seq), then path -- also across the files of a band-hour.  A resume checks the
header (mode, band, date, group, run_label, calibration tables, phase centre, freqs) and
refuses another one.  One writer per file: an O_EXCL lock file ``<file>.lock`` (pid, host),
touched on every append, taken over when its process is gone or it is stale.

The reader is duplicated in ``ttcalx_imager/gsiv.py`` (imager venv); tests/test_v3_hourstore.py
cross-checks the two when the imager is importable.

This module needs numpy only.  Building records (``records_from_ms`` / ``records_from_snap``)
imports ``ttcalx_imager.msio`` (and casacore for R), i.e. runs in the imager venv: the worker
hook (``ArchiveClient``) starts ``python -m gpu_subband_imaging.v3.hourstore serve`` there, so the
records and the hour job's own MS reading use the same code and libraries.
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
# header arrays every record must match (a record of another band / channelisation is refused)
CHECK_ARRAYS = ("freqs", "chan_width")
# header keys a resumed file must match (a rerun with another band, run, calibration or phase
# centre must not append to it)
RESUME_KEYS = ("mode", "phase_centre", "pd_new", "band", "date", "group", "run_label", "cal_bp", "cal_xy")
# metadata copied into each index line (the rest is in the record JSON)
INDEX_KEYS = ("stem", "time", "tstart", "zero_frac", "peel", "written_ns")
# a writer lock whose file was not touched for this long is stale (the writer touches it on
# every append); GSIV_LOCK_STALE_S overrides
LOCK_STALE_S = 1800.0

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
    end = off + hdr["nbytes"]
    if end > len(mv):
        raise GSIVError(f"block at {off}: {hdr['nbytes']} B but only {len(mv) - off} B in the buffer")
    js = bytes(mv[off + BLOCK_HEADER: off + BLOCK_HEADER + hdr["meta_len"]])
    try:
        meta = json.loads(js)
        table = meta.pop("arrays")
    except (ValueError, KeyError, AttributeError) as e:
        raise GSIVError(f"block at {off}: bad JSON ({e})") from None
    d0 = off + BLOCK_HEADER + hdr["meta_len"] + _pad(hdr["meta_len"])
    arrays = {}
    for t in table:                          # bounds-checked whether or not the crc was verified
        try:
            dt = np.dtype(t["dtype"])
            shape = [int(x) for x in t["shape"]]
            o, nb = int(t["offset"]), int(t["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise GSIVError(f"block at {off}: bad array table entry {t!r} ({e})") from None
        cnt = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if (dt.hasobject or min(shape, default=0) < 0 or o < 0 or o % ALIGN or nb != cnt * dt.itemsize
                or d0 + o + nb > end):
            raise GSIVError(f"block at {off}: array {t.get('name')!r} outside the block or inconsistent")
        a = np.frombuffer(buf, dtype=dt, count=cnt, offset=d0 + o).reshape(shape)
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

    def problems(self) -> List[str]:
        out = []
        if self.torn_bytes:
            out.append(f"{self.torn_bytes} torn bytes after record {len(self.entries)}")
        if self.idx_dropped:
            out.append(f"{self.idx_dropped} index line(s) dropped")
        if self.unindexed:
            out.append(f"{self.unindexed} record(s) without an index line")
        if self.bad_crc:
            out.append(f"crc mismatch in record(s) {self.bad_crc}")
        return [f"{self.path}: {p}" for p in out]

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


def repair(path: PathLike, log=None, lock: bool = True) -> Scan:
    """Make the file and its index consistent: truncate at the first bad record (torn tail or
    bad crc), rewrite the index for the valid prefix.  Returns the scan of the repaired file.
    lock: take the writer lock for the repair (a RecordWriter already holds it)."""
    lk = WriterLock(path) if lock else None
    try:
        return _repair(path, log)
    finally:
        if lk is not None:
            lk.release()


def _repair(path: PathLike, log=None) -> Scan:
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
                workers: int = 8, log=None, report: Optional[Dict[str, Any]] = None
                ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray],
                           List[Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]]]:
    """(header meta, header arrays, [(meta, arrays, index entry)]) of the records of a GSIV file,
    in file order.  ram: one read per record (threads) into its own buffer, crc computed on that
    buffer; memmap: views of one read-only np.memmap.

    strict (default): any defect raises GSIVError -- torn bytes, dropped index lines, records
    without an index line, a bad crc, a record whose freqs / chan_width differ from the header's.
    strict=False: the defective records are dropped and the defects listed in ``report``
    (dict, filled in: problems, n_records, n_dropped).  Records without a1/a2/ant_pos get the
    header's."""
    sc = scan(path, verify=False)
    entries = sc.entries
    problems = sc.problems()
    if problems and strict:
        raise GSIVError("; ".join(problems))
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
        if bad:
            problems.append(f"{path}: crc mismatch in record(s) {bad}")
            if strict:
                raise GSIVError(problems[-1])
    out = []
    dropped = len(bad)
    for (b, off), e in zip(bufs, entries):
        if e["seq"] in bad:
            continue
        try:
            meta, arrays = decode_block(b, off)
        except GSIVError as err:
            problems.append(f"{path}: record {e['seq']}: {err}")
            if strict:
                raise GSIVError(problems[-1]) from None
            dropped += 1
            continue
        bad_arr = [k for k in CHECK_ARRAYS if k in sc.header_arrays and not _same_array(arrays.get(k), sc.header_arrays[k])]
        if bad_arr:
            problems.append(f"{path}: record {e['seq']} ({meta.get('stem')}): {bad_arr} differ from the header's "
                            f"(another band?)")
            if strict:
                raise GSIVError(problems[-1])
            dropped += 1
            continue
        for k in SHARED_ARRAYS:
            if k not in arrays and k in sc.header_arrays:
                arrays[k] = sc.header_arrays[k]
        out.append((meta, arrays, e))
    if problems and log:
        log("GSIV " + "; ".join(problems))
    if report is not None:
        report.update(problems=problems, n_records=len(entries), n_dropped=dropped, header=sc.header_meta)
    return sc.header_meta, sc.header_arrays, out


def _same_array(a, b) -> bool:
    if a is None or b is None:
        return False
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def newest(blocks_by_file: Sequence[Tuple[str, Sequence[Tuple[Dict[str, Any], Any, Dict[str, Any]]]]]):
    """Newest record per stem over several files: the largest (written_ns, seq) wins, the file
    path breaks a tie -- deterministic and independent of the order the files are given in (a
    reprocessed snapshot supersedes its old record, as v2's keep_ms overwrites the kept MS).
    Returns ({stem: (path, block)}, [(stem, superseded path, seq)])."""
    best: Dict[str, Tuple[Tuple[int, int, str], str, Any]] = {}
    superseded = []
    for path, blocks in blocks_by_file:
        for blk in blocks:
            meta, _, e = blk
            key = (int(meta.get("written_ns", 0)), int(e["seq"]), str(path))
            st = meta["stem"]
            if st not in best or key > best[st][0]:
                if st in best:
                    superseded.append((st, best[st][1], best[st][0][1]))
                best[st] = (key, str(path), blk)
            else:
                superseded.append((st, str(path), int(e["seq"])))
    return {st: (p, blk) for st, (_, p, blk) in best.items()}, superseded


class WriterLock:
    """O_EXCL lock file ``<file>.lock`` (pid, host, time) -- flock is not reliable on Lustre.
    A lock of a dead process on this host, or one not touched for GSIV_LOCK_STALE_S (default
    1800 s; the writer touches it on every append), is taken over."""

    def __init__(self, path: PathLike):
        self.path = Path(f"{path}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stale_s = float(os.environ.get("GSIV_LOCK_STALE_S", LOCK_STALE_S))
        me = dict(pid=os.getpid(), host=os.uname().nodename, t=time.time())
        for _ in range(3):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    other = json.loads(self.path.read_text() or "{}")
                    age = time.time() - self.path.stat().st_mtime
                except (OSError, ValueError):
                    other, age = {}, 0.0
                dead = other.get("host") == me["host"] and not _pid_alive(int(other.get("pid", -1)))
                if dead or age > stale_s:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise GSIVError(f"{path} is being written by pid {other.get('pid')} on {other.get('host')} "
                                f"(lock {self.path}, {age:.0f} s old)") from None
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(me))
            self.held = True
            return
        raise GSIVError(f"cannot take the lock {self.path}")

    def touch(self) -> None:
        try:
            os.utime(str(self.path))
        except OSError:
            pass

    def release(self) -> None:
        if getattr(self, "held", False):
            try:
                if json.loads(self.path.read_text()).get("pid") == os.getpid():
                    self.path.unlink()
            except (OSError, ValueError):
                pass
            self.held = False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        self._flock = WriterLock(self.path)
        try:
            self._open(header_meta, header_arrays, resume, log)
        except BaseException:
            self._flock.release()
            raise

    def _open(self, header_meta, header_arrays, resume, log) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            if not resume:
                raise GSIVError(f"{self.path} exists (resume=False)")
            sc = scan(self.path)                 # header checks before anything is truncated
            if header_meta is not None:
                for k in RESUME_KEYS:
                    if k in header_meta and sc.header_meta.get(k) != header_meta.get(k):
                        raise GSIVError(f"{self.path}: cannot resume, header {k} {sc.header_meta.get(k)!r} != "
                                        f"{header_meta.get(k)!r} (remove the file to start over)")
            for k in CHECK_ARRAYS:
                if header_arrays and k in header_arrays and k in sc.header_arrays and \
                        not _same_array(np.asarray(header_arrays[k]), sc.header_arrays[k]):
                    raise GSIVError(f"{self.path}: cannot resume, header {k} differ (another band?)")
            sc = _repair(self.path, log=log)
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
        self.lock = self._flock
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
        """Append one record (a stem that is already in the file is appended again: the newest
        record of a stem wins on load).  Refused if its freqs / chan_width differ from the
        header's."""
        with self._lock:
            if self._f is None:
                raise GSIVError("writer closed")
            for k in CHECK_ARRAYS:
                if k in self.header_arrays and not _same_array(arrays.get(k), self.header_arrays[k]):
                    raise GSIVError(f"{self.path}: record {meta.get('stem')}: {k} differ from the header's "
                                    f"(another band?)")
            meta = dict(meta)
            meta["written_ns"] = max(time.time_ns(), max((int(e.get("written_ns", 0)) for e in self.entries),
                                                         default=0) + 1)
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
            self._flock.touch()
            return e

    def close(self) -> None:
        with self._lock:
            for f in (self._f, self._idx):
                if f is not None:
                    f.close()
            self._f = self._idx = None
            self._flock.release()


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
                ha = self._ha or {k: rec.arrays[k] for k in SHARED_ARRAYS + CHECK_ARRAYS if k in rec.arrays}
                self._writer = RecordWriter(self.path, hm, ha, fsync=self.fsync, log=self.log)
                self.header_meta, self.header_arrays = self._writer.header_meta, self._writer.header_arrays
                if len(self._writer) != len(self.records):     # repaired on resume: reload
                    _, _, self.records = read_records(self.path, backend=self.backend)
            meta = rec.meta
            e = self._writer.append(meta, rec.arrays)
            if self.backend == "memmap":
                mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=e["offset"], shape=(e["nbytes"],))
                meta, arrays = decode_block(mm, 0)
                rec = Record(meta, _with_shared(arrays, self.header_arrays), e)
            else:
                rec = Record(dict(rec.meta, written_ns=e["written_ns"]), rec.arrays, e)
        self.records.append(rec)
        return rec

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def latest(self) -> Dict[str, Record]:
        """The newest record of every stem (records are kept in append / file order)."""
        out: Dict[str, Record] = {}
        for r in self.records:
            out[r.stem] = r
        return out

    def spill(self, path: Optional[PathLike] = None) -> Path:
        """Switch a RAM store to memory-mapped records (gsi v3 memory policy, V3_MEM_SPILL).

        A store with a file (checkpoint) already has every record there; a RAM-only store first
        writes its records to ``path``.  Afterwards the RAM copies are dropped (the arrays are
        views of the file, read from the page cache / disk when used) and later appends are
        memory-mapped as well.  Returns the file."""
        if self.backend == "memmap":
            return self.path                     # type: ignore[return-value]
        if self.path is None:
            if path is None:
                raise ValueError("HourStore.spill: a RAM-only store needs a path")
            self.path = Path(path)
            if self.path.exists() and self.path.stat().st_size > 0:
                raise GSIVError(f"{self.path} exists: a RAM-only store does not spill into another file")
            old, self.records = self.records, []
            for r in old:
                self.append(Record(r.meta, r.arrays))     # writes the file (first append makes it)
            del old
        if self._writer is not None:
            self._writer._f.flush()
        _, _, self.records = read_records(self.path, backend="memmap", verify=False)
        self.backend = "memmap"
        return self.path


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
                    want_v: bool = True, peel: Optional[Dict[str, Any]] = None, stem: Optional[str] = None,
                    path: Optional[str] = None) -> Record:
    """Record of one (peeled) snapshot MS: msio.read_ms + samples_from_msdata(compact=True),
    i.e. exactly what ``msio.load_samples(ms_path, ra, dec)`` computes.  Needs the imager venv.
    stem / path: the name the snapshot has in the hour job (the worker's MS is the work-dir
    ``<stamp>_avg.ms``; the kept MS, and so load_samples, call it ``<stamp>_<band>MHz``)."""
    from ttcalx_imager.msio import read_ms, samples_from_msdata
    ms = read_ms(str(ms_path), want_v=want_v)
    rec = record_from_compact(samples_from_msdata(ms, ra, dec, want_v, compact=True), peel)
    if stem:
        rec.meta["stem"] = str(stem)
    if path:
        rec.meta["path"] = str(path)
    return rec


def records_from_snap(snap, ra: Optional[float] = None, dec: Optional[float] = None, want_v: bool = True,
                      peel: Optional[Dict[str, Any]] = None, stem: Optional[str] = None,
                      path: Optional[str] = None, R: Optional[np.ndarray] = None) -> Record:
    """Record of one in-memory snapshot (v3.snap.SnapArrays, all rows in MS order) with the
    imager's own expressions (msio.msdata_from_arrays + samples_from_msdata), i.e. what
    ``records_from_ms`` gives for the MS written from ``snap``.

    stem / path: the record's names (as records_from_ms; default ``snap.stem`` and
    ``meta["path"]`` or ``<stem>.ms``).  R: the rotation to (ra, dec) if the caller computed it already with
    ``rotate.snapshot_rotation(cross uvw, a1, a2, ant_pos, TIME[0], ra, dec)`` -- the call
    samples_from_msdata makes (the engine keeps every casacore measures call on one thread).
    The record never refers to ``snap``'s memory (a reused shm slot): every array is a copy."""
    from ttcalx_imager.msio import msdata_from_arrays, samples_from_msdata
    m = snap.meta
    path = str(path or m.get("path") or f"{stem or snap.stem}.ms")
    ms = msdata_from_arrays(path, snap.antenna1, snap.antenna2, snap.uvw,
                            snap.data, snap.flag, snap.weight, snap.time, np.array(m["freqs"], np.float64),
                            np.array(m["chan_width"], np.float64), [float(x) for x in m["phase_dir"]],
                            np.array(m["ant_pos"], np.float64), telescope=m.get("telescope", "OVRO-LWA"),
                            observer=m.get("observer", ""), field_name=m.get("field_name", ""), want_v=want_v)
    if R is not None and ra is not None:
        c = samples_from_msdata(ms, None, None, want_v, compact=True)
        c.update(R=np.asarray(R, np.float64), pd_new=(ra, dec))
    else:
        c = samples_from_msdata(ms, ra, dec, want_v, compact=True)
    rec = record_from_compact(c, peel)
    if stem:
        rec.meta["stem"] = str(stem)
    for k in ("freqs", "chan_width", "ant_pos", "a1", "a2", "R"):     # never a view of the slot
        rec.arrays[k] = np.array(rec.arrays[k], copy=True)
    return rec


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


def expected_path(archive: PathLike) -> Path:
    """Per-batch list of the snapshots the worker archived (FITS in place), written next to the
    archive file: the hour job's reference for what the archive must contain."""
    a = str(archive)
    return Path((a[:-len(".gsiv")] if a.endswith(".gsiv") else a) + ".expected.jsonl")


def add_expected(archive: PathLike, stem: str, **info) -> None:
    p = expected_path(archive)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(dict(stem=stem, host=os.uname().nodename, t=time.time(), **info)) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def expected_stems(cfg, band: int, group: str) -> Optional[set]:
    """Stems the workers archived for the band-hour (all batches), None if no list exists."""
    d = archive_dir(cfg, band, group)
    files = sorted(d.glob("*.expected.jsonl")) if d.is_dir() else []
    if not files:
        return None
    out = set()
    for f in files:
        for ln in f.read_text().splitlines():
            try:
                out.add(json.loads(ln)["stem"])
            except (ValueError, KeyError):
                continue                      # a torn last line: that snapshot is not counted
    return out


def archive_inventory(files: Sequence[PathLike], band: Optional[int] = None,
                      phase_centre: Optional[str] = None, verify: bool = False) -> Dict[str, Any]:
    """What the hour job would load from the files: stems (newest record per stem over all
    files) and every problem that makes the archive unusable as it is: torn bytes, dropped index
    lines, records without an index line, another band / phase centre / channelisation than
    expected, and with ``verify`` a bad crc (reads the files; the imager checks again)."""
    problems: List[str] = []
    per_file = []
    hdr0 = None
    for f in files:
        try:
            sc = scan(f, verify=verify)
        except GSIVError as e:
            problems.append(str(e))
            continue
        problems += sc.problems()
        hm = sc.header_meta
        if band is not None and hm.get("band") is not None and int(hm["band"]) != int(band):
            problems.append(f"{f}: band {hm.get('band')} != {band}")
        if phase_centre is not None and hm.get("phase_centre") != phase_centre:
            problems.append(f"{f}: phase centre {hm.get('phase_centre')!r} != {phase_centre!r}")
        if hdr0 is None:
            hdr0 = sc
        else:
            for k in CHECK_ARRAYS:
                if k in hdr0.header_arrays and not _same_array(sc.header_arrays.get(k), hdr0.header_arrays[k]):
                    problems.append(f"{f}: {k} differ from {hdr0.path}")
        per_file.append((str(f), [(dict(stem=e["stem"], written_ns=e.get("written_ns", 0)), None, e)
                                  for e in sc.entries]))
    best, superseded = newest(per_file)
    return dict(stems=sorted(best), problems=problems, superseded=superseded,
                files={st: p for st, (p, _) in best.items()})


def archive_stems(files: Sequence[PathLike]) -> List[str]:
    """Stems the hour job would load from the files (newest record per stem); raises GSIVError
    if the archive has any defect (archive_inventory)."""
    inv = archive_inventory(files)
    if inv["problems"]:
        raise GSIVError("; ".join(inv["problems"]))
    return inv["stems"]


PROBE = ("import ttcalx_imager, ttcalx_imager.gsiv as g, ttcalx_imager.msio as m; "
         "assert hasattr(m, 'samples_from_msdata') and hasattr(g, 'load_archive'); print(ttcalx_imager.__file__)")


def probe_imager(python: str, env: Optional[Dict[str, str]] = None, timeout: float = 180.0) -> str:
    """Check that ``python`` (with ``env``) imports an imager with gsiv / samples_from_msdata;
    returns its ttcalx_imager path, raises GSIVError with the reason otherwise."""
    try:
        p = subprocess.run([python, "-c", PROBE], capture_output=True, text=True, env=env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GSIVError(f"archive writer python {python}: {e}") from None
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["?"]
        raise GSIVError(f"archive writer python {python} has no usable ttcalx_imager.gsiv "
                        f"(PYTHONPATH={(env or {}).get('PYTHONPATH', '')!r}): {tail[0]}")
    return p.stdout.strip().splitlines()[-1]


# ----------------------------------------------------------------------------- worker hook
class ArchiveClient:
    """Worker side of the archive writer: a ``hourstore serve`` process in the imager venv
    (same msio / casacore / numpy as the hour job), one per archive file.

    Requests carry an id and every reply is matched to its request.  A timeout, a dead child or
    a broken pipe kills the child and marks the client dead (``alive`` False): every later call
    raises, the caller starts a new client (which resumes the file)."""

    def __init__(self, python: str, out: PathLike, phase_centre: Optional[str], header: Dict[str, Any],
                 env: Optional[Dict[str, str]] = None, log_path: Optional[PathLike] = None,
                 want_v: bool = True, timeout: float = 600.0, start_timeout: float = 300.0,
                 cmd: Optional[List[str]] = None):
        self.out = Path(out)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._buf = b""
        self._id = 0
        self.alive = False
        if cmd is None:
            cmd = [python, "-m", "gpu_subband_imaging.v3.hourstore", "serve", "--out", str(out),
                   "--header", json.dumps(header)]
            if phase_centre:
                cmd += ["--phase-centre", phase_centre]
            if not want_v:
                cmd.append("--no-v")
        self._log = open(log_path, "a") if log_path else None      # None: the worker's stderr
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                  env=env, bufsize=0)
        self.alive = True
        try:
            r = self._reply(0, start_timeout)
        except GSIVError:
            self._kill()
            raise
        if not r.get("ok"):
            self._kill()
            raise GSIVError(f"archive writer did not start: {r.get('error', r)}")
        self.info = r

    def _readline(self, deadline: float) -> bytes:
        fd = self.p.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise GSIVError("archive writer: no reply in time")
            r, _, _ = select.select([fd], [], [], left)
            if r:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    raise GSIVError(f"archive writer exited (rc={self.p.poll()})")
                self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def _reply(self, rid: int, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            line = self._readline(deadline)
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("id") == rid:
                return r
            # a reply to an earlier request (cannot happen after a restart; never taken as ours)

    def _kill(self) -> None:
        self.alive = False
        try:
            if self.p.poll() is None:
                self.p.kill()
            self.p.wait(30)
        except Exception:
            pass

    def add(self, ms_path: PathLike, stem: Optional[str] = None, path: Optional[str] = None,
            peel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append the record of ``ms_path``, named ``stem`` / ``path`` (see records_from_ms)."""
        with self._lock:
            if not self.alive or self.p.poll() is not None:
                self._kill()
                raise GSIVError(f"archive writer for {self.out} is not running (rc={self.p.returncode})")
            self._id += 1
            rid = self._id
            try:
                self.p.stdin.write((json.dumps(dict(id=rid, op="add", ms=str(ms_path), stem=stem, path=path,
                                                    peel=peel)) + "\n").encode())
                self.p.stdin.flush()
                r = self._reply(rid, self.timeout)
            except (GSIVError, OSError) as e:
                self._kill()                       # no stale reply can reach the next request
                raise GSIVError(f"archive record {ms_path}: {e}; writer killed") from None
        if not r.get("ok"):
            raise GSIVError(f"archive record {ms_path}: {r.get('error')}")
        return r

    def close(self, timeout: float = 60.0) -> None:
        with self._lock:
            try:
                if self.alive and self.p.poll() is None:
                    self.p.stdin.write((json.dumps(dict(id=-1, op="close")) + "\n").encode())
                    self.p.stdin.flush()
                    self.p.wait(timeout)
            except Exception:
                pass
            self._kill()
            if self._log is not None:
                self._log.close()
                self._log = None


def _serve(a) -> int:
    # the protocol owns the original stdout; anything a library prints goes to stderr (the log)
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(r):
        out.write(json.dumps(r) + "\n"); out.flush()

    w: Optional[RecordWriter] = None
    try:
        import ttcalx_imager
        from ttcalx_imager import gsiv  # noqa: F401  (the hour job must be able to read what we write)
        from ttcalx_imager.msio import samples_from_msdata  # noqa: F401
        from ttcalx_imager.rotate import parse_phase_centre
        ra = dec = None
        if a.phase_centre:
            ra, dec = parse_phase_centre(a.phase_centre)
        header = json.loads(a.header) if a.header else {}
        header.setdefault("phase_centre", a.phase_centre)
        header["pd_new"] = None if ra is None else [ra, dec]
        header["imager"] = ttcalx_imager.__file__
        if Path(a.out).exists() and Path(a.out).stat().st_size > 0:
            w = RecordWriter(a.out, header, resume=True, fsync=not a.no_fsync,
                             log=lambda m: print(m, file=sys.stderr, flush=True))
    except Exception as e:
        reply(dict(id=0, ok=False, error=repr(e)))
        return 1
    reply(dict(id=0, ok=True, n=len(w) if w else 0, pid=os.getpid(), imager=header["imager"]))
    test_sleep = os.environ.get("GSIV_SERVE_TEST_SLEEP", "")      # tests: "<n>:<seconds>"
    n_add = 0
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        rid = None
        try:
            q = json.loads(line)
            rid = q.get("id")
            if q.get("op") == "close":
                break
            n_add += 1
            if test_sleep and int(test_sleep.split(":")[0]) == n_add:
                time.sleep(float(test_sleep.split(":")[1]))
            t0 = time.perf_counter()
            rec = records_from_ms(q["ms"], ra, dec, want_v=not a.no_v, peel=q.get("peel"), stem=q.get("stem"),
                                  path=q.get("path"))
            t1 = time.perf_counter()
            if w is None:
                w = RecordWriter(a.out, header, {k: rec.arrays[k] for k in SHARED_ARRAYS + CHECK_ARRAYS},
                                 fsync=not a.no_fsync)
            superseded = rec.stem in w.stems           # a reprocessed snapshot: the new record wins
            e = w.append(rec.meta, rec.arrays)
            r = dict(id=rid, ok=True, stem=rec.stem, seq=e["seq"], nbytes=e["nbytes"], n_kept=e["n_kept"],
                     superseded=superseded, build_s=t1 - t0, write_s=time.perf_counter() - t1)
        except Exception as e:      # report and keep serving
            r = dict(id=rid, ok=False, error=repr(e))
        reply(r)
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
                w = RecordWriter(a.out, hm, {k: rec.arrays[k] for k in SHARED_ARRAYS + CHECK_ARRAYS})
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
