"""Where the gsi v3 hour store keeps the band-hour's records: RAM or a spill file (design 5.2).

``plan_store`` decides at batch start: ``v3.store: ram | spill | auto``.  auto keeps the
records in RAM when ``need = n_todo * record bytes * 1.15`` fits into MemAvailable minus
``v3.ram_reserve_gb`` minus what the other gsi v3 processes on the node reserved, else it
spills: the records are appended to a GSIV file in the first ``v3.spill_dirs`` entry with
``need * 1.2`` free and memory-mapped from there (``HourStore`` memmap backend).

Reservations: each batch writes ``/dev/shm/gsi3_mem_<uid>_<pid>.json`` with the bytes it keeps
in RAM; ``reserved_by_others`` sums the files of live processes (stale ones are ignored), so
two GPU processes on one node do not both count the same free RAM.

``MemoryGuard.check`` is called during the batch (every ``every`` snapshots): MemAvailable below
the reserve -> spill now (logged V3_MEM_SPILL by the caller).  numpy only; Linux /proc/meminfo
(other systems: MemAvailable unknown -> RAM, the guard never fires).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

GB = 1e9
RECORD_BYTES = 34.5e6          # typical compact record (2.03 M kept samples); design 1.2
RECORD_BYTES_HIGH = 21e6       # bands >= 82 MHz (85-100 MHz flagged)
NEED_FACTOR = 1.15
SPILL_FREE_FACTOR = 1.2
RES_PREFIX = "gsi3_mem_"


def mem_available(meminfo: str = "/proc/meminfo") -> Optional[float]:
    """MemAvailable in bytes, None when unknown."""
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


def peak_rss() -> float:
    """Peak RSS of this process in bytes (ru_maxrss, kB on Linux)."""
    import resource
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0


def est_record_bytes(band: int) -> float:
    return RECORD_BYTES_HIGH if int(band) >= 82 else RECORD_BYTES


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reservation_path(root: str = "/dev/shm", pid: Optional[int] = None) -> Path:
    return Path(root) / f"{RES_PREFIX}{os.getuid()}_{pid or os.getpid()}.json"


def reserve(nbytes: float, root: str = "/dev/shm", **info) -> Optional[Path]:
    """Write this process's RAM reservation (None if the directory is not writable)."""
    p = reservation_path(root)
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(bytes=float(nbytes), pid=os.getpid(), **info)))
        os.replace(tmp, p)
        return p
    except OSError:
        return None


def unreserve(root: str = "/dev/shm") -> None:
    try:
        reservation_path(root).unlink()
    except OSError:
        pass


def reserved_by_others(root: str = "/dev/shm") -> float:
    """Bytes reserved by other live gsi v3 processes (any user) on this node."""
    tot = 0.0
    try:
        names = os.listdir(root)
    except OSError:
        return 0.0
    for n in names:
        if not (n.startswith(RES_PREFIX) and n.endswith(".json")):
            continue
        try:
            pid = int(n[len(RES_PREFIX):-len(".json")].split("_")[-1])
        except ValueError:
            continue
        if pid == os.getpid() or not _pid_alive(pid):
            continue
        try:
            tot += float(json.loads(Path(root, n).read_text()).get("bytes", 0.0))
        except (OSError, ValueError):
            continue
    return tot


@dataclass
class StorePlan:
    backend: str                      # ram | memmap
    need_bytes: float
    avail_bytes: Optional[float]      # MemAvailable - reserve - others (None: unknown)
    spill_dir: Optional[str] = None   # where the GSIV file goes (spill, or the local checkpoint)
    reason: str = ""
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(backend=self.backend, need_GB=self.need_bytes / GB,
                    avail_GB=None if self.avail_bytes is None else self.avail_bytes / GB,
                    spill_dir=self.spill_dir, reason=self.reason, notes=list(self.notes))


def first_spill_dir(dirs: Sequence[str], need: float) -> Optional[str]:
    """First entry of ``dirs`` that exists or can be made and has need x 1.2 free.  Never Lustre."""
    for d in dirs:
        if not d or Path(d).parts[:2] == ("/", "lustre"):
            continue
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(d).free >= need * SPILL_FREE_FACTOR:
                return d
        except OSError:
            continue
    return None


def plan_store(mode: str, n_todo: int, band: int, reserve_gb: float, spill_dirs: Sequence[str],
               record_bytes: Optional[float] = None, avail: Optional[float] = None,
               others: Optional[float] = None, want_file: bool = False) -> StorePlan:
    """RAM or spill for ``n_todo`` records (``mode`` = v3.store).  ``want_file``: a local
    checkpoint is wanted anyway (v3.checkpoint: local), so a spill dir is chosen in RAM mode too.
    Raises RuntimeError when a spill is needed and no spill dir has room."""
    rb = float(record_bytes or est_record_bytes(band))
    need = max(0, int(n_todo)) * rb * NEED_FACTOR
    if avail is None:
        avail = mem_available()
    others = reserved_by_others() if others is None else others
    free = None if avail is None else avail - reserve_gb * GB - others
    notes = []
    if mode == "ram":
        backend, why = "ram", "v3.store=ram"
    elif mode == "spill":
        backend, why = "memmap", "v3.store=spill"
    elif free is None:
        backend, why = "ram", "MemAvailable unknown"
    elif need <= free:
        backend, why = "ram", f"need {need / GB:.1f} GB <= {free / GB:.1f} GB free of the reserve"
    else:
        backend, why = "memmap", f"need {need / GB:.1f} GB > {free / GB:.1f} GB free of the reserve"
    d = None
    if backend == "memmap" or want_file:
        d = first_spill_dir(spill_dirs, need)
        if d is None:
            if backend == "memmap":
                raise RuntimeError(f"v3.store: spill needed ({why}) but no v3.spill_dirs entry {list(spill_dirs)} "
                                   f"has {need * SPILL_FREE_FACTOR / GB:.1f} GB free")
            notes.append(f"no spill dir with {need * SPILL_FREE_FACTOR / GB:.1f} GB free: no local checkpoint")
    return StorePlan(backend, need, free, d, why, notes)


@dataclass
class MemoryGuard:
    """Mid-batch check (design 5.2.2): MemAvailable below the reserve -> spill."""
    reserve_gb: float
    every: int = 10
    n: int = 0
    fired: bool = False

    def check(self, avail: Optional[float] = None) -> bool:
        """True exactly once, when the store should spill now."""
        self.n += 1
        if self.fired or self.n % max(1, self.every):
            return False
        avail = mem_available() if avail is None else avail
        if avail is not None and avail < self.reserve_gb * GB:
            self.fired = True
            return True
        return False
