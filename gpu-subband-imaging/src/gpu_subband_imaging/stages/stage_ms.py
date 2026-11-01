"""Copy one measurement set to worker scratch, extracting archives."""
from __future__ import annotations

import errno
import logging
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Optional

log = logging.getLogger("gsi.stage_ms")

# tarfile stream read size and member copy size.  The stream buffer must stay small
# against the copy size: tarfile slices its leftover buffer on every read, so a 4 MiB
# buffer with the default 16 KiB copies costs 1.4 s CPU per 400 MB tar (0.14 s here).
STREAM_BUFSIZE = 1 << 16
COPY_BUFSIZE = 1 << 20


def stage(input_lustre: str, work_dir: Path, untar_to: Optional[Path] = None) -> Path:
    """Return a scratch copy of an archived or unpacked measurement set.

    ``untar_to`` (gsi v3 step 1): stream the tar straight from Lustre into that
    directory (e.g. /dev/shm/gsi3/<stamp>) instead of copying the tar into
    ``work_dir`` and extracting it there.  ``work_dir`` is still created (the
    averaged MS goes there).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    source = Path(input_lustre)
    if source.is_dir() and source.name.endswith(".ms"):
        destination = work_dir / source.name
        shutil.copytree(source, destination)
        return destination
    if untar_to is not None:
        return stream_untar(source, Path(untar_to))

    local_tar = work_dir / source.name
    # test copy: copies from Lustre to /tmp were seen short or unreadable under load: retry
    for attempt in range(3):
        shutil.copy(input_lustre, local_tar)
        try:
            if local_tar.stat().st_size != source.stat().st_size:
                raise tarfile.ReadError("short copy")
            with tarfile.open(local_tar) as t:
                t.extractall(work_dir)
            break
        except tarfile.ReadError as e:
            if attempt == 2:
                raise
            log.warning("STAGE_RETRY %s: %s", source.name, e)
            shutil.rmtree(work_dir / local_tar.name[:-4], ignore_errors=True)
    local_tar.unlink()
    ms = work_dir / local_tar.name[:-4]
    if not ms.is_dir():
        raise FileNotFoundError(f"expected {ms} after untar of {input_lustre}")
    return ms


def _check_member(m: tarfile.TarInfo, ms_name: str) -> None:
    p = PurePosixPath(m.name)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != ms_name:
        raise tarfile.ReadError(f"unexpected member {m.name!r} (want {ms_name}/...)")
    if not (m.isdir() or m.isfile()):
        raise tarfile.ReadError(f"unsupported member type {m.name!r}")


def _extract_stream(source: Path, dest: Path, ms_name: str) -> int:
    """Extract ``source`` (read sequentially, never seeking) into ``dest``; return files.

    Sanity checks: every member lies under ``<ms_name>/`` and is a file or directory,
    every extracted file has its header size, the members end with tar's end-of-archive
    marker (two zero blocks; tarfile stops without an error at a cut or garbage after a
    member), and the whole tar file was read.
    """
    size = source.stat().st_size
    n_files = 0
    with open(source, "rb", buffering=0) as raw:
        with tarfile.open(fileobj=raw, mode="r|*", bufsize=STREAM_BUFSIZE) as t:
            t.copybufsize = COPY_BUFSIZE
            for m in t:
                _check_member(m, ms_name)
                t.extract(m, str(dest))
                if m.isfile():
                    got = os.stat(dest / m.name).st_size
                    if got != m.size:
                        raise tarfile.ReadError(f"{m.name}: extracted {got} B, header {m.size} B")
                    n_files += 1
            end = t.offset                  # where the next header would start
        while raw.read(COPY_BUFSIZE):       # end-of-archive blocks / record padding
            pass
        nread = raw.tell()
        marker = os.pread(raw.fileno(), 2 * tarfile.BLOCKSIZE, end)
    if nread != size:
        raise tarfile.ReadError(f"read {nread} B of {size} B")
    if len(marker) != 2 * tarfile.BLOCKSIZE or marker.count(0) != len(marker):
        raise tarfile.ReadError(f"no end-of-archive marker at byte {end} of {size}")
    if not n_files:
        raise tarfile.ReadError("no files in archive")
    return n_files


# a full tmpfs or a permission problem does not go away on a retry
NO_RETRY_ERRNOS = (errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EPERM, errno.EROFS)


def _remove(p: Path) -> None:
    if p.is_symlink():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p, ignore_errors=True)


def stream_untar(source: Path, dest: Path) -> Path:
    """Stream-extract ``<stamp>.ms.tar`` into ``dest``; return ``dest/<stamp>.ms``.

    No local copy of the tar is made.  Three attempts, as on the copy-then-extract path.
    """
    name = source.name
    if not name.endswith(".tar"):
        raise ValueError(f"not a tar archive: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"no such tar: {source}")
    ms_name = name[:-4]
    ms = dest / ms_name
    for p in (dest, ms):                    # never follow a planted link
        if p.is_symlink():
            raise FileExistsError(f"refusing to extract through symlink {p}")
    for attempt in range(3):
        _remove(ms)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            _extract_stream(source, dest, ms_name)
            break
        except (tarfile.TarError, OSError) as e:
            _remove(ms)
            if attempt == 2 or getattr(e, "errno", None) in NO_RETRY_ERRNOS:
                raise
            log.warning("STAGE_RETRY %s: %s", name, e)
    if not (ms / "table.dat").is_file():
        _remove(ms)
        raise FileNotFoundError(f"expected {ms}/table.dat after untar of {source}")
    return ms
