"""Copy one measurement set to worker scratch, extracting archives."""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


def stage(input_lustre: str, work_dir: Path) -> Path:
    """Return a scratch copy of an archived or unpacked measurement set."""
    work_dir.mkdir(parents=True, exist_ok=True)
    source = Path(input_lustre)
    if source.is_dir() and source.name.endswith(".ms"):
        destination = work_dir / source.name
        shutil.copytree(source, destination)
        return destination

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
            import logging
            logging.getLogger("gsi.stage_ms").warning("STAGE_RETRY %s: %s", source.name, e)
            shutil.rmtree(work_dir / local_tar.name[:-4], ignore_errors=True)
    local_tar.unlink()
    ms = work_dir / local_tar.name[:-4]
    if not ms.is_dir():
        raise FileNotFoundError(f"expected {ms} after untar of {input_lustre}")
    return ms
