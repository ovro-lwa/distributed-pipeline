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
    shutil.copy(input_lustre, local_tar)
    with tarfile.open(local_tar) as t:
        t.extractall(work_dir)
    local_tar.unlink()
    ms = work_dir / local_tar.name[:-4]
    if not ms.is_dir():
        raise FileNotFoundError(f"expected {ms} after untar of {input_lustre}")
    return ms
