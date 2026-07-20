"""Copy and extract one measurement-set archive to worker scratch."""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


def stage(tar_lustre: str, work_dir: Path) -> Path:
    """Return the extracted measurement-set path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local_tar = work_dir / Path(tar_lustre).name
    shutil.copy(tar_lustre, local_tar)
    with tarfile.open(local_tar) as t:
        t.extractall(work_dir)
    local_tar.unlink()
    ms = work_dir / local_tar.name[:-4]
    if not ms.is_dir():
        raise FileNotFoundError(f"expected {ms} after untar of {tar_lustre}")
    return ms
