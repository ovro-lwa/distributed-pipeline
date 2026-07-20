"""List shared MS archives through a worker and split them into batches."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from .config import PipelineConfig
from .ledger import Job

_SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]
log = logging.getLogger("gsi.worklist")


def _list_tars(node: str, band: int, band_dir: str,
               hours: Union[str, List[str]]) -> List[str]:
    pats = ([f"{band_dir}/*/*.ms.tar"] if hours == "all"
            else [f"{band_dir}/{h}/*.ms.tar" for h in hours])
    cmd = "ls -1 " + " ".join(pats) + " 2>/dev/null | sort"
    log.info("listing %sMHz inputs via %s from %s", band, node, band_dir)
    p = subprocess.run(_SSH + [node, cmd], capture_output=True, text=True)
    tars = [ln for ln in p.stdout.splitlines() if ln.strip()]
    log.info("found %d .ms.tar files for %sMHz", len(tars), band)
    return tars


def _chunk(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


@dataclass
class Batch:
    band: int
    index: int
    tars: List[str]


def build(pl: PipelineConfig, list_via_node: str, manifest_dir: Path
          ) -> List[Batch]:
    """Build batches and write their manifests to a shared directory."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    batches: List[Batch] = []
    for band in pl.bands:
        band_dir = f"{pl.data_root}/{band}MHz/{pl.date}"
        log.info("planning band %sMHz from %s", band, band_dir)
        tars = _list_tars(list_via_node, band, band_dir, pl.hours)
        if not tars:
            log.warning("no .ms.tar files found for %sMHz under %s", band, band_dir)
            continue
        for i, chunk in enumerate(_chunk(tars, pl.batch_size)):
            mf = manifest_dir / f"{band}_{i}.txt"
            mf.write_text("\n".join(chunk) + "\n")
            log.info("band %sMHz batch %d -> %d files (manifest %s)",
                     band, i, len(chunk), mf)
            batches.append(Batch(band, i, chunk))
    return batches


def to_jobs(batches: List[Batch]) -> List[Job]:
    return [Job(b.band, b.index, len(b.tars)) for b in batches]
