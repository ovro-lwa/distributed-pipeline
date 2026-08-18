"""Write one provenance record for each observing date."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from . import __version__
from .config import Config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_revision() -> str:
    env = os.environ.get("GSI_CODE_VERSION")
    if env:
        return env
    for parent in Path(__file__).resolve().parents:
        marker = parent / ".gsi-version"
        if marker.is_file():
            return marker.read_text().strip()
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            try:
                return subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                break
    return "unknown"


def _config_hashes(config_dir: Path) -> Dict[str, str]:
    hashes = {}
    for name in ("cluster.yaml", "subbands.yaml", "pipeline.yaml"):
        path = config_dir / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_run_metadata(cfg: Config, config_dir: str, status: str,
                       summary: Optional[Dict[str, int]] = None,
                       error: Optional[str] = None) -> Path:
    """Atomically create or update this date's provenance record."""
    out = (Path(cfg.pipeline.output_root) / "_metadata" /
           cfg.pipeline.date / "run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if out.is_file():
        try:
            previous = json.loads(out.read_text())
        except (OSError, ValueError):
            previous = {}

    now = _utc_now()
    data = {
        "schema_version": 1,
        "date": cfg.pipeline.date,
        "status": status,
        "started_utc": previous.get("started_utc", now),
        "updated_utc": now,
        "finished_utc": now if status != "running" else None,
        "code": {
            "revision": _code_revision(),
            "package_version": __version__,
            "python": platform.python_version(),
        },
        "host": socket.gethostname(),
        "config_snapshot": str(Path(config_dir).resolve()),
        "config_sha256": _config_hashes(Path(config_dir)),
        "pipeline": asdict(cfg.pipeline),
        "cluster": {
            "slots": [str(slot) for slot in cfg.cluster.slots()],
            "max_feeders": cfg.cluster.max_feeders,
            "wsclean_threads": cfg.cluster.wsclean_threads,
            "aoflagger_threads": cfg.cluster.aoflagger_threads,
            "free_core_budget": cfg.cluster.free_core_budget,
            "work_root": cfg.cluster.work_root,
            "state_dir": cfg.cluster.state_dir,
        },
        "ledger": summary or {},
        "error": error,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    return out
