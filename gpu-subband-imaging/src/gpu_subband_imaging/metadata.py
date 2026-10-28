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


def _run(cmd, timeout: int = 30, strip: bool = True) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip() if strip else out


def git_info(path: str) -> Dict[str, object]:
    """Commit, branch, origin URL, whether a remote branch has the commit, and uncommitted changes."""
    p = str(path)
    commit = _run(["git", "-C", p, "rev-parse", "HEAD"])
    if not commit:
        return {"path": p, "commit": "unknown"}
    # porcelain lines are "XY path"; stripping would eat the leading space of " M path"
    status = _run(["git", "-C", p, "status", "--porcelain", "--untracked-files=no"], strip=False)
    return {
        "path": p,
        "commit": commit,
        "branch": _run(["git", "-C", p, "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["git", "-C", p, "describe", "--tags", "--always", "--dirty"]),
        "origin": _run(["git", "-C", p, "remote", "get-url", "origin"]),
        "on_remote": bool(_run(["git", "-C", p, "branch", "-r", "--contains", commit])),
        "uncommitted_files": [ln[3:] for ln in status.splitlines() if ln.strip()],
    }


def _code_info() -> Dict[str, object]:
    """This package's source: a git checkout, or a copy carrying a .gsi-version marker."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return git_info(str(parent))
        marker = parent / ".gsi-version"
        if marker.is_file():
            return {"path": str(parent), "commit": marker.read_text().strip(), "copy_of_git": True}
    return {"path": str(Path(__file__).resolve().parents[2]), "commit": "unknown"}


def _md5(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


_IMAGER_PROBE = r"""
import importlib, json, os
out = {}
for m in ("ttcalx_imager", "cupy", "cufinufft", "numpy", "astropy", "casacore"):
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "?")
        if m == "ttcalx_imager":
            out["ttcalx_imager_src"] = os.path.dirname(os.path.realpath(mod.__file__))
    except Exception as e:
        out[m] = "unavailable: " + type(e).__name__
print(json.dumps(out))
"""

_SOFTWARE: Dict[str, Dict] = {}


def software_versions(cfg: Config) -> Dict[str, object]:
    """Everything that produced a product: this pipeline, the TTCalX peeling tree, the GPU
    imager install (and the source tree it imports), Julia, and the sky / RFI model files."""
    key = cfg.pipeline.hour_imaging.ttcalx_image
    if key in _SOFTWARE:
        return _SOFTWARE[key]
    env = cfg.pipeline.env
    imager_python = cfg.pipeline.imaging.gpu_python or str(Path(key).parent / "python")
    probe_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    try:
        imager = json.loads(subprocess.run([imager_python, "-c", _IMAGER_PROBE], capture_output=True,
                                           text=True, timeout=120, env=probe_env).stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        imager = {}
    src = imager.get("ttcalx_imager_src")
    if src:
        imager["source_git"] = git_info(src)
    imager["python"] = imager_python
    out = {
        "gpu_subband_imaging": {**_code_info(), "package_version": __version__},
        "ttcalx_peeling": git_info(env.ttcalx),
        "imager": imager,
        "julia": _run([f"{env.julia_bin}/julia", "--version"]),
        "sky_model": {"path": cfg.pipeline.cal.sources, "md5": _md5(cfg.pipeline.cal.sources)},
        "rfi_model": {"path": cfg.pipeline.cal.rfi_sources, "md5": _md5(cfg.pipeline.cal.rfi_sources)},
    }
    _SOFTWARE[key] = out
    return out


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
    name = "run.json" if cfg.pipeline.layout != "lst" else f"run_{cfg.pipeline.run_key}.json"
    out = Path(cfg.pipeline.output_root) / "_metadata" / cfg.pipeline.date / name
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
        "software": software_versions(cfg),
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
