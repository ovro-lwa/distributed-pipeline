"""Snapshot imaging on the worker's GPU with ttcalx_imager (imaging.imager: ttcalx).

One persistent workers/ttcalx_snapshot_daemon.py per worker batch (the imager venv's
python); the archive threads share it under a lock.  Writes the same
<prefix>-I-image.fits / <prefix>-V-image.fits as stages/image.py's WSClean runs
(CLEAN I with orca SNAPSHOT_CLEAN_I_PARAMS, dirty V with the same weighting).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ..config import ImagingGeom, ImagingParams

log = logging.getLogger("gsi.gpu_image")
_PREFIX = "GSI_SNAP "


class SnapshotDaemon:
    def __init__(self, python: str, daemon_py: str, env: Mapping[str, str]):
        self._cmd = [python, daemon_py]
        self._env = dict(env)
        self._lock = threading.Lock()
        self._p: Optional[subprocess.Popen] = None

    def _start(self) -> None:
        self._p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, env=self._env, bufsize=1)
        if self._reply().get("status") != "ready":
            raise RuntimeError("GPU snapshot imager did not start")

    def _reply(self) -> dict:
        assert self._p is not None and self._p.stdout is not None
        for line in self._p.stdout:
            if line.startswith(_PREFIX):
                return json.loads(line[len(_PREFIX):])
        raise RuntimeError(f"GPU snapshot imager exited (rc={self._p.poll()})")

    def image(self, ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
              timings: Optional[Dict[str, float]] = None, retry: bool = True) -> List[Path]:
        """Image one MS; return [I image, V image] like stages.image.image."""
        req = dict(ms=str(ms), prefix=str(out_prefix), pixels=geom.pixels, scale=geom.scale,
                   niter=p.clean_niter)
        with self._lock:
            if self._p is None or self._p.poll() is not None:
                self._start()
            assert self._p is not None and self._p.stdin is not None
            try:
                self._p.stdin.write(json.dumps(req) + "\n")
                self._p.stdin.flush()
                r = self._reply()
            except (BrokenPipeError, RuntimeError):
                self._p = None          # restart for the next snapshot
                raise
        if r.get("status") == "error" and "OutOfMemory" in r.get("error", "") and retry:
            # the GPU is shared with the zest daemon, whose pool may peak at the same moment
            log.warning("GPU snapshot imaging of %s ran out of memory; retrying once", ms)
            time.sleep(2.0)
            return self.image(ms, out_prefix, geom, p, timings, retry=False)
        if r.get("status") != "ok":
            raise RuntimeError(f"GPU snapshot imaging of {ms} failed: {r.get('error', r.get('status'))}")
        if timings is not None:
            timings["image_gpu"] = float(r["timing"].get("TOTAL", 0.0))
        return [Path(f"{out_prefix}-I-image.fits"), Path(f"{out_prefix}-V-image.fits")]

    def close(self) -> None:
        with self._lock:
            if self._p is not None:
                try:
                    if self._p.stdin:
                        self._p.stdin.close()
                    self._p.wait(timeout=60)
                except Exception:
                    self._p.kill()
                self._p = None


def start(python: str, base_env: Optional[Mapping[str, str]] = None,
          extra_env: Optional[Mapping[str, str]] = None) -> SnapshotDaemon:
    env = dict(os.environ if base_env is None else base_env)
    for key in ("PYTHONPATH", "PYTHONHOME"):           # the venv, not the conda runtime
        env.pop(key, None)
    env.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.update(extra_env or {})
    daemon_py = str(Path(os.environ.get("GSI_WORKERS", Path(__file__).resolve().parents[3] / "workers"))
                    / "ttcalx_snapshot_daemon.py")
    return SnapshotDaemon(python, daemon_py, env)
