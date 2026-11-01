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
REPLY_TIMEOUT_S = float(os.environ.get("GSI_SNAP_REPLY_TIMEOUT_S", "900"))   # a snapshot takes ~1 s


class SnapshotDaemon:
    """Client of workers/ttcalx_snapshot_daemon.py.  Thread-safe and pipelined: each archive
    thread's request is sent at once (tagged with an id) and the daemon's in-order replies are
    dispatched by a reader thread, so with archive_workers >= 2 the daemon reads the next MS
    while the GPU images the current one.  (An old daemon without ids answers one at a time:
    replies without "id" are matched in send order, which is the same thing.)"""

    def __init__(self, python: str, daemon_py: str, env: Mapping[str, str]):
        self._cmd = [python, daemon_py]
        self._env = dict(env)
        self._lock = threading.Lock()          # process start / stdin writes / pending table
        self._p: Optional[subprocess.Popen] = None
        self._pending: "Dict[int, list]" = {}  # id -> [Event, reply]
        self._order: List[int] = []
        self._ids = 0

    def _start(self) -> None:
        self._p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, env=self._env, bufsize=1)
        if self._reply_line(self._p).get("status") != "ready":
            raise RuntimeError("GPU snapshot imager did not start")
        threading.Thread(target=self._dispatch, args=(self._p,), daemon=True).start()

    @staticmethod
    def _reply_line(p) -> dict:
        assert p.stdout is not None
        for line in p.stdout:
            if line.startswith(_PREFIX):
                return json.loads(line[len(_PREFIX):])
        raise RuntimeError(f"GPU snapshot imager exited (rc={p.poll()})")

    def _dispatch(self, p) -> None:
        while True:
            try:
                r = self._reply_line(p)
            except Exception as e:             # daemon died: fail everything outstanding
                with self._lock:
                    if self._p is p:
                        self._p = None
                    for rid in self._order:
                        ev = self._pending[rid]; ev[1] = dict(status="error", error=f"daemon died: {e}")
                        ev[0].set()
                    self._order.clear()
                return
            with self._lock:
                rid = r.get("id", self._order[0] if self._order else None)
                if rid in self._pending:
                    self._order.remove(rid)
                    ev = self._pending[rid]; ev[1] = r; ev[0].set()

    def image(self, ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
              timings: Optional[Dict[str, float]] = None, retry: bool = True) -> List[Path]:
        """Image one MS; return [I image, V image] like stages.image.image."""
        ev = [threading.Event(), None]
        with self._lock:
            if self._p is None or self._p.poll() is not None:
                self._start()
            assert self._p is not None and self._p.stdin is not None
            self._ids += 1; rid = self._ids
            req = dict(ms=str(ms), prefix=str(out_prefix), pixels=geom.pixels, scale=geom.scale,
                       niter=p.clean_niter, id=rid)
            self._pending[rid] = ev; self._order.append(rid)
            try:
                self._p.stdin.write(json.dumps(req) + "\n")
                self._p.stdin.flush()
            except BrokenPipeError:
                self._pending.pop(rid, None); self._order.remove(rid)
                self._p = None          # restart for the next snapshot
                raise
        ok = ev[0].wait(timeout=REPLY_TIMEOUT_S)
        with self._lock:
            self._pending.pop(rid, None)
            if not ok and rid in self._order:
                self._order.remove(rid)
        if not ok:
            raise RuntimeError(f"GPU snapshot imaging of {ms}: no reply within {REPLY_TIMEOUT_S:.0f} s")
        r = ev[1]
        if r.get("status") == "error" and "daemon died" in r.get("error", ""):
            raise RuntimeError(f"GPU snapshot imager exited: {r['error']}")
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
            p, self._p = self._p, None
        if p is not None:
            try:
                if p.stdin:
                    p.stdin.close()
                p.wait(timeout=60)
            except Exception:
                p.kill()


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
