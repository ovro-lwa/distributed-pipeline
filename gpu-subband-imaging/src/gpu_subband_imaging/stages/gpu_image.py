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


def _timeout(env: Mapping[str, str], default: Optional[float]) -> Optional[float]:
    v = env.get("GSI_SNAP_REPLY_TIMEOUT_S", "")
    return float(v) if v.strip() else default


class SnapshotDaemon:
    """Client of workers/ttcalx_snapshot_daemon.py.

    Default (as 13405ce): one request at a time under the lock, the archive threads take turns; no
    reply deadline unless GSI_SNAP_REPLY_TIMEOUT_S is set, and then a daemon that misses it is killed
    (the request fails, the next one starts a new daemon), so no late reply can be read as another
    request's.

    GSI_SNAP_PIPELINE=1 in the daemon environment (opt-in, imaging.gpu_env): pipelined.  Each archive
    thread's request is sent at once, tagged with an id, and a reader thread hands the daemon's
    in-order replies to their requests by id, so with archive_workers >= 2 the daemon reads the next MS
    while the GPU images the current one.  The daemon must say {"pipeline": true} when ready; a reply
    without an id, or no reply within GSI_SNAP_REPLY_TIMEOUT_S (default 900 s in this mode), kills it
    and fails every outstanding request (they are never matched to another request).
    """

    def __init__(self, python: str, daemon_py: str, env: Mapping[str, str]):
        self._cmd = [python, daemon_py]
        self._env = dict(env)
        self._lock = threading.Lock()          # process start / stdin writes / pending table
        self._p: Optional[subprocess.Popen] = None
        self.pipelined = self._env.get("GSI_SNAP_PIPELINE", "0") == "1"
        self.reply_timeout = _timeout(self._env, 900.0 if self.pipelined else None)
        self._pending: "Dict[int, list]" = {}  # pipelined: id -> [Event, reply, daemon Popen]
        self._ids = 0

    def _start(self) -> None:
        self._p = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, env=self._env, bufsize=1)
        r = self._reply_line(self._p)
        if r.get("status") != "ready":
            raise RuntimeError("GPU snapshot imager did not start")
        if self.pipelined:
            if r.get("pipeline") is not True:
                self._kill(self._p)
                self._p = None
                raise RuntimeError("GSI_SNAP_PIPELINE=1 but the GPU snapshot imager is not pipelined "
                                   "(an older workers/ttcalx_snapshot_daemon.py?)")
            threading.Thread(target=self._dispatch, args=(self._p,), daemon=True).start()

    def _reply(self) -> dict:
        assert self._p is not None
        return self._reply_line(self._p)

    @staticmethod
    def _reply_line(p) -> dict:
        assert p.stdout is not None
        for line in p.stdout:
            if line.startswith(_PREFIX):
                return json.loads(line[len(_PREFIX):])
        raise RuntimeError(f"GPU snapshot imager exited (rc={p.poll()})")

    @staticmethod
    def _kill(p) -> None:
        try:
            p.kill()
            p.wait(timeout=60)
        except Exception:
            pass

    def image(self, ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams,
              timings: Optional[Dict[str, float]] = None, retry: bool = True) -> List[Path]:
        """Image one MS; return [I image, V image] like stages.image.image."""
        r = self._request_pipelined(ms, out_prefix, geom, p) if self.pipelined else \
            self._request(ms, out_prefix, geom, p)
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

    # -- default: one at a time (13405ce) --------------------------------------------------------

    def _request(self, ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams) -> dict:
        req = dict(ms=str(ms), prefix=str(out_prefix), pixels=geom.pixels, scale=geom.scale,
                   niter=p.clean_niter)
        with self._lock:
            if self._p is None or self._p.poll() is not None:
                self._start()
            assert self._p is not None and self._p.stdin is not None
            proc, timer, fired = self._p, None, []
            if self.reply_timeout is not None:
                def expire():
                    fired.append(1)
                    log.error("GPU snapshot imaging of %s: no reply within %.0f s; killing the imager "
                              "(pid %d)", ms, self.reply_timeout, proc.pid)
                    self._kill(proc)
                timer = threading.Timer(self.reply_timeout, expire)
                timer.daemon = True
                timer.start()
            try:
                self._p.stdin.write(json.dumps(req) + "\n")
                self._p.stdin.flush()
                r = self._reply()
            except (BrokenPipeError, RuntimeError):
                self._p = None          # restart for the next snapshot
                if fired:
                    raise RuntimeError(f"GPU snapshot imaging of {ms}: no reply within "
                                       f"{self.reply_timeout:.0f} s (imager killed)") from None
                raise
            finally:
                if timer is not None:
                    timer.cancel()
        return r

    # -- GSI_SNAP_PIPELINE=1 ---------------------------------------------------------------------

    def _fail_all(self, p, why: str) -> None:
        """Daemon p is gone or untrusted: kill it, fail every outstanding request (caller holds the lock)."""
        if self._p is p:
            self._p = None
        self._kill(p)
        for rid in [k for k, ev in self._pending.items() if ev[2] is p]:
            ev = self._pending.pop(rid)
            ev[1] = dict(status="error", error=f"daemon died: {why}")
            ev[0].set()

    def _dispatch(self, p) -> None:
        while True:
            try:
                r = self._reply_line(p)
            except Exception as e:             # daemon died: fail everything outstanding
                with self._lock:
                    self._fail_all(p, str(e))
                return
            with self._lock:
                rid = r.get("id")
                if rid is None or rid not in self._pending or self._pending[rid][2] is not p:
                    # never guessed: an id-less or unknown reply means the stream cannot be trusted
                    log.error("GPU snapshot imager reply without a pending id (%s): killing it", str(r)[:200])
                    self._fail_all(p, f"reply without a pending id: {str(r)[:200]}")
                    return
                ev = self._pending.pop(rid)
                ev[1] = r
                ev[0].set()

    def _request_pipelined(self, ms: Path, out_prefix: Path, geom: ImagingGeom, p: ImagingParams) -> dict:
        with self._lock:
            if self._p is None or self._p.poll() is not None:
                self._start()
            assert self._p is not None and self._p.stdin is not None
            proc = self._p
            ev = [threading.Event(), None, proc]     # [done, reply, the daemon it was sent to]
            self._ids += 1
            rid = self._ids
            req = dict(ms=str(ms), prefix=str(out_prefix), pixels=geom.pixels, scale=geom.scale,
                       niter=p.clean_niter, id=rid)
            self._pending[rid] = ev
            try:
                self._p.stdin.write(json.dumps(req) + "\n")
                self._p.stdin.flush()
            except BrokenPipeError:
                self._fail_all(proc, "stdin closed")
                raise
        ok = ev[0].wait(timeout=self.reply_timeout)
        if not ok:
            with self._lock:
                if not ev[0].is_set():      # kill: a late reply must never be read as another request's
                    log.error("GPU snapshot imaging of %s: no reply within %.0f s; killing the imager (pid %d)",
                              ms, self.reply_timeout, proc.pid)
                    self._fail_all(proc, f"no reply within {self.reply_timeout:.0f} s")
            raise RuntimeError(f"GPU snapshot imaging of {ms}: no reply within {self.reply_timeout:.0f} s "
                               f"(imager killed)")
        r = ev[1]
        if r.get("status") == "error" and "daemon died" in r.get("error", ""):
            raise RuntimeError(f"GPU snapshot imager exited: {r['error']}")
        return r

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
