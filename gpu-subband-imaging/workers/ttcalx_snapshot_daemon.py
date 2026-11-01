"""Persistent GPU snapshot imager for one worker (run with the ttcalx_imager venv python).

Reads one JSON request per stdin line, {"ms", "prefix", "pixels", "scale", "niter"[, "id"]},
images the MS with ttcalx_imager.SnapshotImager (orca SNAPSHOT_CLEAN_I: CLEAN I + dirty V, as
stages/image.py's WSClean runs) into <prefix>-I-image.fits and <prefix>-V-image.fits, and
answers with one stdout line "GSI_SNAP <json>" per request, in request order (the reply
carries the request's "id" if it had one).

Pipelined (SnapshotImager.image_many): requests may be sent before the previous reply; while
the GPU images one snapshot, a helper process reads the next MS and the previous one's FITS are
written by a host thread.  One imager per process keeps the cuFINUFFT plans and the w-term table
of the band's geometry (snapshot_svd keeps exactly one table and rebuilds it only when the
snapshot w range changes, so nothing has to be cleared per snapshot).
GSI_SNAP_PIPELINE=0 restores the one-at-a-time loop (each request answered before the next is read).
"""
import json
import math
import os
import queue
import sys
import threading
import traceback

import cupy as cp
from ttcalx_imager import bands, gridder, snapload, snapshot
from ttcalx_imager.snapshot import SnapshotImager

PREFIX = "GSI_SNAP "
_out = threading.Lock()


def reply(d):
    with _out:
        print(PREFIX + json.dumps(d, default=float), flush=True)


def reuse_svd(quantum, pool):
    """snapshot.snapshot_svd with wmax rounded up to ``quantum`` lambda and one cached table."""
    def svd(N, sc, wf, tol):
        wq = math.ceil((float(wf) * 1.0001 + 1e-6) / quantum) * quantum
        key = (N, round(sc, 12), round(wq, 3), tol, 600)
        if key not in gridder._SVD_CACHE:
            gridder._SVD_CACHE.clear()
            pool.free_all_blocks()
            gridder._SVD_CACHE[key] = gridder.SvdWTerm(N, sc, wq, tol, nw=600)
        return gridder._SVD_CACHE[key]
    return svd


def to_request(req):
    params = bands.SNAPSHOT_CLEAN_I.with_(niter=int(req.get("niter", 50000)))
    return dict(ms_path=req["ms"], N=bands.wsclean_size(int(req["pixels"])), scale_deg=float(req["scale"]),
                out_prefix=req["prefix"], pols="IV", params=params, _id=req.get("id"))


def answer(info, rid):
    if info.get("status") == "error":
        out = dict(status="error", error=info.get("error"))
        if info.get("traceback"):
            print(info["traceback"], file=sys.stderr, flush=True)
    else:
        out = dict(status=info["status"], files=info.get("files", []), timing=info["timing"],
                   niter=info.get("niter"), gpu_pool_GB=info.get("gpu_pool_GB"))
    if rid is not None:
        out["id"] = rid
    reply(out)


def main() -> int:
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9))
    quantum = float(os.environ.get("GSI_SVD_QUANTUM", "0"))
    if quantum > 0:
        snapshot.snapshot_svd = reuse_svd(quantum, pool)
    n = snapload.sweep_stale_shm()          # blocks of a previous daemon that was killed
    if n:
        print(f"removed {n} stale shared-memory blocks", file=sys.stderr, flush=True)
    im = SnapshotImager()
    pipeline = os.environ.get("GSI_SNAP_PIPELINE", "1") != "0"
    reply({"status": "ready", "pipeline": pipeline})
    if not pipeline:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                rq = to_request(json.loads(line)); rid = rq.pop("_id")
                info = im.image(**rq)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                info, rid = dict(status="error", error=repr(e)), None
            pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            answer(info, rid)
        return 0

    q = queue.Queue()

    def stdin_reader():
        for line in sys.stdin:
            if not line.strip():
                continue
            rid = None
            try:
                d = json.loads(line)
                rid = d.get("id") if isinstance(d, dict) else None
                q.put(to_request(d))
            except Exception as e:           # malformed request: answer it at once (with its id)
                out = dict(status="error", error=repr(e))
                if rid is not None:
                    out["id"] = rid
                reply(out)
        q.put(None)
    threading.Thread(target=stdin_reader, daemon=True).start()

    def done(info):
        rq = info.pop("request", None) or {}
        answer(info, info.pop("_id", rq.get("_id")))
    for _ in im.image_many(q, on_done=done, free_pool=True):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
