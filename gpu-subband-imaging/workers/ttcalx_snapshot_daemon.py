"""Persistent GPU snapshot imager for one worker (run with the ttcalx_imager venv python).

Reads one JSON request per stdin line, {"ms", "prefix", "pixels", "scale", "niter"}, images
the MS with ttcalx_imager.SnapshotImager (orca SNAPSHOT_CLEAN_I: CLEAN I + dirty V, as
stages/image.py's WSClean runs) into <prefix>-I-image.fits and <prefix>-V-image.fits, and
answers with one stdout line "GSI_SNAP <json>".  One imager per process keeps the cuFINUFFT
plans and w-term tables of the band's geometry.

Default: one request at a time, exactly as 13405ce (the per-snapshot w-term cache clear, no new
imager modules).  GSI_SNAP_PIPELINE=1 (opt-in; needs ttcalx_imager >= 86afd89: snapload,
SnapshotImager.image_many, the one-table snapshot_svd): pipelined -- requests carry an "id" and may
be sent before the previous reply; while the GPU images one snapshot, a helper process reads the
next MS and the previous one's FITS are written by a host thread; replies come in request order
with the request's "id"; the ready line says {"pipeline": true}.  stages/gpu_image.py uses the
matching client mode from the same variable.
"""
import json
import math
import os
import sys
import traceback

import cupy as cp
from ttcalx_imager import bands, gridder, snapshot
from ttcalx_imager.snapshot import SnapshotImager

PREFIX = "GSI_SNAP "


def reuse_svd(quantum, pool):
    """snapshot.snapshot_svd with wmax rounded up to ``quantum`` lambda and one cached table.

    The snapshot wmax drifts by ~0.25 lambda per 10 s at 82 MHz, so the stock 0.25-lambda key builds
    a new table (1.5 s at 82 MHz) for nearly every snapshot.  A 2-lambda key reuses one for ~8
    snapshots; the table still meets the SVD tolerance over the larger w range (images change by
    ~0.01 sigma rms).  The previous table is dropped before a new one is built, so the GPU never
    holds two."""
    def svd(N, sc, wf, tol):
        wq = math.ceil((float(wf) * 1.0001 + 1e-6) / quantum) * quantum
        key = (N, round(sc, 12), round(wq, 3), tol, 600)
        if key not in gridder._SVD_CACHE:
            gridder._SVD_CACHE.clear()
            pool.free_all_blocks()
            gridder._SVD_CACHE[key] = gridder.SvdWTerm(N, sc, wq, tol, nw=600)
        return gridder._SVD_CACHE[key]
    return svd


def pipeline_on() -> bool:
    return os.environ.get("GSI_SNAP_PIPELINE", "0") == "1"


def main() -> int:
    # The GPU is shared with the zest daemon (and its restarts): cap CuPy's pool and hand
    # cached blocks back after every snapshot.
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9))
    quantum = float(os.environ.get("GSI_SVD_QUANTUM", "0"))
    if quantum > 0:
        snapshot.snapshot_svd = reuse_svd(quantum, pool)
    if pipeline_on():
        return main_pipelined(pool)
    im = SnapshotImager()
    print(PREFIX + json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if quantum <= 0:
            # gridder caches one w-term SVD table per snapshot w-range (0.25 lambda steps; several
            # GB at 82 MHz) and snapshots rarely share one: drop it, or old + new tables overflow
            gridder._SVD_CACHE.clear()
            pool.free_all_blocks()
        try:
            params = bands.SNAPSHOT_CLEAN_I.with_(niter=int(req.get("niter", 50000)))
            info = im.image(req["ms"], N=bands.wsclean_size(int(req["pixels"])),
                            scale_deg=float(req["scale"]), out_prefix=req["prefix"], pols="IV",
                            params=params)
            out = dict(status=info["status"], files=info.get("files", []), timing=info["timing"],
                       niter=info.get("niter"), gpu_pool_GB=info.get("gpu_pool_GB"))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            out = dict(status="error", error=repr(e))
        pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print(PREFIX + json.dumps(out, default=float), flush=True)
    return 0


# -- GSI_SNAP_PIPELINE=1 ---------------------------------------------------------------------------

def main_pipelined(pool) -> int:
    """SnapshotImager.image_many: the w-term table is kept by snapshot_svd (one table, rebuilt only when
    the snapshot w range changes), so nothing is cleared per snapshot."""
    import queue
    import threading
    from ttcalx_imager import snapload

    out_lock = threading.Lock()

    def reply(d):
        with out_lock:
            print(PREFIX + json.dumps(d, default=float), flush=True)

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
        out["id"] = rid
        reply(out)

    n = snapload.sweep_stale_shm()          # blocks of a previous daemon that was killed
    if n:
        print(f"removed {n} stale shared-memory blocks", file=sys.stderr, flush=True)
    im = SnapshotImager()
    reply({"status": "ready", "pipeline": True})
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
            except Exception as e:           # malformed request: answered at once, with its id if it had one
                reply(dict(status="error", error=repr(e), id=rid))
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
