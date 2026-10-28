"""Persistent GPU snapshot imager for one worker (run with the ttcalx_imager venv python).

Reads one JSON request per stdin line, {"ms", "prefix", "pixels", "scale", "niter"}, images
the MS with ttcalx_imager.SnapshotImager (orca SNAPSHOT_CLEAN_I: CLEAN I + dirty V, as
stages/image.py's WSClean runs) into <prefix>-I-image.fits and <prefix>-V-image.fits, and
answers with one stdout line "GSI_SNAP <json>".  One imager per process keeps the cuFINUFFT
plans and w-term tables of the band's geometry.
"""
import json
import os
import sys
import traceback

import cupy as cp
from ttcalx_imager import bands, gridder
from ttcalx_imager.snapshot import SnapshotImager

PREFIX = "GSI_SNAP "


def main() -> int:
    # The GPU is shared with the zest daemon (and its restarts): cap CuPy's pool and hand
    # cached blocks back after every snapshot.
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9))
    im = SnapshotImager()
    print(PREFIX + json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        # gridder caches one w-term SVD table per snapshot w-range (0.25 lambda steps):
        # keep the newest few, or a long-lived process fills the GPU
        while len(gridder._SVD_CACHE) > 4:
            gridder._SVD_CACHE.pop(next(iter(gridder._SVD_CACHE)))
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


if __name__ == "__main__":
    sys.exit(main())
