#!/usr/bin/env python
"""Pre-warm the CuPy kernel cache of a GPU node for the gsi v3 engine's imager (no real data).

CuPy compiles each new kernel source once per GPU architecture and keeps it in CUPY_CACHE_DIR.  A new imager
release (v5: the zenith-frame snapshot gridder, the hour-stage v5 switches) brings new kernels, and without this
the first snapshot of every batch compiles them (~36 s, ~220 kernels in the v5 e2e test).  This script runs, on
synthetic visibilities at each band-geometry tier (bands.PIXEL_TIERS: the snapshot image size and scale of the
engine's request), the engine's snapshot code path (rotate.rephase on the GPU, snapshot.gpu_select with the
stored-frame UVW, SnapshotImager.compute) with

  * the zenith path (TTCALX_SNAP_ZEN=force) and the production path (TTCALX_SNAP_ZEN=0),
  * DFT major cycles and degridding major cycles (dft_max 0: every major cycle degrids),

and one small hour stage (cli.run_hour_snaps, as the engine's hour op) per tier, all with the engine's settings
(use_fast_settings with TTCALX_V5=1, unless --no-v5).  Images are not written; the products of a later run do
not depend on the cache (the same kernels, compiled once).

Run it with the imager venv's python on the node, with the CUPY_CACHE_DIR of the deploy's config
(hour_imaging.env), for each GPU model of the node:

    CUPY_CACHE_DIR=/fast/pipeline/gsi_v5/cupy_cache CUDA_VISIBLE_DEVICES=0 \\
        /opt/devel/pipeline/gsi-v5/venv/bin/python deploy/warm_cupy_cache.py

(setup_node.sh does this with GSI_WARM_CUPY=1).  It prints the number of kernel files added to the cache; a
second run should add none.
"""
import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np

C = 299792458.0
OVRO_LAT = math.radians(37.2397808)


def _count(d):
    n = 0
    for _, _, fs in os.walk(d):
        n += len(fs)
    return n


def _array(rng, nant):
    """A compact core plus outriggers (OVRO-LWA-like extent, m), nearly coplanar."""
    core = rng.normal(0, 60, (nant - nant // 8, 3))
    r = rng.uniform(250, 1300, nant // 8); a = rng.uniform(0, 2 * math.pi, nant // 8)
    out = np.stack([r * np.cos(a), r * np.sin(a), rng.normal(0, 2, nant // 8)], 1)
    ant = np.concatenate([core, out]); ant[:, 2] *= 0.02
    return ant


def _rot(ax, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if ax == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def synthetic_snapshot(band, rng, nant=96, nchan=8, tilt=6.0):
    """The engine's 'small' dict + raw columns of a snapshot: stored (zenith-frame) UVW, hour UVW = R stored,
    DATA with a few point sources predicted in the stored frame and re-phased to the hour frame (the same
    rotate.rephase as the engine's prepare)."""
    from ttcalx_imager import rotate
    ant = _array(rng, nant)
    a1, a2 = (np.array(x, np.int32) for x in zip(*[(i, j) for i in range(nant) for j in range(i, nant)]))
    uvw_s = ant[a1] - ant[a2]
    R = _rot(0, tilt) @ _rot(1, 0.6 * tilt)
    uvw_h = uvw_s @ R.T
    f = band * 1e6 + (np.arange(nchan) - nchan / 2) * 24e3
    lam = C / f
    us, vs, ws = (uvw_s[:, i:i + 1] / lam[None] for i in range(3))
    vis = np.zeros(us.shape, np.complex128)
    for l, m, s in ((0.1, 0.2, 100.0), (-0.45, 0.3, 40.0), (0.6, -0.5, 20.0), (0.05, -0.9, 30.0)):
        n = math.sqrt(1 - l * l - m * m)
        vis += s * np.exp(-2j * np.pi * (us * l + vs * m + ws * (n - 1)))
    vis += 0.5 * (rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape))
    data = np.zeros(vis.shape + (4,), np.complex64)
    data[:, :, 0] = vis; data[:, :, 3] = vis
    data[:, :, 1] = 0.02 * vis; data[:, :, 2] = -0.02 * vis
    flag = np.zeros(data.shape, bool)
    flag[rng.random(len(a1)) < 0.05] = True
    flag[a1 == a2] = True
    small = dict(a1=a1, a2=a2, uvw=uvw_h, uvw_stored=uvw_s, wcol=np.ones((len(a1), 4), np.float32), time=5.3e9,
                 tstart=5.3e9, tmid=5.3e9, freqs=f, chan_width=np.full(f.size, 24e3), phase_dir=(1.0, OVRO_LAT),
                 telescope="OVRO-LWA", observer="", field_name="", path="warm.ms", stem="warm", column="DATA")
    return small, data, uvw_s, uvw_h, flag, ant


def warm_snapshot(band, rng, log):
    import cupy as cp
    from ttcalx_imager import bands, rotate, snapshot
    from ttcalx_imager.snapshot import SnapshotImager
    N, sc = bands.geometry(band)
    pool = cp.get_default_memory_pool()
    for mode in ("force", "0"):
        os.environ["TTCALX_SNAP_ZEN"] = mode
        for dft_max in (8000, 0):
            small, data, uvw_s, uvw_h, flag, _ = synthetic_snapshot(band, rng)
            t0 = time.perf_counter()
            d = rotate.rephase(cp.asarray(data), uvw_s[:, 2], uvw_h[:, 2], small["freqs"], xp=cp)   # engine prepare
            L = snapshot.gpu_select(small, {"DATA": d, "FLAG": flag}, sc, "IV")
            del d
            im = SnapshotImager(dft_max=dft_max)
            info, prods = im.compute(L, N, sc, os.path.join(tempfile.gettempdir(), "gsi_warm"), "IV",
                                     bands.SNAPSHOT_CLEAN_I.with_(niter=2000))
            del L, prods
            pool.free_all_blocks()
            path = "zenith" if info.get("zen") else "production"
            log(f"  {band} MHz snapshot N={N} TTCALX_SNAP_ZEN={mode} dft_max={dft_max}: {path} path, "
                f"niter {info.get('niter')}, {time.perf_counter() - t0:.1f} s")


def warm_hour(band, rng, log, nsnap=6):
    """A small band-hour through cli.run_hour_snaps (the engine's hour op) at the band's geometry: a deep I + V
    product and a 10-min product, few snapshots, low niter."""
    import cupy as cp
    from ttcalx_imager import cli, gridder
    from ttcalx_imager.rotate import analytic_rotation, parse_phase_centre
    ant = _array(rng, 64)
    b = (ant[:, None] - ant[None])[np.triu_indices(64, 1)]
    freqs = np.array([band * 1e6]); lam = C / freqs[0]
    out = tempfile.mkdtemp(prefix="gsi_warm_hour_")
    argv = ["--band", str(band), "--out", os.path.join(out, f"{band}MHz"), "--products",
            "I-Deep-Taper-Robust-0,V-Taper-Deep,I-Taper-10min", "--lst-hour", "5", "--qa", "none", "--nproc", "2"]

    def get_snaps(ra_h, dec, want_v):
        ras = ra_h + (np.arange(nsnap) - nsnap // 2) * math.radians(2.5)
        snaps = []
        for i in range(nsnap):
            u, v, w = (b[:, k] / lam for k in range(3))
            R = analytic_rotation(ras[i], dec, ra_h, dec)
            uh, vh, wh = R @ np.stack([u, v, w])
            vis = 10 * np.exp(2j * np.pi * (uh * 0.1 + vh * -0.05)) * np.exp(-2j * np.pi * (wh - w))
            vis = vis + 0.05 * (rng.normal(size=u.size) + 1j * rng.normal(size=u.size))
            s = dict(path=f"s{i}", stem=f"20261024_10{i:02d}03_{band}MHz", time=5.3e9 + 600 * i,
                     tstart=5.3e9 - 5 + 600 * i, freqs=freqs, chan_width=24e3, R=R, u=u.astype(np.float32),
                     v=v.astype(np.float32), w=w.astype(np.float32), nw=np.ones(u.size, np.float32),
                     I=vis.astype(np.complex64), flagfrac=0.0, zero_frac=0.0, telescope="OVRO-LWA", observer="",
                     field_name="")
            if want_v:
                s["V"] = (0.1 * vis).astype(np.complex64)
            snaps.append(s)
        return snaps

    t0 = time.perf_counter()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        cli.run_hour_snaps(argv, get_snaps)
    gridder.clear_caches()
    cp.get_default_memory_pool().free_all_blocks()
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    log(f"  {band} MHz hour stage: {time.perf_counter() - t0:.1f} s")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bands", default=None,
                   help="comma-separated bands (default: one per geometry tier of bands.PIXEL_TIERS)")
    p.add_argument("--no-hour", action="store_true", help="snapshot kernels only")
    p.add_argument("--no-v5", action="store_true", help="without TTCALX_V5=1 (the v4 imager settings)")
    a = p.parse_args(argv)
    cache = os.environ.get("CUPY_CACHE_DIR")
    if not cache:
        print("warm_cupy_cache: set CUPY_CACHE_DIR (the deploy config's hour_imaging.env value)", file=sys.stderr)
        return 2
    os.makedirs(cache, exist_ok=True)
    if not a.no_v5:
        os.environ.setdefault("TTCALX_V5", "1")
    os.environ.setdefault("GSI_SVD_QUANTUM", "0")              # as the v5 template's imaging.gpu_env
    import ttcalx_imager
    ttcalx_imager.use_fast_settings()                          # as the engine's GpuBackend
    import cupy as cp
    from ttcalx_imager import bands
    cp.get_default_memory_pool().set_limit(size=int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9))
    tiers = [int(x) for x in a.bands.split(",")] if a.bands else [lo for lo, _, _, _ in bands.PIXEL_TIERS]
    log = lambda m: print(m, flush=True)
    n0 = _count(cache)
    t0 = time.perf_counter()
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    log(f"warm_cupy_cache: {ttcalx_imager.__file__} ({getattr(ttcalx_imager, '__version__', '?')}), "
        f"GPU {dev['name'].decode()} sm_{dev['major']}{dev['minor']}, cache {cache} ({n0} files), bands {tiers}, "
        f"TTCALX_V5={os.environ.get('TTCALX_V5', '0')}")
    rng = np.random.default_rng(1)
    for band in tiers:
        warm_snapshot(band, rng, log)
        if not a.no_hour:
            cp.get_default_memory_pool().set_limit(size=0)     # the engine's before_hour
            warm_hour(band, rng, log)
            cp.get_default_memory_pool().set_limit(size=int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9))
    n1 = _count(cache)
    log(f"warm_cupy_cache: {n1 - n0} kernel file(s) added to {cache} ({n1} now) in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
