"""gsi v3 engine: snapshot imaging from a shm slot + the band-hour store + the hour products,
on the worker's GPU (docs/v3_design.md 3.1, 3.3, 3.4; steps 4 and 6).

Runs in the ttcalx_imager venv (CuPy, cuFINUFFT, python-casacore) with the pipeline's src on
PYTHONPATH (v3.slots, v3.hourstore).  A superset of ttcalx_snapshot_daemon.py: one JSON request
per stdin line, one reply line "GSI_V3 <json>" per request on stdout, every reply carries the
request's "id".  The protocol owns the original stdout; everything a library prints goes to
stderr (the worker log).

Ops
  store_open  {path|null, backend ram|memmap, header{}}   open (resume) the hour store: records
              already in the GSIV file are loaded (checkpoint restart) -> {stems, n, nbytes}
  image_slot  {slot, prefix, pixels, scale, niter, phase_centre, record_phase_centre, image, record,
               stem, path, peel}
              image the PEELED slot at the LST-hour phase centre in memory, exactly as v2's
              chgcentre copy + MS read (rotate.chgcentre_uvw, snapshot.rephase, gpu_select,
              SnapshotImager.compute / write: <prefix>-I-image.fits, -V-image.fits, -qa.json)
              and/or turn it into the compact hour record (hourstore.records_from_snap: the
              record load_samples gives for the kept MS).  The reply comes once nothing reads
              the slot any more (FITS written, record stored): the caller may then reuse it.
  spill       {path}             RAM store -> memory-mapped GSIV file (V3_MEM_SPILL)
  hour        {argv, expected, log}   the band-hour products from the store's records
              (ttcalx_imager.cli.run_hour_snaps: the same as ``ttcalx-image hour <argv>`` on the
              kept MSes / archive), the newest record of each expected stem; -> summary
  info / close

The GPU thread is the main loop (requests in order).  Records are built in a store thread and
FITS written in a writer thread while the GPU images the next snapshot.  casacore measures calls
(the rotations) stay on the main thread.

Backends: "gpu" (default) or "fake" (GSI_V3_ENGINE_BACKEND=fake, tests: no CuPy; writes tiny
FITS images, the records are real).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

PREFIX = "GSI_V3 "
FATAL_MARKERS = ("cudaErrorIllegalAddress", "cudaErrorLaunchFailure", "CUDA_ERROR_ILLEGAL_ADDRESS",
                 "cudaErrorMisaligned", "CUDA_ERROR_LAUNCH_FAILED", "an illegal memory access")


class Out:
    """The protocol stream (a dup of the original stdout); stdout itself goes to stderr."""

    def __init__(self):
        fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self._f = os.fdopen(fd, "w")
        self._lock = threading.Lock()

    def reply(self, d):
        with self._lock:
            self._f.write(PREFIX + json.dumps(d, default=_jsonable) + "\n")
            self._f.flush()


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ----------------------------------------------------------------------------- rotation
def rotation(snap, image_centre, record_centre):
    """The rotations of a slot snapshot, None if neither centre is given:

    uvw_new, pd_image  (image_centre) the UVW of every row as chgcentre writes them and the PHASE_DIR
                       it writes: chgcentre's own (ra, dec) parse and per-antenna UVW at TIME
                       (bit-identical PHASE_DIR, UVW within 2e-7 m: scripts/v3_engine/chgcentre_check.py);
    R, pd_record       (record_centre) the hour record's R: rotate.snapshot_rotation with
                       rotate.parse_phase_centre -- the call load_samples / the v2 hour job make."""
    if not image_centre and not record_centre:
        return None
    from ttcalx_imager import rotate
    a1, a2 = np.asarray(snap.antenna1), np.asarray(snap.antenna2)
    pos = np.asarray(snap.meta["ant_pos"], np.float64)
    t0 = float(snap.time[0])
    out = dict(uvw_new=None, pd_image=None, R=None, pd_record=None)
    if image_centre:
        ra_c, dec_c = rotate.parse_phase_centre_chgcentre(image_centre)
        out.update(uvw_new=rotate.chgcentre_uvw(pos, a1, a2, t0, ra_c, dec_c), pd_image=(ra_c, dec_c))
    if record_centre:
        keep = a1 != a2
        ra_h, dec_h = rotate.parse_phase_centre(record_centre)
        R = rotate.snapshot_rotation(np.asarray(snap.uvw)[keep].astype(np.float64), a1[keep], a2[keep], pos, t0,
                                     ra_h, dec_h)
        out.update(R=np.asarray(R, np.float64), pd_record=(ra_h, dec_h))
    return out


def small_of(snap, req, rot):
    """The RawLoader 'small' dict of a slot snapshot (snapload._raw_to_shm's values), rotated."""
    m = snap.meta
    t = np.asarray(snap.time)
    rotated = rot is not None and rot["uvw_new"] is not None
    pd = rot["pd_image"] if rotated else (float(m["phase_dir"][0]), float(m["phase_dir"][1]))
    return dict(a1=np.asarray(snap.antenna1), a2=np.asarray(snap.antenna2),
                uvw=rot["uvw_new"] if rotated else np.asarray(snap.uvw, np.float64), wcol=np.asarray(snap.weight),
                time=float(t[0]), tstart=float(t.min()), tmid=float(t.mean()),
                freqs=np.array(m["freqs"], np.float64), chan_width=np.array(m["chan_width"], np.float64),
                phase_dir=(float(pd[0]), float(pd[1])), telescope=str(m.get("telescope") or "OVRO-LWA"),
                observer=str(m.get("observer") or ""), field_name=str(m.get("field_name") or ""),
                path=str(req.get("path") or req.get("slot")), stem=str(req.get("stem") or snap.stem), column="DATA")


def prepare(snap, req, rot, xp=np):
    """(small, DATA) of a slot snapshot as the imager would read them from v2's chgcentre'd MS copy:
    the rotated UVW of every row in ``small`` and DATA re-phased with rotate.rephase (on ``xp``:
    CuPy in the GPU backend, numpy in the fake one -- the same code)."""
    from ttcalx_imager import rotate
    small = small_of(snap, req, rot)
    d = xp.asarray(snap.data)
    if rot is not None and rot["uvw_new"] is not None:
        d = rotate.rephase(d, np.asarray(snap.uvw)[:, 2], rot["uvw_new"][:, 2], small["freqs"], xp=xp)
    return small, d


# ----------------------------------------------------------------------------- backends
class GpuBackend:
    name = "gpu"

    def __init__(self):
        import cupy as cp
        from ttcalx_imager import gridder, snapshot
        from ttcalx_imager.snapshot import SnapshotImager
        self.cp, self.snapshot, self.gridder = cp, snapshot, gridder
        self.pool = cp.get_default_memory_pool()
        self.limit = int(float(os.environ.get("GSI_GPU_IMAGER_MEM_GB", "10")) * 1e9)
        self.pool.set_limit(size=self.limit)
        quantum = float(os.environ.get("GSI_SVD_QUANTUM", "0"))
        if quantum > 0:
            snapshot.snapshot_svd = _reuse_svd(quantum, self.pool, gridder)
        self.im = SnapshotImager()
        import ttcalx_imager
        self.imager = ttcalx_imager.__file__

    def image(self, snap, req, rot):
        """GPU part: (info, host products, t0)."""
        from ttcalx_imager import bands
        cp, snapshot = self.cp, self.snapshot
        t0 = time.perf_counter()
        T = {}
        t1 = time.perf_counter()
        small, d = prepare(snap, req, rot, xp=cp)
        cp.cuda.Device().synchronize()
        T["H2D + rephase"] = time.perf_counter() - t1
        N, sc = bands.wsclean_size(int(req["pixels"])), float(req["scale"])
        L = snapshot.gpu_select(small, {"DATA": d, "FLAG": snap.flag}, sc, "IV")
        del d
        L["timing"].update(T)
        params = bands.SNAPSHOT_CLEAN_I.with_(niter=int(req.get("niter", 50000)))
        try:
            info, prods = self.im.compute(L, N, sc, req["prefix"], "IV", params)
        finally:
            del L
            self.pool.free_all_blocks()
        return info, prods, t0

    def write(self, info, prods, t0):
        return self.im.write(info, prods, t0)

    def free(self):
        self.pool.free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()

    def before_hour(self):
        """The snapshot imager's caches go (cuFINUFFT plans, w-term tables / factors / geometry, cuFFT
        plans); the hour stage gets the whole GPU (no pool limit), as v2's separate hour job.  At 82 MHz
        the cached snapshot plans left cuFINUFFT too little memory for the hour's setpts."""
        import gc
        self.gridder.clear_caches()
        self.cp.fft.config.get_plan_cache().clear()
        gc.collect()
        self.free()
        self.pool.set_limit(size=0)

    def after_hour(self):
        self.cp.fft.config.get_plan_cache().clear()
        self.free()
        self.pool.set_limit(size=self.limit)

    def gpu_info(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        return dict(gpu_free_GB=free / 1e9, gpu_total_GB=total / 1e9, pool_GB=self.pool.total_bytes() / 1e9)


def _reuse_svd(quantum, pool, gridder):
    """ttcalx_snapshot_daemon.reuse_svd: snapshot_svd with wmax rounded up to ``quantum``."""
    def svd(N, sc, wf, tol):
        wq = math.ceil((float(wf) * 1.0001 + 1e-6) / quantum) * quantum
        key = (N, round(sc, 12), round(wq, 3), tol, 600)
        if key not in gridder._SVD_CACHE:
            gridder._SVD_CACHE.clear()
            pool.free_all_blocks()
            gridder._SVD_CACHE[key] = gridder.SvdWTerm(N, sc, wq, tol, nw=600)
        return gridder._SVD_CACHE[key]
    return svd


class FakeBackend:
    """Tests: no GPU.  'Images' the selected Stokes I / V sums into 8 x 8 FITS files.  Test hooks
    by environment: GSI_V3_FAKE_SLEEP (s per image), GSI_V3_FAKE_FAIL (stem: error reply),
    GSI_V3_FAKE_DIE (stem: the process exits), GSI_V3_FAKE_FATAL (stem: a fatal CUDA-like error)."""
    name = "fake"
    imager = "fake"

    def image(self, snap, req, rot):
        t0 = time.perf_counter()
        stem = str(req.get("stem") or snap.stem)
        time.sleep(float(os.environ.get("GSI_V3_FAKE_SLEEP", "0")))
        if os.environ.get("GSI_V3_FAKE_DIE") == stem:
            os._exit(17)
        if os.environ.get("GSI_V3_FAKE_FAIL") == stem:
            raise RuntimeError(f"fake failure for {stem}")
        if os.environ.get("GSI_V3_FAKE_FATAL") == stem:
            raise RuntimeError("CUDARuntimeError: cudaErrorIllegalAddress: an illegal memory access")
        small, d = prepare(snap, req, rot, xp=np)
        if os.environ.get("GSI_V3_FAKE_DUMP"):            # tests: what the GPU imager would get
            np.savez(f"{req['prefix']}-prep.npz", uvw=small["uvw"], data=d, phase_dir=np.array(small["phase_dir"]),
                     a1=small["a1"], a2=small["a2"])
        f = np.asarray(snap.flag)
        ok = ~(f[:, :, 0] | f[:, :, 3]) & (small["a1"] != small["a2"])[:, None]
        i_img = np.full((8, 8), float(np.abs(0.5 * (d[:, :, 0] + d[:, :, 3]))[ok].sum()), np.float32)
        v_img = np.full((8, 8), float(np.abs(0.5j * (d[:, :, 2] - d[:, :, 1]))[ok].sum()), np.float32)
        info = dict(ms=small["path"], stem=small["stem"], status="ok", timing={"fake": 0.0},
                    phase_dir=list(small["phase_dir"]), uvw_new_sum=float(np.abs(small["uvw"]).sum()))
        prods = [(f"{req['prefix']}-I-image.fits", i_img, "I"), (f"{req['prefix']}-V-image.fits", v_img, "V")]
        return info, prods, t0

    def write(self, info, prods, t0):
        from astropy.io import fits
        files = []
        for p, a, st in prods:
            h = fits.Header()
            h["CRVAL1"] = math.degrees(info["phase_dir"][0])
            h["CRVAL2"] = math.degrees(info["phase_dir"][1])
            fits.PrimaryHDU(a[None, None], header=h).writeto(p, overwrite=True)
            files.append(p)
        info["files"] = files
        info["timing"]["TOTAL"] = time.perf_counter() - t0
        return info

    def free(self):
        pass

    def before_hour(self):
        if os.environ.get("GSI_V3_FAKE_HOUR"):
            sys.modules["ttcalx_imager.hour"] = _fake_hour_module()

    def after_hour(self):
        pass

    def gpu_info(self):
        return {}


def _fake_hour_module():
    """Tests (GSI_V3_FAKE_HOUR): ttcalx_imager.hour with an image_hour_snaps that writes one tiny
    FITS per product image (the file names of the real one) and the summary JSON."""
    import types
    m = types.ModuleType("ttcalx_imager.hour")

    def _want_v(products, pols, qa):
        return True

    def image_hour_snaps(snaps, band, out_prefix, ra, dec, products=(), exclude_stems=(), timing=None, t0=None,
                         **kw):
        from astropy.io import fits
        mode = os.environ.get("GSI_V3_FAKE_HOUR")
        if mode == "fail":
            raise RuntimeError("fake hour failure")
        if mode.startswith("die_once:"):                  # tests: the first engine dies in the hour
            marker = Path(mode.split(":", 1)[1])
            if not marker.exists():
                marker.write_text(str(os.getpid()))
                os._exit(9)
        if mode.startswith("fail_once:"):                 # tests: fails in the first engine only
            marker = Path(mode.split(":", 1)[1])
            if not marker.exists():
                marker.write_text(str(os.getpid()))
                raise RuntimeError("Error setting non-uniform points. (fake, first engine)")
        snaps.sort(key=lambda x: x["time"])
        xs = set(exclude_stems)
        drop = [i for i, x in enumerate(snaps) if x["stem"] in xs]
        keep = [x for x in snaps if x["stem"] not in xs]
        val = float(sum(np.abs(np.asarray(x["I"])).sum() for x in keep))
        out = {}
        for p in products:
            files = []
            for k in range(p.n_intervals):
                sfx = "" if p.n_intervals == 1 else f"-t{k:04d}"
                names = ([f"{out_prefix}-{p.tag}{sfx}-image.fits"] if p.image_i else []) + \
                    ([f"{out_prefix}-{p.v_tag}{sfx}-image.fits"] if p.v_tag else [])
                for n in names:
                    fits.PrimaryHDU(np.full((1, 1, 4, 4), val, np.float32)).writeto(n, overwrite=True)
                files += names
            out[p.tag if p.image_i else p.v_tag] = [dict(name=f"{out_prefix}-{p.tag}", n_snap=len(keep), files=files)]
        summary = dict(band=band, n_loaded=len(snaps), rejected=drop, phase_centre=[math.degrees(ra), math.degrees(dec)],
                       qa=[dict(stem=x["stem"], time=x["time"]) for x in snaps], timing=dict(timing or {}),
                       products=out, n_samples=int(sum(x.n for x in keep)))
        json.dump(summary, open(f"{out_prefix}-summary.json", "w"), default=float)
        return summary
    m._want_v = _want_v
    m.image_hour_snaps = image_hour_snaps
    return m


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, backend, out: Out):
        self.b, self.out = backend, out
        self.store = None
        self.store_lock = threading.Lock()
        self.store_ex = ThreadPoolExecutor(1, thread_name_prefix="v3-store")
        self.write_ex = ThreadPoolExecutor(1, thread_name_prefix="v3-write")
        self.outstanding = set()
        self._olock = threading.Lock()
        self.n_images = 0
        self.n_records = 0

    # -- helpers ------------------------------------------------------------------------------------
    def _track(self, fut: Future) -> Future:
        with self._olock:
            self.outstanding.add(fut)
        fut.add_done_callback(lambda f: self._untrack(f))
        return fut

    def _untrack(self, f):
        with self._olock:
            self.outstanding.discard(f)

    def drain(self):
        """Wait for every record / FITS write in flight."""
        while True:
            with self._olock:
                futs = list(self.outstanding)
            if not futs:
                return
            for f in futs:
                try:
                    f.result()
                except Exception:
                    pass

    def _when_done(self, futs, build):
        """Reply build(results) once every future is done (results: value or exception)."""
        futs = [f for f in futs if f is not None]
        if not futs:
            self.out.reply(build([]))
            return
        left = [len(futs)]
        lock = threading.Lock()

        def cb(_):
            with lock:
                left[0] -= 1
                last = left[0] == 0
            if last:
                res = []
                for f in futs:
                    e = f.exception()
                    res.append(e if e is not None else f.result())
                try:
                    r = build(res)
                except Exception as e:          # never lose a reply
                    r = dict(status="error", error=f"reply: {e!r}")
                self.out.reply(r)
        for f in futs:
            f.add_done_callback(cb)

    # -- ops ------------------------------------------------------------------------------------------
    def op_store_open(self, q):
        from gpu_subband_imaging.v3 import hourstore
        self.drain()
        path = q.get("path")
        header = dict(q.get("header") or {}) or None
        if header is not None:                   # as hourstore serve: pd_new from the imager's own parse
            import ttcalx_imager
            from ttcalx_imager.rotate import parse_phase_centre
            pc = q.get("phase_centre")
            header.setdefault("phase_centre", pc)
            header["pd_new"] = list(parse_phase_centre(pc)) if pc else None
            header["imager"] = ttcalx_imager.__file__
        with self.store_lock:
            if self.store is not None:
                self.store.close()
                self.store = None
            verify = True
            if path:
                _break_abandoned_lock(path, q.get("abandoned") or [])
            if path and Path(path).exists() and Path(path).stat().st_size > 0:
                # a killed engine may have left a torn tail / a dropped index line: truncate it
                # (the records before it are kept; the snapshot in flight is redone by the caller)
                hourstore.repair(path, log=lambda m: print(f"V3_STORE repair {path}: {m}", file=sys.stderr,
                                                           flush=True))
                verify = False                   # repair checked every crc
            self.store = hourstore.HourStore(path, backend=q.get("backend", "ram"), header_meta=header,
                                             verify=verify)
            st = self.store
            return dict(status="ok", path=str(st.path) if st.path else None, backend=st.backend, n=len(st),
                        stems=sorted(st.latest()), nbytes=st.nbytes())

    def _build_record(self, snap, req, rot):
        from gpu_subband_imaging.v3 import hourstore
        t0 = time.perf_counter()
        if rot is not None and rot["R"] is not None:
            rec = hourstore.records_from_snap(snap, rot["pd_record"][0], rot["pd_record"][1], want_v=True,
                                              peel=req.get("peel"), stem=req.get("stem"), path=req.get("path"),
                                              R=rot["R"])
        else:
            rec = hourstore.records_from_snap(snap, None, None, want_v=True, peel=req.get("peel"),
                                              stem=req.get("stem"), path=req.get("path"))
        t1 = time.perf_counter()
        with self.store_lock:
            if self.store is None:
                raise RuntimeError("no hour store open (store_open first)")
            rec = self.store.append(rec)
        self.n_records += 1
        return dict(stem=rec.stem, n_kept=rec.n, nbytes=rec.nbytes(), build_s=t1 - t0,
                    append_s=time.perf_counter() - t1, seq=rec.entry.get("seq"))

    def op_image_slot(self, q):
        """Not answered here: the reply comes from _when_done once the slot is free again."""
        from gpu_subband_imaging.v3 import slots
        rid = q.get("id")
        t0 = time.perf_counter()
        snap = slots.view_slot(q["slot"], writable=False)
        state = snap.meta["_slot"]["state"]
        want = q.get("require_state", slots.PEELED)
        if want is not None and state != want:
            raise RuntimeError(f"slot {q['slot']} is {slots.STATE_NAMES.get(state, state)}, "
                               f"expected {slots.STATE_NAMES.get(want, want)}")
        if q.get("stem") and snap.stem not in str(q["stem"]):
            raise RuntimeError(f"slot {q['slot']} holds {snap.stem}, request is for {q['stem']}")
        rot = rotation(snap, q.get("phase_centre"), q.get("record_phase_centre") if q.get("record") else None)
        t_rot = time.perf_counter() - t0
        rec_f = self._track(self.store_ex.submit(self._build_record, snap, q, rot)) if q.get("record") else None
        img_f = None
        err = None
        if q.get("image", True):
            try:
                info, prods, t_img = self._image_retry(snap, q, rot)
                img_f = self._track(self.write_ex.submit(self.b.write, info, prods, t_img))
                self.n_images += 1
            except Exception as e:              # reported after the record is done (slot in use)
                err = e
        del snap

        def build(res):
            r = dict(id=rid, op="image_slot", stem=q.get("stem"), timing=dict(rotation=t_rot))
            out = dict(zip([k for k, f in (("record", rec_f), ("image", img_f)) if f is not None], res))
            errors = []
            if err is not None:
                errors.append(f"image: {err!r}")
                r["fatal"] = any(m in repr(err) for m in FATAL_MARKERS)
                if not isinstance(err, MemoryError):
                    traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
            for k in ("record", "image"):
                v = out.get(k)
                if isinstance(v, BaseException):
                    errors.append(f"{k}: {v!r}")
                    r["fatal"] = r.get("fatal", False) or any(m in repr(v) for m in FATAL_MARKERS)
                elif v is not None:
                    r[k] = _image_summary(v) if k == "image" else v
            if q.get("image", True) and img_f is None and err is None:
                errors.append("image: not run")
            r["status"] = "error" if errors else "ok"
            if errors:
                r["error"] = "; ".join(errors)
            r["timing"]["total"] = time.perf_counter() - t0
            return r
        self._when_done([rec_f, img_f], build)
        return None

    def _image_retry(self, snap, q, rot):
        try:
            return self.b.image(snap, q, rot)
        except Exception as e:
            if "OutOfMemory" not in type(e).__name__ and "out of memory" not in str(e).lower():
                raise
            print(f"GPU out of memory imaging {q.get('stem')}: freeing the pool, retrying once", file=sys.stderr,
                  flush=True)
            self.b.free()
            time.sleep(2.0)
            return self.b.image(snap, q, rot)

    def op_spill(self, q):
        self.drain()
        with self.store_lock:
            if self.store is None:
                raise RuntimeError("no hour store open")
            before = self.store.backend
            p = self.store.spill(q.get("path"))
            return dict(status="ok", path=str(p), before=before, backend=self.store.backend, n=len(self.store))

    def op_hour(self, q):
        from ttcalx_imager import cli, gsiv
        self.drain()
        t0 = time.perf_counter()
        with self.store_lock:
            if self.store is None:
                raise RuntimeError("no hour store open")
            latest = self.store.latest()
            src = str(self.store.path) if self.store.path else "ram"
        expected = q.get("expected")
        stems = sorted(latest) if expected is None else sorted(expected)
        missing = [s for s in stems if s not in latest]
        if missing:
            return dict(status="error", error=f"{len(missing)} expected snapshot(s) have no record: {missing[:10]}",
                        missing=missing)
        extra = sorted(set(latest) - set(stems))
        recs = [latest[s] for s in stems]
        self.b.before_hour()
        log = q.get("log")
        try:
            with (open(log, "a") if log else contextlib.nullcontext(sys.stderr)) as fh, \
                    contextlib.redirect_stdout(fh):
                T = {}
                s = cli.run_hour_snaps(list(q["argv"]), lambda ra, dec, want_v: gsiv.lazy_snaps(
                    [(r.meta, r.arrays, src) for r in recs], ra, dec, want_v=want_v), timing=T, t0=t0)
        finally:
            self.b.after_hour()
        return dict(status="ok", n_loaded=s.get("n_loaded"), rejected=s.get("rejected"),
                    qa_stems=[x["stem"] for x in s.get("qa", [])], timing=s.get("timing"),
                    phase_centre=s.get("phase_centre"), extra_records=extra, wall=time.perf_counter() - t0,
                    products={k: [dict(name=o["name"], n_snap=o["n_snap"], beam=o.get("beam")) for o in v]
                              for k, v in s.get("products", {}).items()})

    def op_info(self, q):
        import resource
        with self.store_lock:
            st = self.store
            d = dict(status="ok", backend=self.b.name, n_images=self.n_images, n_records=self.n_records,
                     store=None if st is None else dict(path=str(st.path) if st.path else None,
                                                        backend=st.backend, n=len(st), nbytes=st.nbytes()),
                     maxrss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9)
        d.update(self.b.gpu_info())
        return d

    def close(self):
        self.drain()
        self.store_ex.shutdown(wait=True)
        self.write_ex.shutdown(wait=True)
        with self.store_lock:
            if self.store is not None:
                self.store.close()


def _break_abandoned_lock(path, pids):
    """The writer lock of an earlier engine of this worker that did not exit after SIGKILL (D state,
    'abandoned' by the client): it can never append again, so its lock must not block this one for
    the 30 min stale time.  Only locks of this host whose pid the client abandoned."""
    lk = Path(f"{path}.lock")
    try:
        other = json.loads(lk.read_text() or "{}")
    except (OSError, ValueError):
        return
    if other.get("host") == os.uname().nodename and int(other.get("pid", -1)) in {int(p) for p in pids}:
        print(f"V3_STORE {path}: removing the lock of abandoned engine pid {other.get('pid')}", file=sys.stderr,
              flush=True)
        try:
            lk.unlink()
        except OSError:
            pass


def _image_summary(info):
    return dict(status=info.get("status"), files=info.get("files", []), timing=info.get("timing"),
                niter=info.get("niter"), beam=info.get("beam"), gpu_pool_GB=info.get("gpu_pool_GB"),
                V_rms_box=info.get("V_rms_box"))


def main() -> int:
    out = Out()
    try:
        backend = FakeBackend() if os.environ.get("GSI_V3_ENGINE_BACKEND") == "fake" else GpuBackend()
        import ttcalx_imager
        from ttcalx_imager import cli, gsiv, rotate  # noqa: F401  (fail at start, not at the hour)
        assert hasattr(cli, "run_hour_snaps") and hasattr(gsiv, "lazy_snaps") and \
            hasattr(rotate, "chgcentre_uvw"), "ttcalx_imager without the gsi v3 engine support"
        from gpu_subband_imaging.v3 import hourstore, slots  # noqa: F401
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        out.reply(dict(id=0, status="error", error=f"engine start: {e!r}"))
        return 1
    eng = Engine(backend, out)
    out.reply(dict(id=0, status="ready", pid=os.getpid(), backend=backend.name,
                   imager=getattr(ttcalx_imager, "__file__", "?"), host=os.uname().nodename))
    ops = dict(store_open=eng.op_store_open, image_slot=eng.op_image_slot, spill=eng.op_spill,
               hour=eng.op_hour, info=eng.op_info)
    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            q = json.loads(line)
            rid = q.get("id")
            op = q.get("op")
            if op == "close":
                eng.close()
                out.reply(dict(id=rid, status="ok", op="close"))
                return 0
            if op not in ops:
                raise ValueError(f"unknown op {op!r}")
            r = ops[op](q)
            if r is not None:
                r.setdefault("id", rid)
                r.setdefault("op", op)
                out.reply(r)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            out.reply(dict(id=rid, status="error", error=repr(e),
                           fatal=any(m in repr(e) for m in FATAL_MARKERS)))
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
