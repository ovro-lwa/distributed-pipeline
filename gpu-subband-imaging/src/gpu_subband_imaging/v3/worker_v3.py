"""gsi v3 batch without MS files (v3.snapshot_io: shm; docs/v3_design.md 3.1, steps 4, 6, 7).

One band-hour in one process:

    tar --stream--> tmpfs raw MS --prep pool (spawn)--> shm slot (PREPPED, no write_ms_like)
        --> zest daemon "SHM <slot>" peels in place (PEELED)
        --> engine (workers/gsi_v3_engine.py): image_slot at the LST-hour centre in memory
            (no chgcentre copy, no MS read) + the compact hour record into the hour store
        --> [v3.write_kept_ms: the kept MS from the slot in a pool process]
        --> slot released; FITS fpacked / framed / moved as v2 (Worker._post_image)
    end of batch (v3.hour_source: memory): zest daemon closed, QA from the in-memory pilot rows,
        engine "hour" (the hour products from the store), place_products, the movie (in a thread
        started before the hour stage), provenance -- the band-hour's hour job in the batch.

The ring has v3.lookahead + 2 slots: at most lookahead snapshots are prepped ahead of the peel
(this replaces GSI_PREP_LOOKAHEAD), two more are with the engine.

Hour store (v3.store, v3.checkpoint; memory.plan_store): RAM, or memory-mapped from a GSIV file
on node-local disk.  With a file (v3.archive: the archive file; v3.checkpoint: local: a file
in the first v3.spill_dirs entry with room) every record is appended there first, so a rerun of
the batch on the node reuses them: a snapshot whose FITS exist and whose record is in the file
is skipped, one whose FITS exist but whose record is missing is "replayed" (prep + peel +
record, no imaging).  With records, a snapshot counts as done only with its FITS *and* its
record (the FITS are moved only after the record is stored), so the hour stage never images a
reduced set.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .. import lst
from ..config import v3_params

log = logging.getLogger("gsi.worker")


@dataclass
class Item:
    idx: int
    tar: str
    stamp: str
    slot: Path
    replay: bool = False
    raw_ms: Optional[Path] = None        # kept for the kept-MS writer (v3.write_kept_ms)
    stats: dict = field(default_factory=dict)


def _stamp(tar: str) -> str:
    from ..worker import _stamp as s
    return s(tar)


class V3Batch:
    def __init__(self, w, tars: List[str]):
        self.w = w
        self.cfg = w.cfg
        self.pl = w.cfg.pipeline
        self.v3 = v3_params(self.pl)
        self.band, self.batch = w.band, w.batch
        self.tars = list(tars)
        self.memory = self.v3.hour_source == "memory"
        self.records = self.memory or self.v3.archive != "none"
        self.write_kept = bool(self.v3.write_kept_ms and (self.pl.keep_ms_root or getattr(self.pl, "keep_ms_local", None)))
        self.lookahead = max(1, int(self.v3.lookahead))
        self.ring = None
        self.engine = None
        self.daemon = None
        self.store_info: dict = {}
        self.plan = None
        self.store_path: Optional[Path] = None
        self.recorded: set = set()           # kept stems with a record in the store
        self.record_bytes: List[int] = []
        self._lock = threading.Lock()
        self.mem_start = None
        self.hour_rec: Optional[dict] = None
        self.n_ok = self.n_fail = 0

    # -- names ------------------------------------------------------------------------------------
    def kept_stem(self, stamp: str) -> str:
        return self.w._kept_stem(stamp)

    def group(self) -> str:
        groups = {self.w.group(_stamp(t)) for t in self.tars}
        if len(groups) != 1:
            raise RuntimeError(f"batch {self.band}MHz/{self.batch} spans groups {sorted(groups)}: the gsi v3 "
                               "store needs one band-hour per batch")
        return groups.pop()

    def phase_centre(self, stamp: str) -> Optional[str]:
        if getattr(self.pl, "snapshot_phase_centre", "zenith") != "lst":
            return None
        return lst.phase_centre(self.w.lst_hour(stamp))

    def record_phase_centre(self, stamp: str) -> Optional[str]:
        """The hour record's centre: the LST-hour centre whenever the layout is lst (the hour job's)."""
        return lst.phase_centre(self.w.lst_hour(stamp)) if self.w.lst_layout else None

    # -- start ------------------------------------------------------------------------------------
    def _engine_env(self):
        from .. import hour_image
        pl = self.pl
        python = pl.imaging.gpu_python or str(Path(pl.hour_imaging.ttcalx_image).parent / "python")
        env = hour_image.imager_env(self.cfg)
        env.update(pl.imaging.gpu_env)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return python, env

    def _open_store(self, client) -> dict:
        """(Re-)open the hour store in the engine (also after an engine restart)."""
        if not self.records:
            return {}
        stamp = _stamp(self.tars[0])
        hdr = None
        if self.store_path is not None:
            from . import hourstore
            hdr = hourstore.header_for(self.band, self.pl.date, self.w.group(stamp), self.record_phase_centre(stamp),
                                       self.pl.run_label, batch=self.batch, layout=getattr(self.pl, "layout", "utc"),
                                       cal_bp=str(self.pl.cal.bp), cal_xy=str(self.pl.cal.xy))
        r = client.request("store_open", timeout=max(600.0, self.v3.engine_timeout_s),
                           path=str(self.store_path) if self.store_path else None,
                           backend=self.plan.backend if self.plan else "ram", header=hdr,
                           phase_centre=self.record_phase_centre(stamp))
        self.store_info = r
        log.info("V3_STORE %sMHz batch%d: %s backend=%s, %d record(s) already there (%.2f GB)", self.band,
                 self.batch, r.get("path") or "RAM only", r.get("backend"), r.get("n", 0), r.get("nbytes", 0) / 1e9)
        return r

    def _plan_store(self, n_todo: int) -> None:
        from . import hourstore, memory
        if not self.records:
            return
        group = self.group()
        want_file = self.v3.archive == "none" and self.v3.checkpoint == "local"
        self.plan = memory.plan_store(self.v3.store, n_todo, self.band, self.v3.ram_reserve_gb, self.v3.spill_dirs,
                                      want_file=want_file)
        if self.v3.archive != "none":
            self.store_path = hourstore.archive_path(self.cfg, self.band, group, self.batch)
        elif self.plan.spill_dir is not None and (want_file or self.plan.backend == "memmap"):
            self.store_path = (Path(self.plan.spill_dir) / self.pl.run_key /
                               f"{self.band}MHz-{self.pl.date}-{group}-b{int(self.batch):03d}.gsiv")
        if self.plan.backend == "memmap":
            log.warning("V3_MEM_SPILL %sMHz batch%d at start: %s -> %s", self.band, self.batch, self.plan.reason,
                        self.store_path)
        else:
            log.info("V3_STORE_PLAN %sMHz batch%d: RAM (%s)%s", self.band, self.batch, self.plan.reason,
                     f", checkpoint {self.store_path}" if self.store_path else "")
        for n in self.plan.notes:
            log.warning("V3_STORE_PLAN %sMHz batch%d: %s", self.band, self.batch, n)
        memory.reserve(self.plan.need_bytes if self.plan.backend == "ram" else 0.0, band=self.band, batch=self.batch)

    def _start_engine(self):
        from . import engine_client
        python, env = self._engine_env()
        c = engine_client.EngineClient(python, engine_client.engine_py(), env,
                                       start_timeout=self.v3.engine_start_timeout_s,
                                       on_restart=lambda cl: self._open_store(cl))
        log.info("batch %sMHz/%d: gsi v3 engine pid %s (%s, imager %s)", self.band, self.batch, c.pid,
                 c.info.get("backend"), c.info.get("imager"))
        return c

    # -- feeders -----------------------------------------------------------------------------------
    def _feed(self, idx: int, tar: str, slot: Path, replay: bool) -> Optional[Item]:
        from . import fastprep
        w = self.w
        stamp = _stamp(tar)
        d = w.work / stamp
        udir = w._untar_dir(stamp)
        keep_raw = False
        try:
            t0 = time.perf_counter()
            ms = w._stage(tar, d, stamp)
            log.info("timing %sMHz batch%d stamp=%s phase=stage_ms elapsed=%.2f s",
                     self.band, self.batch, stamp, time.perf_counter() - t0)
            if self.write_kept:
                why = None
                if w._v3_template_for(ms) is None:
                    why = "no mstransform template"
                else:
                    try:
                        fastprep.check_template(str(w._v3_template), str(ms), self.pl.chanbin)
                        keep_raw = True
                    except Exception as e:
                        why = f"template does not fit: {e}"
                if why:
                    if self.v3.hour_source == "kept_ms":
                        raise RuntimeError(f"cannot write the kept MS ({why}); hour_source=kept_ms needs it")
                    log.warning("V3_KEPT %sMHz stamp=%s: no kept MS (%s)", self.band, stamp, why)
            peel = self.pl.peel_for_band(self.band)
            t0 = time.perf_counter()
            res = w._v3_run(fastprep.prep_slot_task, *w._v3_prep_args(ms), str(slot),
                            float(peel.min_visibility_nonzero_fraction))
            self._log_prep(stamp, res, time.perf_counter() - t0)
            st = res["stats"]
            log.info("input_quality %sMHz stamp=%s sampled=%d usable=%d nonzero_fraction=%.6g median_abs=%.6g "
                     "max_abs=%.6g", self.band, stamp, st["sampled_values"], st["usable_values"],
                     st["nonzero_fraction"], st["median_amplitude"], st["max_amplitude"])
            if not res["slot_written"]:
                log.warning("skipping %sMHz stamp=%s before peeling: visibility nonzero_fraction %.6g is below %.6g",
                            self.band, stamp, st["nonzero_fraction"], peel.min_visibility_nonzero_fraction)
                raise _Skip()
            if not keep_raw:
                shutil.rmtree(ms, ignore_errors=True)
            return Item(idx, tar, stamp, slot, replay, ms if keep_raw else None, st)
        except _Skip:
            keep_raw = False
            self._release(slot)
            return None
        except Exception as e:
            log.error("preprocess %s failed: %s", stamp, e)
            keep_raw = False
            self._release(slot)
            return None
        finally:
            if not keep_raw:
                shutil.rmtree(d, ignore_errors=True)
                if udir is not None:
                    shutil.rmtree(udir, ignore_errors=True)

    def _log_prep(self, stamp: str, res: dict, elapsed: float) -> None:
        meta = res["meta"]
        self.w._log_v3_timings(stamp, meta, elapsed)
        tm = meta.get("timings", {})
        wall = lambda *names: sum(tm.get(n, {}).get("wall", 0.0) for n in names)  # noqa: E731
        task = float(res.get("task_wall", elapsed))
        for phase, value in (("flag_cal", wall("read", "cal")), ("average", wall("average")),
                             ("flag_cal_avg", task), ("inmem_wait", max(0.0, elapsed - task))):
            log.info("timing %sMHz batch%d stamp=%s phase=%s elapsed=%.2f s", self.band, self.batch, stamp, phase, value)
        mode = self.pl.fix_uvw
        if mode != "off":
            if meta.get("fix_uvw_error"):
                log.warning("fix_uvw %sMHz stamp=%s failed, stored UVW kept: %s", self.band, stamp, meta["fix_uvw_error"])
            elif meta.get("fix_uvw"):
                f = meta["fix_uvw"]
                log.info("fix_uvw %sMHz stamp=%s epoch=%s rot_deg=%.4f rms_duv_m=%.3f max_dw_m=%.2e rephased=%s",
                         self.band, stamp, mode, f["rot_deg"], f["rms_duv_m"], f["max_dw_m"], f["rephased"])
                log.info("timing %sMHz batch%d stamp=%s phase=fix_uvw elapsed=%.2f s", self.band, self.batch, stamp,
                         wall("uvw"))

    def _release(self, slot: Path) -> None:
        try:
            self.ring.release(slot)
        except Exception as e:
            log.error("slot release %s: %s", slot, e)

    def _cleanup_item(self, it: Item) -> None:
        shutil.rmtree(self.w.work / it.stamp, ignore_errors=True)
        udir = self.w._untar_dir(it.stamp)
        if udir is not None:
            shutil.rmtree(udir, ignore_errors=True)

    # -- after the peel (one thread per snapshot in flight) -------------------------------------------
    def _engine_request(self, it: Item, prefix: Path):
        geom, ip = self.w.geom, self.pl.imaging
        stem = self.kept_stem(it.stamp)
        # the record's path: the kept MS's (as the step-5 archive hook names it), else <stem>.ms
        path = str(self.w.kept_ms_path(it.stamp, self.w.keep_root())) if self.write_kept else f"{stem}.ms"
        return dict(slot=str(it.slot), prefix=str(prefix), pixels=geom.pixels, scale=geom.scale,
                    niter=ip.clean_niter, phase_centre=self.phase_centre(it.stamp),
                    record_phase_centre=self.record_phase_centre(it.stamp), image=not it.replay,
                    record=self.records, stem=stem, path=path)

    def _post(self, it: Item) -> bool:
        """Engine (image + record), kept MS, slot release, FITS archive.  True: snapshot done."""
        from . import engine_client, fastprep
        w = self.w
        t0 = time.perf_counter()
        prefix = w.work / it.stamp / f"img_{it.stamp}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        req = self._engine_request(it, prefix)
        released = False
        try:
            fut = self.engine.submit("image_slot", **req)
            kept_tmp = None
            if self.write_kept and it.raw_ms is not None and not (it.replay and w.kept_ms_path(it.stamp, w.keep_root()).exists()):
                t1 = time.perf_counter()
                kept_tmp = w.work / it.stamp / f"{it.stamp}_kept.ms"
                try:
                    w._v3_run(fastprep.kept_from_slot_task, str(w._v3_template), str(it.raw_ms), str(it.slot),
                              str(kept_tmp))
                    log.info("timing %sMHz batch%d stamp=%s phase=kept_write elapsed=%.2f s", self.band,
                             self.batch, it.stamp, time.perf_counter() - t1)
                except Exception as e:
                    log.error("kept MS %s failed: %s", it.stamp, e)
                    kept_tmp = None
                    if self.v3.hour_source == "kept_ms":
                        raise
            try:
                r = fut.result(self.v3.engine_timeout_s)
            except (engine_client.EngineDied, engine_client.EngineTimeout) as e:
                log.error("V3_ENGINE %sMHz stamp=%s: %s; retrying once on a new engine", self.band, it.stamp, e)
                r = self.engine.request("image_slot", timeout=self.v3.engine_timeout_s, **req)
            self._release(it.slot)
            released = True
            self._log_engine(it, r, time.perf_counter() - t0)
            if r.get("record"):
                with self._lock:
                    self.recorded.add(self.kept_stem(it.stamp))
                    self.record_bytes.append(int(r["record"].get("nbytes", 0)))
            if kept_tmp is not None:
                t1 = time.perf_counter()
                kept = w.keep_ms(it.stamp, kept_tmp)
                log.info("timing %sMHz batch%d stamp=%s phase=keep_ms elapsed=%.2f s dst=%s", self.band, self.batch,
                         it.stamp, time.perf_counter() - t1, kept)
            if not it.replay:
                files = [Path(f"{prefix}-I-image.fits"), Path(f"{prefix}-V-image.fits")]
                missing = [str(f) for f in files if not f.exists()]
                if missing:
                    raise RuntimeError(f"engine reported ok but {missing} are missing")
                w._post_image(it.stamp, files)
                if self.v3.archive != "none":
                    w._archive_expected(it.stamp)
            return True
        finally:
            if not released:
                self._release(it.slot)
            self._cleanup_item(it)

    def _log_engine(self, it: Item, r: dict, wall: float) -> None:
        tm = r.get("timing", {})
        img = r.get("image") or {}
        it_t = img.get("timing") or {}
        if not it.replay:
            log.info("timing %sMHz batch%d stamp=%s phase=image_gpu elapsed=%.2f s", self.band, self.batch, it.stamp,
                     float(it_t.get("TOTAL", 0.0)))
            for k in ("H2D + rephase", "select (GPU)", "D2H", "write FITS"):
                if k in it_t:
                    log.info("timing %sMHz batch%d stamp=%s phase=engine_%s elapsed=%.3f s", self.band, self.batch,
                             it.stamp, k.split()[0].lower().replace("+", ""), float(it_t[k]))
        log.info("timing %sMHz batch%d stamp=%s phase=rotation elapsed=%.3f s", self.band, self.batch, it.stamp,
                 float(tm.get("rotation", 0.0)))
        rec = r.get("record")
        if rec:
            log.info("timing %sMHz batch%d stamp=%s phase=record elapsed=%.2f s build=%.2f append=%.2f nbytes=%s "
                     "n_kept=%s", self.band, self.batch, it.stamp, rec.get("build_s", 0) + rec.get("append_s", 0),
                     rec.get("build_s", 0), rec.get("append_s", 0), rec.get("nbytes"), rec.get("n_kept"))
        log.info("timing %sMHz batch%d stamp=%s phase=%s elapsed=%.2f s", self.band, self.batch, it.stamp,
                 "replay" if it.replay else "image", wall)

    # -- batch ----------------------------------------------------------------------------------------
    def _hour_done(self) -> bool:
        stamp = _stamp(self.tars[0])
        p = self.w.band_dir(stamp) / "provenance.json"
        try:
            h = json.loads(p.read_text()).get("hour", {})
            return h.get("status") == "ok" and h.get("hour_source") == "memory"
        except (OSError, ValueError):
            return False

    def run(self) -> None:
        from . import memory, slots
        w = self.w
        have_fits = {t for t in self.tars if w._fits_path_for_stamp(_stamp(t)).exists()}
        todo = [t for t in self.tars if t not in have_fits]
        if not todo and (not self.memory or self._hour_done()):
            log.info("batch %sMHz/%d: nothing to do", self.band, self.batch)
            return
        self.mem_start = memory.mem_available()
        batch_start = time.perf_counter()
        w.work.mkdir(parents=True, exist_ok=True)
        self._plan_store(len(todo))
        t0 = time.perf_counter()
        w._bp = w._stage_cal(self.pl.cal.bp)
        w._xy = w._stage_cal(self.pl.cal.xy)
        log.info("timing %sMHz batch%d phase=stage_cal elapsed=%.2f s", self.band, self.batch, time.perf_counter() - t0)
        if getattr(self.pl, "do_badants", True):
            from ..stages import flag
            from ..worker import _mjd
            t0 = time.perf_counter()
            w._badants = flag.badants(_mjd(_stamp((todo or self.tars)[0])), f"{self.pl.env.dev_env}/bin/python")
            log.info("timing %sMHz batch%d phase=badants(batch) n=%s elapsed=%.2f s", self.band, self.batch,
                     len(w._badants.split(",")) if w._badants else 0, time.perf_counter() - t0)
        hour_ok = False
        self._qa_written = False
        try:
            self.daemon = w._start_daemon()          # Julia JIT overlaps the engine start and the prep
            self.engine = self._start_engine()
            have_rec = set(self._open_store(self.engine).get("stems", []))
            replay = [t for t in self.tars if t in have_fits and self.records and
                      self.kept_stem(_stamp(t)) not in have_rec]
            self.recorded |= have_rec
            if replay:
                log.warning("V3_REPLAY %sMHz batch%d: %d snapshot(s) have FITS but no record: prep + peel + record "
                            "without imaging", self.band, self.batch, len(replay))
            work = [(t, False) for t in todo] + [(t, True) for t in replay]
            work.sort(key=lambda x: _stamp(x[0]))
            self.ring = slots.SlotRing(self.lookahead + 2)
            log.info("batch %sMHz/%d: gsi v3 shm batch, %d to image, %d to replay, ring of %d slots, write_kept_ms=%s, "
                     "records=%s, hour_source=%s", self.band, self.batch, len(todo), len(replay), len(self.ring),
                     self.write_kept, self.records, self.v3.hour_source)
            w._v3_sweep_untar()
            w._v3_start_pool()
            t_snap = time.perf_counter()
            self._run_snapshots(work)
            log.info("timing %sMHz batch%d phase=snapshots elapsed=%.2f s n=%d", self.band, self.batch,
                     time.perf_counter() - t_snap, len(work))
            self.daemon.close()
            self.daemon = None
            w._v3_stop_pool()
            if self.ring is not None:
                self.ring.close()
                self.ring = None
            if self.memory:
                # the batch's output is the band-hour: snapshots done in earlier runs count (a rerun
                # whose only remaining inputs are skipped snapshots still makes the hour products)
                from .. import hour_lst
                if work and not self.n_ok and not hour_lst._fits_stems(
                        self.cfg, self.band, int(self.group().rstrip("h"))):
                    raise RuntimeError(f"batch {self.band}MHz/{self.batch} produced no outputs from "
                                       f"{len(work)} inputs")
                self._hour_stage()
                hour_ok = True
        finally:
            if self.daemon is not None:
                self.daemon.close()
            if self.engine is not None:
                try:
                    info = self.engine.request("info", timeout=60)
                    log.info("V3_ENGINE_INFO %sMHz batch%d: maxrss %.1f GB, %s", self.band, self.batch,
                             info.get("maxrss_GB", 0), {k: v for k, v in info.items() if k not in ("id", "op")})
                except Exception:
                    pass
                self.engine.close()
            w._v3_stop_pool()
            if self.ring is not None:
                self.ring.close()
            memory.unreserve()
            shutil.rmtree(w.work, ignore_errors=True)
            root = w._untar_root()
            if root is not None and not root.is_symlink():
                shutil.rmtree(root, ignore_errors=True)
            if not self._qa_written:
                w._write_qa_stats()
            if hour_ok and self.store_path is not None and self.v3.archive == "none" and self.v3.delete_checkpoint:
                self._delete_checkpoint()
        log.info("batch %sMHz/%d: total elapsed %.2f s for %d files (%d failed)", self.band, self.batch,
                 time.perf_counter() - batch_start, self.n_ok, self.n_fail)
        if self.n_fail:
            log.warning("batch %sMHz/%d completed with %d failed file(s); continuing", self.band, self.batch, self.n_fail)
        if not self.memory:
            from ..worker import _require_batch_outputs
            _require_batch_outputs(self.band, self.batch, len(todo), self.n_ok)

    def _delete_checkpoint(self) -> None:
        from . import hourstore
        p = self.store_path
        for f in (p, hourstore.idx_path(p), Path(f"{p}.lock")):
            try:
                Path(f).unlink()
            except OSError:
                pass
        log.info("V3_CHECKPOINT %sMHz batch%d: deleted %s (hour products done)", self.band, self.batch, p)

    def _run_snapshots(self, work) -> None:
        from . import memory
        n = len(work)
        guard = memory.MemoryGuard(self.v3.ram_reserve_gb, every=self.v3.mem_check_every)
        recycle = getattr(self.cfg.cluster, "zest_recycle_files", 25)
        futs: Dict[int, Future] = {}
        nxt = [0]
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.cluster.max_feeders)) as feeders, \
                ThreadPoolExecutor(max_workers=3) as post:
            def top_up(i: int) -> None:
                while nxt[0] < n and nxt[0] <= i + self.lookahead:
                    try:
                        slot = self.ring.acquire(timeout=None if nxt[0] <= i else 0)
                    except TimeoutError:
                        return
                    j = nxt[0]
                    futs[j] = feeders.submit(self._feed, j, work[j][0], slot, work[j][1])
                    nxt[0] += 1
            posts: List[Future] = []
            n_zested = 0
            for i in range(n):
                top_up(i)
                it = futs.pop(i).result()
                top_up(i)
                if it is None:
                    continue
                if recycle and n_zested and n_zested % recycle == 0:
                    log.info("batch %sMHz/%d: recycling zest daemon after %d files", self.band, self.batch, n_zested)
                    self.daemon.close()
                    self.daemon = self.w._start_daemon()
                try:
                    t0 = time.perf_counter()
                    self._peel(it)
                    n_zested += 1
                    log.info("timing %sMHz batch%d stamp=%s phase=zest elapsed=%.2f s", self.band, self.batch,
                             it.stamp, time.perf_counter() - t0)
                except Exception as e:
                    log.error("zest %s failed: %s", it.stamp, e)
                    self._release(it.slot)
                    self._cleanup_item(it)
                    self.n_fail += 1
                    continue
                posts.append(post.submit(self._post, it))
                if self.records and guard.check():
                    self._spill_now("MemAvailable below v3.ram_reserve_gb")
            for f in posts:
                try:
                    f.result()
                    self.n_ok += 1
                except Exception as e:
                    log.error("image/archive failed: %s", e)
                    self.n_fail += 1

    def _peel(self, it: Item) -> None:
        from . import slots
        self.daemon.zest_slot(it.slot, timeout=self.v3.peel_timeout_s)
        state = slots.get_state(it.slot)
        if state != slots.PEELED:
            try:
                self.daemon.reset_warm()
            except Exception as e:
                log.error("RESET_WARM failed: %s", e)
            raise RuntimeError(f"slot {it.slot} is {slots.STATE_NAMES.get(state, state)} after ZESTED")

    def _spill_now(self, why: str) -> None:
        from . import memory
        if self.plan is None or self.plan.backend == "memmap":
            return
        path = self.store_path
        if path is None:
            d = memory.first_spill_dir(self.v3.spill_dirs, max(self.record_bytes or [0]) * len(self.tars))
            if d is None:
                log.error("V3_MEM_SPILL %sMHz batch%d wanted (%s) but no spill dir has room: staying in RAM",
                          self.band, self.batch, why)
                return
            path = Path(d) / self.pl.run_key / f"{self.band}MHz-{self.pl.date}-{self.group()}-b{int(self.batch):03d}.gsiv"
        try:
            r = self.engine.request("spill", timeout=max(600.0, self.v3.engine_timeout_s), path=str(path))
        except Exception as e:
            log.error("V3_MEM_SPILL %sMHz batch%d failed: %s", self.band, self.batch, e)
            return
        self.store_path = Path(r["path"])
        self.plan.backend = "memmap"
        self.lookahead = min(self.lookahead, 2)
        memory.reserve(0.0, band=self.band, batch=self.batch)
        log.warning("V3_MEM_SPILL %sMHz batch%d: %s -> %s (%d records), lookahead %d", self.band, self.batch, why,
                    r["path"], r.get("n", 0), self.lookahead)

    # -- hour stage (v3.hour_source: memory) ----------------------------------------------------------------
    def _hour_stage(self) -> None:
        from .. import hour_lst
        w, pl, h = self.w, self.pl, self.pl.hour_imaging
        group = self.group()
        lst_hour = int(group.rstrip("h"))
        bdir = w.band_dir(_stamp(self.tars[0]))
        qa_dir = bdir / "QA"
        qa_dir.mkdir(parents=True, exist_ok=True)
        t_all = time.perf_counter()
        rec: Dict = dict(band=self.band, date=pl.date, lst_label=group, run_label=pl.run_label, host=os.uname().nodename,
                         gpu=w.gpu, products=list(h.products), hour_source="memory",
                         start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # 1. QA from the in-memory pilot rows (+ the rows of earlier runs of this band-hour)
        t0 = time.perf_counter()
        w._write_qa_stats()
        self._qa_written = True
        rows = {r["ts"]: r for r in hour_lst.read_stats(qa_dir)}
        rows.update({r["ts"]: r for r in w._qa})
        bad_ts, rec["qa"] = hour_lst.qa_step(qa_dir, self.band, [rows[k] for k in sorted(rows)])
        expected = sorted(hour_lst._fits_stems(self.cfg, self.band, lst_hour))
        xs = [s for s in expected if lst.timestamp(s) in bad_ts]
        rec["qa"]["excluded_from_hour"] = len(xs)
        rec["n_records"] = len(expected)
        rec["timing_s"] = {"qa": time.perf_counter() - t0}
        log.info("hour %sMHz %s: %d snapshots from memory, QA %s", self.band, group, len(expected), rec["qa"])
        # 4. the movie overlaps the GPU hour stage (every frame is written by now)
        mv: dict = {}
        mt = None
        if pl.movie.enabled:
            mt = threading.Thread(target=lambda: mv.update(r=hour_lst.movie_step(self.cfg, self.band, group, bdir)),
                                  name="v3-movie", daemon=True)
            mt.start()
        # 2. the hour products in the engine
        work = bdir / ".hour_work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        excl = []
        if xs:
            (work / "exclude-stems.txt").write_text("\n".join(xs) + "\n")
            excl = ["--exclude-stems", f"@{work / 'exclude-stems.txt'}"]
        argv = (["--band", str(self.band), "--out", str(work / f"{self.band}MHz"), "--products", ",".join(h.products),
                 "--lst-hour", str(lst_hour), "--qa", "none", "--nproc", str(h.nproc), "-v"] + excl + list(h.extra_args))
        rec["cmd"] = ["engine:hour"] + argv
        t0 = time.perf_counter()
        rc, imaged = 0, set()
        try:
            r = self.engine.request("hour", timeout=h.timeout_seconds, argv=argv, expected=expected,
                                    log=str(qa_dir / f"{self.band}MHz-ttcalx-image.log"))
            imaged = set(r.get("qa_stems") or [])
            rec["imager_timing_s"] = r.get("timing", {})
            rec["n_loaded"] = r.get("n_loaded")
            rec["n_rejected_hour"] = len(r.get("rejected") or [])
            rec["phase_centre_deg"] = r.get("phase_centre")
            if r.get("extra_records"):
                rec["extra_records"] = r["extra_records"]
                log.warning("hour %sMHz %s: %d record(s) without CLEAN FITS not imaged: %s", self.band, group,
                            len(r["extra_records"]), r["extra_records"][:5])
        except Exception as e:
            log.error("hour %sMHz %s (memory) failed: %s", self.band, group, e)
            rec["error"] = str(e)
            rc = -1
        rec["timing_s"]["hour_image"] = time.perf_counter() - t0
        summary = work / f"{self.band}MHz-summary.json"
        if summary.exists():
            shutil.move(str(summary), str(qa_dir / summary.name))
        if rc == 0 and imaged != set(expected):
            rec["imaged_mismatch"] = dict(missing=sorted(set(expected) - imaged), extra=sorted(imaged - set(expected)))
            log.error("hour %sMHz %s: the engine imaged %d snapshots, expected %d: %s", self.band, group, len(imaged),
                      len(expected), rec["imaged_mismatch"])
            rc, imaged = -3, set()
        rec["rc"] = rc
        # 3. products
        t0 = time.perf_counter()
        rec["files"] = hour_lst.place_products(work, bdir, expected) if rc == 0 else []
        rec["timing_s"]["place_products"] = time.perf_counter() - t0
        shutil.rmtree(work, ignore_errors=True)
        rec["status"] = "ok" if rc == 0 else "failed"
        if mt is not None:
            t0 = time.perf_counter()
            mt.join()
            rec["movie"], rec["timing_s"]["movie"] = mv.get("r", (None, 0.0))
            rec["timing_s"]["movie_wait"] = time.perf_counter() - t0
        rec["timing_s"]["total"] = time.perf_counter() - t_all
        from . import memory
        v3b = dict(hour_source="memory", snapshot_io=self.v3.snapshot_io, peel_io=self.v3.peel_io,
                   prep_inmem=self.v3.prep_inmem, untar_to=self.v3.untar_to, write_kept_ms=self.write_kept,
                   store=self.plan.as_dict() if self.plan else None,
                   store_path=str(self.store_path) if self.store_path else None, archive=self.v3.archive,
                   n_records=len(expected), engine_restarts=getattr(self.engine, "restarts", 0),
                   worker_peak_rss_GB=memory.peak_rss() / 1e9,
                   mem_available_GB_start=None if self.mem_start is None else self.mem_start / 1e9,
                   mem_available_GB_end=(memory.mem_available() or 0) / 1e9)
        try:
            info = self.engine.request("info", timeout=60)
            v3b["engine_peak_rss_GB"] = info.get("maxrss_GB")
        except Exception:
            pass
        if self.store_path is not None and self.store_path.exists():
            v3b["store_file_bytes"] = self.store_path.stat().st_size
        hour_lst.write_provenance(self.cfg, self.band, group, w.gpu, rec, v3b)
        self.hour_rec = rec
        log.info("timing %sMHz lst=%s phase=hour_total elapsed=%.2f s rc=%s n_ms=%d", self.band, group,
                 rec["timing_s"]["total"], rc, len(expected))
        if rc != 0:
            raise RuntimeError(f"hour {self.band}MHz {group} (memory) failed (rc={rc}): {rec.get('error', '')}")
        if h.delete_kept_ms and imaged:
            hour_lst._delete_kept(self.cfg, self.band, group, imaged)


class _Skip(Exception):
    pass


def run_batch_v3(w, tars: List[str]) -> None:
    V3Batch(w, tars).run()
