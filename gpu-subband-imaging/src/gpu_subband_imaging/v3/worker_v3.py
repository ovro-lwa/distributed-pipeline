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
from contextlib import contextmanager
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
    imaged: threading.Event = field(default_factory=threading.Event)    # the engine is done with it


class _EngineGate:
    """Engine requests are "shared" (several in flight); a retry after an engine death is
    "exclusive" (alone in the engine), so a request that kills the engine again while alone is
    the culprit, and a healthy sibling that died with it gets its retry without being blamed."""

    def __init__(self):
        self._cv = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting = 0

    @contextmanager
    def shared(self):
        with self._cv:
            self._cv.wait_for(lambda: not self._writer and not self._waiting)
            self._readers += 1
        try:
            yield
        finally:
            with self._cv:
                self._readers -= 1
                self._cv.notify_all()

    @contextmanager
    def exclusive(self):
        with self._cv:
            self._waiting += 1
            self._cv.wait_for(lambda: not self._writer and self._readers == 0)
            self._waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cv:
                self._writer = False
                self._cv.notify_all()


class BatchAborted(RuntimeError):
    """The batch cannot go on (engine failed for good, records lost, budget): stop prepping and peeling."""


def _permanent_engine_error(e: BaseException) -> bool:
    """An engine error reply marked permanent (the imager's DegeneratePSFError: the snapshot's data
    cannot give a usable PSF): a property of the input, recorded as given up at once (no retry)."""
    from .engine_client import EngineError
    return isinstance(e, EngineError) and bool(e.reply.get("permanent"))


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
        self.n_prep_fail = 0                 # prep errors (not the no-data skips)
        self.failures: Dict[str, dict] = {}  # this attempt: stamp -> {reason, permanent}
        self.given_up: Dict[str, dict] = {}  # inputs failed in max_input_failures attempts (or vanished)
        self._budget_fired = False
        self.gate = _EngineGate()
        self.fatal: Optional[str] = None     # set -> the batch is aborted at once
        self.deadline: Optional[float] = None
        self.n_guard = 0                     # memory guard escalation level

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
        if self.memory and self.store_path is None and client.restarts:
            raise BatchAborted("engine restarted without a record file: the band-hour's records are lost")
        r = client.request("store_open", timeout=max(600.0, self.v3.engine_timeout_s),
                           abandoned=list(client.abandoned),
                           path=str(self.store_path) if self.store_path else None,
                           backend=self.plan.backend if self.plan else "ram", header=hdr,
                           phase_centre=self.record_phase_centre(stamp))
        self.store_info = r
        log.info("V3_STORE %sMHz batch%d: %s backend=%s, %d record(s) already there (%.2f GB)", self.band,
                 self.batch, r.get("path") or "RAM only", r.get("backend"), r.get("n", 0), r.get("nbytes", 0) / 1e9)
        return r

    def _ck_name(self, d) -> Path:
        return Path(d) / self.pl.run_key / f"{self.band}MHz-{self.pl.date}-{self.group()}-b{int(self.batch):03d}.gsiv"

    def _existing_store(self):
        """(path, n records, bytes) of a record file this batch wrote before (archive or local checkpoint)."""
        from . import hourstore
        cands = [hourstore.archive_path(self.cfg, self.band, self.group(), self.batch)] if self.v3.archive != "none" \
            else [self._ck_name(d) for d in self.v3.spill_dirs if d]
        for p in cands:
            try:
                if p.exists() and p.stat().st_size > 0:
                    return p, len(hourstore.scan(p).entries), p.stat().st_size
            except Exception as e:                       # a damaged file is repaired at store_open
                log.warning("V3_STORE_PLAN %s: %s", p, e)
                return p, 0, p.stat().st_size
        return None, 0, 0

    def _old_ck_names(self) -> List[Path]:
        """Checkpoints named with the run_key before it got its "_v3mem" suffix (review L6)."""
        rk = self.pl.run_key
        if not rk.endswith("_v3mem"):
            return []
        old = rk[:-len("_v3mem")]
        return [Path(d) / old / self._ck_name(d).name for d in self.v3.spill_dirs if d]

    def _migrate_old_checkpoints(self) -> None:
        from . import hourstore
        if self.v3.archive != "none":
            return
        for o in self._old_ck_names():
            n = self._ck_name(o.parents[1])
            if not o.exists() or n.exists():
                continue
            n.parent.mkdir(parents=True, exist_ok=True)
            for a, b in ((o, n), (hourstore.idx_path(o), hourstore.idx_path(n))):
                if a.exists():
                    os.replace(a, b)
            try:
                Path(f"{o}.lock").unlink()
            except OSError:
                pass
            log.warning("V3_CHECKPOINT %sMHz batch%d: moved the checkpoint of the old run_key %s -> %s", self.band,
                        self.batch, o, n)

    def _delete_done_checkpoints(self) -> None:
        """The band-hour's hour products are done: its local checkpoints (old and new names) can go."""
        from . import hourstore
        if self.v3.archive != "none" or not self.v3.delete_checkpoint or not self.records:
            return
        for p in [self._ck_name(d) for d in self.v3.spill_dirs if d] + self._old_ck_names():
            if p.exists():
                for f in (p, hourstore.idx_path(p), Path(f"{p}.lock")):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                log.info("V3_CHECKPOINT %sMHz batch%d: deleted %s (hour products done)", self.band, self.batch, p)

    def _plan_store(self, n_inputs: int) -> None:
        """RAM or spill for the band-hour's records: every input that may give one (todo, replays and
        the records already in the checkpoint), at the checkpoint's own bytes per record when there is
        one, plus the fixed per-process term (prep processes, Julia, engine)."""
        from . import hourstore, memory
        if not self.records:
            return
        group = self.group()
        want_file = self.v3.archive == "none" and self.v3.checkpoint == "local"
        self._migrate_old_checkpoints()
        old, n_old, b_old = self._existing_store()
        rb = b_old / n_old if n_old else None
        self.plan = memory.plan_store(self.v3.store, max(n_inputs, n_old), self.band, self.v3.ram_reserve_gb,
                                      self.v3.spill_dirs, record_bytes=rb, want_file=want_file,
                                      prefer_dir=str(old.parents[1]) if old is not None and want_file else None)
        if self.v3.archive != "none":
            self.store_path = hourstore.archive_path(self.cfg, self.band, group, self.batch)
        elif self.plan.spill_dir is not None and (want_file or self.plan.backend == "memmap"):
            self.store_path = self._ck_name(self.plan.spill_dir)
        if self.memory and self.store_path is None:
            log.error("V3_NO_CHECKPOINT %sMHz batch%d: no record file (%s): an engine death aborts the batch",
                      self.band, self.batch, "; ".join(self.plan.notes) or "no spill dir")
        if self.plan.backend == "memmap":
            log.warning("V3_MEM_SPILL %sMHz batch%d at start: %s -> %s", self.band, self.batch, self.plan.reason,
                        self.store_path)
        else:
            log.info("V3_STORE_PLAN %sMHz batch%d: RAM (%s)%s", self.band, self.batch, self.plan.reason,
                     f", checkpoint {self.store_path}" if self.store_path else "")
        for n in self.plan.notes:
            log.warning("V3_STORE_PLAN %sMHz batch%d: %s", self.band, self.batch, n)
        # only the records: the processes' own fixed part is in the other workers' MemAvailable already
        memory.reserve(self.plan.record_bytes if self.plan.backend == "ram" else 0.0, band=self.band,
                       batch=self.batch)

    def _start_engine(self):
        from . import engine_client
        python, env = self._engine_env()
        c = engine_client.EngineClient(python, engine_client.engine_py(), env,
                                       start_timeout=self.v3.engine_start_timeout_s,
                                       on_restart=lambda cl: self._open_store(cl),
                                       max_restarts=self.v3.engine_max_restarts)
        log.info("batch %sMHz/%d: gsi v3 engine pid %s (%s, imager %s)", self.band, self.batch, c.pid,
                 c.info.get("backend"), c.info.get("imager"))
        return c

    # -- skipped inputs, budget, /dev/shm, memory guard ---------------------------------------------
    def _skipped_path(self) -> Path:
        return self.w.band_dir(_stamp(self.tars[0])) / "QA" / ".v3_skipped.jsonl"

    def _known_skipped(self) -> set:
        """Stamps an earlier run skipped before the peel (no data): not prepped again on reruns."""
        try:
            return {json.loads(ln)["stamp"] for ln in self._skipped_path().read_text().splitlines() if ln.strip()}
        except (OSError, ValueError, KeyError):
            return set()

    def _note_skipped(self, stamp: str, nonzero: float) -> None:
        p = self._skipped_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, open(p, "a") as fh:
                fh.write(json.dumps(dict(stamp=stamp, nonzero_fraction=nonzero, t=time.time())) + "\n")
        except OSError as e:
            log.warning("cannot note skipped %s: %s", stamp, e)

    # -- inputs that fail every attempt (review M1) ----------------------------------------------------
    def _failed_path(self) -> Path:
        return self.w.band_dir(_stamp(self.tars[0])) / "QA" / ".v3_failed.jsonl"

    def _max_failures(self) -> int:
        n = int(getattr(self.v3, "max_input_failures", 0) or 0)
        return n if n > 0 else int(getattr(self.pl, "max_retries", 1)) + 1

    def _fail(self, stamp: str, reason: str, permanent: bool = False) -> None:
        with self._lock:
            d = self.failures.setdefault(stamp, dict(reasons=[], permanent=False))
            d["reasons"].append(reason[:500])
            d["permanent"] = d["permanent"] or permanent

    def _known_failed(self) -> Dict[str, dict]:
        """stamp -> {n: attempts that failed it, permanent, reasons} from QA/.v3_failed.jsonl."""
        out: Dict[str, dict] = {}
        try:
            lines = self._failed_path().read_text().splitlines()
        except OSError:
            return out
        for ln in lines:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            d = out.setdefault(r["stamp"], dict(n=0, permanent=False, reasons=[]))
            d["n"] += 1
            d["permanent"] = d["permanent"] or bool(r.get("permanent"))
            d["reasons"].append(str(r.get("reason", "")))
        return out

    def _given_up(self) -> Dict[str, dict]:
        m = self._max_failures()
        return {k: v for k, v in self._known_failed().items() if v["permanent"] or v["n"] >= m}

    def _record_failures(self) -> None:
        """One line per stamp that failed in this attempt; then the given-up set is re-read."""
        if not self.failures:
            return
        p = self._failed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            for stamp, d in sorted(self.failures.items()):
                fh.write(json.dumps(dict(stamp=stamp, reason="; ".join(d["reasons"]), permanent=d["permanent"],
                                         batch=self.batch, host=os.uname().nodename, t=time.time())) + "\n")
        self.given_up = self._given_up()

    _budget = None

    def _start_budget(self, t0: float) -> None:
        """hour_source=memory: the worker stops itself (killing its daemon, engine and prep processes)
        v3.budget_margin_s before batch_timeout_seconds, so no orphan keeps writing after the
        orchestrator gave up and dispatched a retry; the hour gets what is left of the budget."""
        bt = getattr(self.pl, "batch_timeout_seconds", None)
        if not (self.memory and bt):
            return
        left = float(bt) - float(self.v3.budget_margin_s)
        self.deadline = t0 + left
        self._budget = threading.Timer(max(1.0, left), self._budget_expired)
        self._budget.daemon = True
        self._budget.start()

    def _budget_expired(self) -> None:
        log.critical("V3_BUDGET %sMHz batch%d: batch_timeout_seconds - v3.budget_margin_s reached: stopping "
                     "(daemon, engine, prep processes killed)", self.band, self.batch)
        self._budget_fired = True
        if self.engine is not None:
            self.engine.failed = "worker budget expired"       # no restart once the kill has started
        for kill in (lambda: self.daemon._p.kill() if self.daemon is not None else None,
                     lambda: self.engine._kill("worker budget") if self.engine is not None else None):
            try:
                kill()
            except Exception:
                pass
        pool = getattr(self.w, "_v3_pool", None)
        for p in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                p.kill()
            except Exception:
                pass
        try:
            from . import memory
            memory.unreserve()
            if self.ring is not None:
                self.ring.close()
        finally:
            os._exit(124)

    def _plan_shm(self) -> None:
        """Untar into tmpfs only if it has room next to the ring: (lookahead + 2) slots + lookahead raw
        MSes (+ the raw MSes kept for the kept-MS writer), else untar to work_root (V3_SHM_TIGHT)."""
        root = self.w._untar_root()
        if root is None:
            return
        try:
            tar_b = Path(self.tars[0]).stat().st_size
            free = shutil.disk_usage(str(root.parent if root.parent.exists() else "/dev/shm")).free
            shm_free = shutil.disk_usage("/dev/shm").free
        except OSError:
            return
        n_raw = self.lookahead + (2 if self.write_kept else 0)
        need_untar = n_raw * tar_b * self.w.UNTAR_SPACE_MARGIN
        need_slots = (self.lookahead + 2) * 130e6
        same = os.stat(str(root.parent) if root.parent.exists() else "/dev/shm").st_dev == os.stat("/dev/shm").st_dev
        if shm_free < need_slots * 1.2 + (need_untar if same else 0) or free < need_untar:
            self.w._untar_disabled = True
            log.warning("V3_SHM_TIGHT %sMHz batch%d: %.1f GB free in tmpfs for %.1f GB of slots + raw MSes: untar to "
                        "work_root", self.band, self.batch, min(shm_free, free) / 1e9, (need_slots + need_untar) / 1e9)

    def _memory_tight(self) -> None:
        """MemAvailable below the reserve (checked every v3.mem_check_every snapshots, acted on each time):
        1st spill the store, 2nd lookahead 2 and untar to work_root, then only logged."""
        self.n_guard += 1
        if self.n_guard == 1 and self.records and self.plan is not None and self.plan.backend == "ram":
            self._spill_now("MemAvailable below v3.ram_reserve_gb")
            return
        if not getattr(self.w, "_untar_disabled", False) or self.lookahead > 2:
            self.lookahead = min(self.lookahead, 2)
            self.w._untar_disabled = True
            log.warning("V3_MEM_SPILL %sMHz batch%d: MemAvailable below the reserve: lookahead 2, untar to "
                        "work_root", self.band, self.batch)
            return
        log.warning("V3_MEM_TIGHT %sMHz batch%d: MemAvailable still below v3.ram_reserve_gb (check %d)",
                    self.band, self.batch, self.n_guard)

    # -- feeders -----------------------------------------------------------------------------------
    def _feed(self, idx: int, tar: str, slot: Path, replay: bool) -> Optional[Item]:
        from . import fastprep
        w = self.w
        stamp = _stamp(tar)
        d = w.work / stamp
        udir = w._untar_dir(stamp, configured=True)
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
                self._note_skipped(stamp, st["nonzero_fraction"])
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
            with self._lock:
                self.n_prep_fail += 1
            self._fail(stamp, f"prep: {e}", permanent=not Path(tar).exists())     # a vanished tar never comes back
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
        udir = self.w._untar_dir(it.stamp, configured=True)     # also after untar to tmpfs was turned off
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
                    record_phase_centre=self.record_phase_centre(it.stamp),
                    # v3.profile_no_image: profiling only (the peel without the engine's GPU imaging)
                    image=not it.replay and not self.v3.profile_no_image,
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
        kept_box: list = []
        try:
            try:
                r = self._engine_image(it, req, lambda: kept_box.append(self._write_kept(it)))
            finally:
                it.imaged.set()
            kept_tmp = kept_box[0] if kept_box else None
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
            if not it.replay and req["image"]:
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

    def _write_kept(self, it: Item) -> Optional[Path]:
        """The kept MS from the peeled slot (v3.write_kept_ms), in a pool process; None if not wanted."""
        from . import fastprep
        w = self.w
        if not (self.write_kept and it.raw_ms is not None) or \
                (it.replay and w.kept_ms_path(it.stamp, w.keep_root()).exists()):
            return None
        t1 = time.perf_counter()
        kept_tmp = w.work / it.stamp / f"{it.stamp}_kept.ms"
        try:
            w._v3_run(fastprep.kept_from_slot_task, str(w._v3_template), str(it.raw_ms), str(it.slot),
                      str(kept_tmp))
        except Exception as e:
            log.error("kept MS %s failed: %s", it.stamp, e)
            if self.v3.hour_source == "kept_ms":
                raise
            return None
        log.info("timing %sMHz batch%d stamp=%s phase=kept_write elapsed=%.2f s", self.band, self.batch, it.stamp,
                 time.perf_counter() - t1)
        return kept_tmp

    def _engine_image(self, it: Item, req: dict, while_waiting) -> dict:
        """image_slot with the culprit rule: a request that dies with others in flight is retried
        alone (exclusive) -- the death need not be its own; one that dies alone is the culprit and
        fails.  The engine failed for good (restarts exhausted, records lost) -> the batch aborts."""
        from . import engine_client as ec
        try:
            with self.gate.shared():
                h = self.engine.submit("image_slot", **req)
                while_waiting()
                return h.result(self.v3.engine_timeout_s)
        except ec.EngineFailed as e:
            self._abort(f"engine failed: {e}")
            raise
        except (ec.EngineDied, ec.EngineTimeout) as e:
            if not self.records or self.store_path is not None or not self.memory:
                log.error("V3_ENGINE %sMHz stamp=%s: %s; retrying alone on the restarted engine", self.band,
                          it.stamp, e)
            else:                                # no record file: the restart lost the band-hour's records
                self._abort(f"engine died without a checkpoint file ({e}): the records are lost")
                raise BatchAborted(self.fatal) from e
        with self.gate.exclusive():
            try:
                return self.engine.request("image_slot", timeout=self.v3.engine_timeout_s, **req)
            except ec.EngineFailed as e:
                self._abort(f"engine failed: {e}")
                raise
            except (ec.EngineDied, ec.EngineTimeout) as e:
                log.error("V3_ENGINE_CULPRIT %sMHz stamp=%s: the engine died again with only this snapshot "
                          "in it: %s", self.band, it.stamp, e)
                raise

    def _abort(self, why: str) -> None:
        if self.fatal is None:
            self.fatal = why
            log.critical("V3_ABORT %sMHz batch%d: %s", self.band, self.batch, why)

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
        if self.memory and self._hour_done():
            log.info("batch %sMHz/%d: nothing to do (hour products done)", self.band, self.batch)
            self._delete_done_checkpoints()
            return
        skipped = self._known_skipped()
        self.given_up = self._given_up()
        tars = [t for t in self.tars if _stamp(t) not in skipped and _stamp(t) not in self.given_up]
        if len(tars) < len(self.tars):
            log.info("batch %sMHz/%d: %d input(s) skipped before (no data) and %d given up after failing are not "
                     "prepped again", self.band, self.batch, len([t for t in self.tars if _stamp(t) in skipped]),
                     len([t for t in self.tars if _stamp(t) in self.given_up]))
        have_fits = {t for t in tars if w._fits_path_for_stamp(_stamp(t)).exists()}
        todo = [t for t in tars if t not in have_fits]
        if not todo and not self.memory:
            log.info("batch %sMHz/%d: nothing to do", self.band, self.batch)
            return
        self.mem_start = memory.mem_available()
        batch_start = time.perf_counter()
        self._start_budget(batch_start)
        try:
            self._prepare(tars, todo)
        except BaseException:
            if self._budget is not None:
                self._budget.cancel()
            memory.unreserve()
            raise
        self._run_main(tars, todo, have_fits, batch_start)

    def _prepare(self, tars, todo) -> None:
        w = self.w
        w.work.mkdir(parents=True, exist_ok=True)
        self._plan_store(len(tars))
        self._plan_shm()
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

    def _run_main(self, tars, todo, have_fits, batch_start) -> None:
        from . import memory, slots
        w = self.w
        hour_ok = False
        self._qa_written = False
        try:
            self.daemon = w._start_daemon()          # Julia JIT overlaps the engine start and the prep
            self.engine = self._start_engine()
            have_rec = set(self._open_store(self.engine).get("stems", []))
            replay = [t for t in tars if t in have_fits and self.records and
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
            if self.v3.profile_no_image:            # profiling only: never products, never "done"
                raise RuntimeError("V3_PROFILE batch (v3.profile_no_image): no FITS, no hour products")
            if self.memory:
                # never an hour without a snapshot that could have been imaged: a failed one (prep, peel,
                # engine) fails the batch, and the orchestrator's retry redoes just the missing ones; one
                # that failed in v3.max_input_failures attempts (or whose tar vanished) is given up and
                # the hour is made from the rest (V3_HOUR_REDUCED, listed in provenance)
                self._record_failures()
                pending = sorted(s for s in self.failures if s not in self.given_up)
                if pending:
                    raise RuntimeError(f"V3_INCOMPLETE batch {self.band}MHz/{self.batch}: {len(pending)} snapshot(s) "
                                       f"failed ({self.n_fail} after prep, {self.n_prep_fail} prep error(s)): no hour "
                                       f"products (retry; given up after {self._max_failures()} attempt(s)): "
                                       f"{pending[:5]}")
                if self.given_up:
                    log.error("V3_HOUR_REDUCED %sMHz batch%d: the hour is made without %d input(s) that failed in "
                              "every attempt: %s", self.band, self.batch, len(self.given_up),
                              {k: v["reasons"][-1][:120] for k, v in list(self.given_up.items())[:5]})
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
            if self._budget is not None:
                self._budget.cancel()
            if self.engine is not None and self.engine.alive:     # never restart an engine just for info
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
            root = w._untar_root(configured=True)
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
        posts: List[tuple] = []
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
            n_zested = 0
            prev: Optional[Item] = None
            for i in range(n):
                if self.fatal:
                    for f in futs.values():
                        f.cancel()
                    for f in futs.values():         # running feeders: wait, give their slots back
                        if not f.cancelled():
                            try:
                                x = f.result()
                                if x is not None:
                                    self._release(x.slot)
                                    self._cleanup_item(x)
                            except Exception:
                                pass
                    for _, f in posts:
                        try:
                            f.result()
                        except Exception:
                            pass
                    raise BatchAborted(self.fatal)
                top_up(i)
                it = futs.pop(i).result()
                top_up(i)
                if it is None:
                    continue
                if recycle and n_zested and n_zested % recycle == 0:
                    log.info("batch %sMHz/%d: recycling zest daemon after %d files", self.band, self.batch, n_zested)
                    self.daemon.close()
                    self.daemon = self.w._start_daemon()
                if not self.v3.image_overlap and prev is not None:     # no GPU overlap with the engine
                    t0 = time.perf_counter()
                    prev.imaged.wait(self.v3.engine_timeout_s)
                    log.info("timing %sMHz batch%d stamp=%s phase=image_gate_wait elapsed=%.3f s", self.band,
                             self.batch, it.stamp, time.perf_counter() - t0)
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
                    self._fail(it.stamp, f"peel: {e}")
                    continue
                posts.append((it.stamp, post.submit(self._post, it)))
                prev = it
                if guard.check():
                    self._memory_tight()
            for stamp, f in posts:
                try:
                    f.result()
                    self.n_ok += 1
                except Exception as e:
                    permanent = _permanent_engine_error(e)
                    if permanent:
                        log.error("V3_INPUT_PERMANENT %sMHz stamp=%s: the imager rejected the input data (not "
                                  "retried; the hour is made without it): %s", self.band, stamp, e)
                    else:
                        log.error("image/archive failed: %s", e)
                    self.n_fail += 1
                    if not isinstance(e, BatchAborted) and not self.fatal:
                        self._fail(stamp, f"engine/archive: {e}", permanent=permanent)
            if self.fatal:
                raise BatchAborted(self.fatal)

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
            path = self._ck_name(d)
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
        from . import engine_client
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
        # never silently: every input is imaged (CLEAN FITS), skipped for no data, or given up after failing
        skipped = self._known_skipped()
        inputs = {self.kept_stem(_stamp(t)) for t in self.tars
                  if _stamp(t) not in skipped and _stamp(t) not in self.given_up}
        unaccounted = sorted(inputs - set(expected))
        if unaccounted:
            raise RuntimeError(f"V3_UNACCOUNTED hour {self.band}MHz {group}: {len(unaccounted)} input(s) neither imaged "
                               f"nor skipped nor given up: {unaccounted[:5]}")
        if self.given_up:
            rec["failed_inputs"] = {self.kept_stem(k): dict(attempts=v["n"], permanent=v["permanent"],
                                                            reason=v["reasons"][-1][:300])
                                    for k, v in sorted(self.given_up.items())}
            rec["hour_reduced"] = True
        rec["n_skipped_inputs"] = len([t for t in self.tars if _stamp(t) in skipped])
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
            hto = h.timeout_seconds
            if self.deadline is not None:            # what is left of the batch budget
                left = self.deadline - time.perf_counter() - 30.0
                hto = left if hto is None else min(float(hto), left)
                if hto <= 0:
                    raise RuntimeError("no batch budget left for the hour stage")
            hq = dict(argv=argv, expected=expected, log=str(qa_dir / f"{self.band}MHz-ttcalx-image.log"))
            try:
                r = self.engine.request("hour", timeout=hto, **hq)
            except engine_client.EngineFailed as e:
                raise RuntimeError(f"hour stage: the engine cannot be restarted ({e}; restarts are exhausted or the "
                                   f"worker budget expired): no hour retry") from e
            except (engine_client.EngineError, engine_client.EngineDied) as e:
                # e.g. a GPU allocation failure or a CUDA fatal error in a long-lived engine: once more in a fresh
                # one (records reloaded from the checkpoint), as v2's separate hour job would start -- not the
                # deterministic "no record" refusal, not without a checkpoint, not without budget
                if isinstance(e, engine_client.EngineError) and (e.reply or {}).get("missing"):
                    raise
                if self.store_path is None:
                    raise
                left = None if self.deadline is None else self.deadline - time.perf_counter() - 30.0
                if left is not None and left < max(300.0, 0.5 * (time.perf_counter() - t0)):
                    raise RuntimeError(f"hour stage failed ({e}) and only {left:.0f} s of the batch budget are left: "
                                       f"no retry") from e
                if self.engine.failed or self.engine.restarts >= self.engine.max_restarts:
                    raise RuntimeError(f"hour stage failed ({e}) and the engine cannot be restarted (restarts "
                                       f"exhausted earlier: {self.engine.restarts}/{self.engine.max_restarts}"
                                       f"{'; ' + self.engine.failed if self.engine.failed else ''}): no retry") from e
                log.error("V3_HOUR_RETRY %sMHz %s: %s; again in a fresh engine", self.band, group, e)
                rec["first_hour_error"] = str(e)
                self.engine.restart("fresh engine for the hour stage")
                if left is not None:
                    hto = left if hto is None else min(hto, left)
                try:
                    r = self.engine.request("hour", timeout=hto, **hq)
                except engine_client.EngineFailed as e2:
                    raise RuntimeError(f"hour stage retry: the engine cannot be restarted ({e2})") from e2
            imaged = set(r.get("qa_stems") or [])
            rec["imager_timing_s"] = r.get("timing", {})
            rec["n_loaded"] = r.get("n_loaded")
            rec["n_rejected_hour"] = len(r.get("rejected") or [])
            rec["phase_centre_deg"] = r.get("phase_centre")
            if r.get("cache_fallback"):      # imager: a GPU OOM, imaged again without the device chunk cache
                rec["hour_cache_fallback"] = r["cache_fallback"]
                log.warning("hour %sMHz %s: GPU OOM fallback (no device chunk cache) for %s", self.band, group,
                            [os.path.basename(x.get("name", "")) for x in r["cache_fallback"]])
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
        if self.engine.alive:                        # an engine that died in the hour is not restarted for this
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
