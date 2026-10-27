"""Schedule batches on load-aware GPU slots and stitch completed bands.

With ``stream_hour_images`` (layout=lst) each band-hour's hour job is a ledger row
(batch id ``hour_job_id(group)``) dispatched as soon as that band-hour's batches are
done; with ``keep_ms_local`` a band-hour's batches and hour job stay on one node, whose
NVMe holds its peeled MSes.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import dispatch, worklist
from .config import Config, Slot
from .ledger import DONE, FAILED, QUEUED, SKIPPED, Job, Ledger
from .metadata import write_run_metadata

log = logging.getLogger("gsi.orchestrator")

HOUR_JOB_BASE = -1000       # ledger batch id of a band-hour's hour job: -1000 - LST hour


def hour_job_id(group: str) -> int:
    return HOUR_JOB_BASE - int(group.rstrip("h"))


def is_hour_job(j: Job) -> bool:
    return j.batch <= HOUR_JOB_BASE


def hour_job_group(batch: int) -> str:
    from . import lst
    return lst.label(HOUR_JOB_BASE - batch)


def _run_key(pl) -> str:
    return getattr(pl, "run_key", pl.date)


class Orchestrator:
    def __init__(self, cfg: Config, config_dir: str, worker_sh: str,
                 list_via_node: Optional[str] = None):
        self.cfg = cfg
        self.config_dir = config_dir
        self.worker_sh = worker_sh
        self.slots = cfg.cluster.slots()
        self.list_node = list_via_node or self.slots[0].node
        self.state = Path(cfg.cluster.state_dir)
        self.ledger = Ledger(self.state / f"ledger_{_run_key(cfg.pipeline)}.sqlite")
        self._lock = threading.Lock()
        runtime = cfg.pipeline.env
        self.worker_env = {
            "GSI_RUNTIME_ENV": runtime.conda_env,
            "GSI_DEV_ENV": runtime.dev_env,
            "GSI_JULIA_BIN": runtime.julia_bin,
            "GSI_TTCALX_DIR": runtime.ttcalx,
            "GSI_SHARED_DEPOT": runtime.shared_depot,
        }
        for key in ("GSI_CONDA_SH", "GSI_ALLOWED_LUSTRE_WRITE_ROOT",
                    "GSI_TASKSET_CPUS", "GSI_NICE", "GSI_IONICE",
                    "NUMPY_MADVISE_HUGEPAGE", "CASARCFILES"):
            if key in os.environ:
                self.worker_env[key] = os.environ[key]
        self.group_of: Dict[Tuple[int, int], str] = {}   # (band, batch) -> LST hour label
        self._wideband_done: set = set()
        self._wideband_pool: Optional[ThreadPoolExecutor] = None

    @property
    def streaming(self) -> bool:
        pl = self.cfg.pipeline
        return (getattr(pl, "stream_hour_images", False) and
                getattr(pl, "layout", "utc") == "lst")

    # -- planning ------------------------------------------------------------

    def plan(self) -> List[Job]:
        log.info("planning worklist for date %s via %s", self.cfg.pipeline.date,
                 self.list_node)
        batches = worklist.build(self.cfg.pipeline, self.list_node,
                                 self.state / "manifests" / _run_key(self.cfg.pipeline))
        jobs = worklist.to_jobs(batches)
        self.group_of = {(b.band, b.index): b.group for b in batches}
        if self.streaming:
            n_batches: Dict[Tuple[int, str], int] = {}
            for b in batches:
                n_batches[(b.band, b.group)] = n_batches.get((b.band, b.group), 0) + 1
            jobs += [Job(band, hour_job_id(group), n)
                     for (band, group), n in sorted(n_batches.items())]
        log.info("built %d batches (%d jobs) for planning", len(batches), len(jobs))
        self.ledger.seed(jobs)
        n = self.ledger.reset_stale_running()
        if n:
            log.info("reset %d stale running job(s) for re-dispatch", n)
        pending = self.ledger.pending()
        log.info("pending jobs after planning: %d", len(pending))
        return pending

    def dry_run(self) -> None:
        jobs = self.plan()
        print(f"slots ({len(self.slots)}): " +
              ", ".join(str(s) for s in self.slots))
        by_band: Dict[int, int] = {}
        for j in jobs:
            by_band[j.band] = by_band.get(j.band, 0) + 1
        print(f"pending jobs: {len(jobs)}")
        for band in sorted(by_band):
            print(f"  {band}MHz: {by_band[band]} batches")
        log.info("dry-run summary: %d pending jobs across %d bands",
                 len(jobs), len(by_band))

    # -- execution -----------------------------------------------------------

    def _manifest(self, j: Job) -> str:
        return str(self.state / "manifests" / _run_key(self.cfg.pipeline) /
                   f"{j.band}_{j.batch}.txt")

    def _run_one(self, slot: Slot, j: Job) -> bool:
        log.info("dispatching %sMHz batch%d to %s (%d files)",
                 j.band, j.batch, slot, j.n_files)
        r = dispatch.run_worker(slot.node, slot.gpu, self.worker_sh,
                               self.config_dir, j.band, j.batch, self._manifest(j),
                               timeout=self.cfg.pipeline.batch_timeout_seconds,
                               env=self.worker_env)
        with self._lock:
            if r.ok:
                self.ledger.mark_done(j)
            else:
                self.ledger.mark_failed(j, r.tail)
        log.info("%s %sMHz batch%d %s", slot, j.band, j.batch,
                 "done" if r.ok else f"FAILED rc={r.rc}")
        if not r.ok:
            log.warning("worker output tail for %sMHz batch%d:\n%s",
                        j.band, j.batch, r.tail)
        return r.ok

    def run(self, poll: float = 2.0) -> None:
        pending0 = self.ledger.pending(include_running=False)
        log.info("run: %d pending jobs, %d slots (budget=%d free cores)",
                 len(pending0), len(self.slots), self.cfg.cluster.free_core_budget)
        free: List[Slot] = list(self.slots)
        running: Dict[Future, Slot] = {}
        pool = ThreadPoolExecutor(max_workers=len(self.slots) or 1)
        self._write_metadata("running")
        final_status = "completed"
        error = None
        try:
            if self.streaming:
                self._run_streaming(pool, poll)
            while not self.streaming:
                pending = self.ledger.pending(include_running=False)
                if not pending and not running:
                    break
                while free and pending:
                    slot = self._pick_slot(free)
                    if slot is None:
                        break
                    j = pending.pop(0)
                    if j.attempts >= self.cfg.pipeline.max_retries + 1:
                        with self._lock:
                            self.ledger.mark_skipped(j, "exhausted retries")
                        log.warning("skipping %sMHz batch%d after %d attempt(s)",
                                    j.band, j.batch, j.attempts)
                        continue
                    free.remove(slot)
                    # Claim before submitting so the next scheduler poll cannot
                    # select this batch again while its thread is starting.
                    with self._lock:
                        self.ledger.mark_running(j, str(slot))
                    fut = pool.submit(self._run_one, slot, j)
                    running[fut] = slot
                done = [f for f in running if f.done()]
                for f in done:
                    slot = running.pop(f)
                    free.append(slot)
                    exc = f.exception()
                    if exc is not None:
                        log.error("dispatch thread on %s crashed: %r", slot, exc)
                if not done:
                    time.sleep(poll)
            if self.streaming:
                pass                       # hour jobs ran as their band-hours finished
            elif getattr(self.cfg.pipeline, "layout", "utc") == "lst":
                self._hour_step()          # QA, 1-h / 10-min images, movies, provenance
            else:
                self._stitch_movies()
            if getattr(self.cfg.pipeline, "calcache_cleanup", "run") == "run":
                self._cleanup_calcache()
            self._cleanup_manifests()
        except BaseException as exc:
            final_status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            pool.shutdown(wait=True)
            if getattr(self, "_wideband_pool", None) is not None:
                self._wideband_pool.shutdown(wait=True)
            self._write_metadata(final_status, error)
            self.ledger.close()

    # -- streaming (stream_hour_images) ---------------------------------------

    def _group(self, j: Job) -> str:
        return hour_job_group(j.batch) if is_hour_job(j) else self.group_of.get((j.band, j.batch), "")

    def _band_hours(self, rows: List[Job]) -> Dict[Tuple[int, str], dict]:
        """(band, group) -> its batches, hour job and node (where its batches ran)."""
        st: Dict[Tuple[int, str], dict] = {}
        for j in rows:
            e = st.setdefault((j.band, self._group(j)), {"batches": [], "hour": None, "node": None})
            if is_hour_job(j):
                e["hour"] = j
            else:
                e["batches"].append(j)
                if j.slot and not e["node"]:
                    e["node"] = j.slot.split(":")[0]
        return st

    def _choose(self, slot: Slot, pending: List[Job], st: Dict[Tuple[int, str], dict],
                local: bool, order: Dict[str, int]) -> Optional[Job]:
        """A ready hour job for this slot, else a batch (this node's band-hours first)."""
        nodes = {s.node for s in self.slots}

        def node_ok(e: dict) -> bool:     # a node no longer in the cluster frees its band-hours
            return not local or e["node"] in (None, slot.node) or e["node"] not in nodes

        for j in pending:
            if is_hour_job(j):
                e = st[(j.band, self._group(j))]
                if node_ok(e) and all(b.status in (DONE, SKIPPED) for b in e["batches"]):
                    return j
        cands = [j for j in pending if not is_hour_job(j)
                 and node_ok(st[(j.band, self._group(j))])]
        if not cands:
            return None
        return min(cands, key=lambda j: (st[(j.band, self._group(j))]["node"] != slot.node,
                                         order.get(self._group(j), 99), j.band, j.batch))

    def _run_streaming(self, pool: ThreadPoolExecutor, poll: float) -> None:
        pl = self.cfg.pipeline
        local = bool(getattr(pl, "keep_ms_local", None))
        from . import lst
        order = {lst.label(h): i for i, h in enumerate(pl.lst_hours)}
        budget = self.cfg.cluster.free_core_budget
        free: List[Slot] = list(self.slots)
        running: Dict[Future, Slot] = {}
        while True:
            rows = self.ledger.all()
            st = self._band_hours(rows)
            pending = []
            for j in rows:
                if j.status not in (QUEUED, FAILED):
                    continue
                if j.attempts >= pl.max_retries + 1:
                    with self._lock:
                        self.ledger.mark_skipped(j, "exhausted retries")
                    log.warning("skipping %sMHz %s after %d attempt(s)", j.band,
                                f"hour {self._group(j)}" if is_hour_job(j) else f"batch{j.batch}",
                                j.attempts)
                    continue
                if is_hour_job(j) and all(b.status == SKIPPED for b in
                                          st[(j.band, self._group(j))]["batches"]):
                    with self._lock:
                        self.ledger.mark_skipped(j, "no snapshots")
                    continue
                pending.append(j)
            if not pending and not running:
                break
            cores: Dict[str, bool] = {}
            for slot in list(free):
                j = self._choose(slot, pending, st, local, order)
                if j is None:
                    continue
                if slot.node not in cores:
                    fc = dispatch.node_free_cores(slot.node)
                    cores[slot.node] = fc is None or fc >= budget
                if not cores[slot.node]:
                    continue
                free.remove(slot)
                pending.remove(j)
                with self._lock:
                    self.ledger.mark_running(j, str(slot))
                j.status = "running"
                e = st[(j.band, self._group(j))]
                e["node"] = e["node"] or slot.node
                fn = self._run_hour_job if is_hour_job(j) else self._run_one
                running[pool.submit(fn, slot, j)] = slot
            done = [f for f in running if f.done()]
            for f in done:
                slot = running.pop(f)
                free.append(slot)
                exc = f.exception()
                if exc is not None:
                    log.error("dispatch thread on %s crashed: %r", slot, exc)
            if done:
                self._maybe_wideband()
            else:
                time.sleep(poll)
        self._maybe_wideband()
        n = self.ledger.summary()
        log.info("streaming run finished: %s", n)

    def _run_hour_job(self, slot: Slot, j: Job) -> bool:
        from . import hour_image
        group = hour_job_group(j.batch)
        t0 = time.perf_counter()
        log.info("hour-image %sMHz %s -> %s", j.band, group, slot)
        r = dispatch.run_worker(slot.node, slot.gpu, self.worker_sh, self.config_dir,
                                j.band, j.batch, f"{hour_image.MANIFEST_PREFIX}{group}",
                                timeout=self.cfg.pipeline.hour_imaging.timeout_seconds,
                                env=self.worker_env)
        with self._lock:
            if r.ok:
                self.ledger.mark_done(j)
            else:
                self.ledger.mark_failed(j, r.tail)
        log.info("hour-image %sMHz %s on %s %s in %.1f s", j.band, group, slot,
                 "done" if r.ok else f"FAILED rc={r.rc}", time.perf_counter() - t0)
        if not r.ok:
            log.warning("hour-image tail %sMHz %s:\n%s", j.band, group, r.tail)
        return r.ok

    def _maybe_wideband(self) -> None:
        """wideband_per_hour: queue an LST hour's wideband summary once all its hour jobs ended."""
        pl = self.cfg.pipeline
        if not getattr(pl, "wideband_per_hour", False):
            return
        by_group: Dict[str, List[Job]] = {}
        for j in self.ledger.all():
            if is_hour_job(j):
                by_group.setdefault(hour_job_group(j.batch), []).append(j)
        for group, js in sorted(by_group.items()):
            if group in self._wideband_done or not all(j.status in (DONE, SKIPPED) for j in js):
                continue
            self._wideband_done.add(group)
            if self._wideband_pool is None:
                self._wideband_pool = ThreadPoolExecutor(max_workers=1)
            self._wideband_pool.submit(self._wideband, group)

    def _wideband(self, group: str) -> None:
        """pipeline/wideband_summary.py on <output_root>/<LST>h/<date>/<run_label> (control host)."""
        pl = self.cfg.pipeline
        script = Path(__file__).resolve().parents[3] / "pipeline" / "wideband_summary.py"
        run_dir = Path(pl.output_root) / group / pl.date / pl.run_label
        if not script.exists() or not run_dir.is_dir():
            log.warning("wideband %s skipped: %s or %s missing", group, script, run_dir)
            return
        log_path = Path(pl.output_root) / "run_logs" / f"wideband_{pl.date}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            with open(log_path, "a") as fh:
                rc = subprocess.run([sys.executable, str(script), str(run_dir)], stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
        except Exception as e:
            log.error("wideband summary %s failed: %s", run_dir, e)
            return
        log.info("wideband summary %s rc=%s in %.1f s", run_dir, rc, time.perf_counter() - t0)

    def cleanup_nodes(self) -> None:
        """After the night: drop node calibration caches and this date's node-local kept
        MSes; MSes left by failed hour jobs move to keep_ms_root (Lustre) for reruns."""
        pl = self.cfg.pipeline
        cache = str(Path(self.cfg.cluster.work_root) / "_calcache")
        cmd = f"rm -rf {shlex.quote(cache)}"
        local = getattr(pl, "keep_ms_local", None)
        if local:
            date = shlex.quote(pl.date)
            move = (f"mkdir -p {shlex.quote(pl.keep_ms_root)}/\"$d\" && "
                    f"mv \"$d\"/*.ms {shlex.quote(pl.keep_ms_root)}/\"$d\"/ && "
                    "echo \"moved $n MS from $d\"") if pl.keep_ms_root else "true"
            cmd += (f"; cd {shlex.quote(local)} 2>/dev/null && {{ for d in */{date}/*; do "
                    "[ -d \"$d\" ] || continue; n=$(ls -d \"$d\"/*.ms 2>/dev/null | wc -l); "
                    f"[ \"$n\" -gt 0 ] && {{ {move}; }}; done; rm -rf */{date}; }}; true")
        for node in sorted({s.node for s in self.slots}):
            r = dispatch._ssh(node, cmd, timeout=3600)
            log.info("node cleanup on %s: %s %s", node, "ok" if r.ok else "FAILED", r.tail.strip())

    def _write_metadata(self, status: str,
                        error: Optional[str] = None) -> None:
        try:
            path = write_run_metadata(
                self.cfg, self.config_dir, status,
                summary=self.ledger.summary(), error=error)
            log.info("run metadata: %s", path)
        except Exception as exc:
            log.warning("run metadata write failed: %s", exc)

    def _cleanup_calcache(self) -> None:
        """Remove calibration tables cached on worker NVMe."""
        import shlex
        cache = str(Path(self.cfg.cluster.work_root) / "_calcache")
        for node in sorted({s.node for s in self.slots}):
            r = dispatch._ssh(node, f"rm -rf {shlex.quote(cache)}", timeout=30)
            log.info("calcache cleanup on %s: %s", node, "ok" if r.ok else "FAILED")

    def _cleanup_manifests(self) -> None:
        """Remove this date's manifests after all workers finish."""
        manifest_dir = self.state / "manifests" / _run_key(self.cfg.pipeline)
        try:
            shutil.rmtree(manifest_dir, ignore_errors=True)
            log.info("deleted manifest directory %s", manifest_dir)
        except Exception as e:
            log.warning("manifest cleanup failed for %s: %s", manifest_dir, e)

    def _pick_slot(self, free: List[Slot]) -> Optional[Slot]:
        """Return the first slot on a node with enough free cores."""
        budget = self.cfg.cluster.free_core_budget
        for slot in free:
            fc = dispatch.node_free_cores(slot.node)
            if fc is None or fc >= budget:
                return slot
        return None

    def _hour_step(self) -> None:
        """layout=lst: one job per band and LST hour (hour_lst.run_band) on the GPU slots."""
        from . import hour_image
        t0 = time.perf_counter()
        res = hour_image.run_all(self.cfg, self.config_dir, self.worker_sh, self.worker_env)
        failed = sorted(k for k, ok in res.items() if not ok)
        log.info("hour step: %d ok, %d failed %s in %.1f s", len(res) - len(failed),
                 len(failed), failed, time.perf_counter() - t0)

    def _stitch_movies(self) -> None:
        """Build hourly movies for bands whose batches all finished."""
        if not self.cfg.pipeline.movie.enabled:
            log.info("movie stitching disabled")
            return
        done_bands = self._bands_complete()
        for band in sorted(done_bands):
            slot = self.slots[0]
            log.info("stitching hourly movies for %sMHz on %s", band, slot)
            r = dispatch.run_worker(slot.node, slot.gpu, self.worker_sh,
                                   self.config_dir, band, -1, "STITCH",
                                   timeout=self.cfg.pipeline.batch_timeout_seconds,
                                   env=self.worker_env)
            if not r.ok:
                log.warning("movie stitch failed for %sMHz rc=%s; continuing:\n%s",
                            band, r.rc, r.tail)

    def _bands_complete(self) -> List[int]:
        by_band: Dict[int, List[Job]] = {}
        for j in self.ledger.all():
            by_band.setdefault(j.band, []).append(j)
        return [b for b, js in by_band.items()
                if js and all(x.status in {DONE, SKIPPED} for x in js)]
