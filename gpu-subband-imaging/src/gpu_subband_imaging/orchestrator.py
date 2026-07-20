"""Schedule batches on load-aware GPU slots and stitch completed bands."""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional

from . import dispatch, worklist
from .config import Config, Slot
from .ledger import DONE, SKIPPED, Job, Ledger

log = logging.getLogger("gsi.orchestrator")


class Orchestrator:
    def __init__(self, cfg: Config, config_dir: str, worker_sh: str,
                 list_via_node: Optional[str] = None):
        self.cfg = cfg
        self.config_dir = config_dir
        self.worker_sh = worker_sh
        self.slots = cfg.cluster.slots()
        self.list_node = list_via_node or self.slots[0].node
        self.state = Path(cfg.cluster.state_dir)
        self.ledger = Ledger(self.state / f"ledger_{cfg.pipeline.date}.sqlite")
        self._lock = threading.Lock()
        runtime = cfg.pipeline.env
        self.worker_env = {
            "GSI_RUNTIME_ENV": runtime.conda_env,
            "GSI_DEV_ENV": runtime.dev_env,
            "GSI_JULIA_BIN": runtime.julia_bin,
            "GSI_TTCALX_DIR": runtime.ttcalx,
            "GSI_SHARED_DEPOT": runtime.shared_depot,
        }
        for key in ("GSI_CONDA_SH", "GSI_ALLOWED_LUSTRE_WRITE_ROOT"):
            if key in os.environ:
                self.worker_env[key] = os.environ[key]

    # -- planning ------------------------------------------------------------

    def plan(self) -> List[Job]:
        log.info("planning worklist for date %s via %s", self.cfg.pipeline.date,
                 self.list_node)
        batches = worklist.build(self.cfg.pipeline, self.list_node,
                                 self.state / "manifests" / self.cfg.pipeline.date)
        jobs = worklist.to_jobs(batches)
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
        return str(self.state / "manifests" / self.cfg.pipeline.date /
                   f"{j.band}_{j.batch}.txt")

    def _run_one(self, slot: Slot, j: Job) -> bool:
        with self._lock:
            self.ledger.mark_running(j, str(slot))
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
        try:
            while True:
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
            self._stitch_movies()
            self._cleanup_calcache()
            self._cleanup_manifests()
        finally:
            pool.shutdown(wait=True)
            self.ledger.close()

    def _cleanup_calcache(self) -> None:
        """Remove calibration tables cached on worker NVMe."""
        import shlex
        cache = str(Path(self.cfg.cluster.work_root) / "_calcache")
        for node in sorted({s.node for s in self.slots}):
            r = dispatch._ssh(node, f"rm -rf {shlex.quote(cache)}", timeout=30)
            log.info("calcache cleanup on %s: %s", node, "ok" if r.ok else "FAILED")

    def _cleanup_manifests(self) -> None:
        """Remove this date's manifests after all workers finish."""
        manifest_dir = self.state / "manifests" / self.cfg.pipeline.date
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
