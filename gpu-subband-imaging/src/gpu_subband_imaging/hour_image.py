"""1-h / 10-min imaging of kept peeled MSes with ``ttcalx-image hour``.

Needs ``keep_ms_root`` (the snapshot run moves each peeled MS there).  The
orchestrator side (:func:`run_all`) sends one band-hour per GPU slot through the
usual worker launcher (manifest ``HOUR_IMAGE:<hh>``); the worker side
(:func:`run_band`) lists the band-hour's MSes, runs ``ttcalx-image hour`` and writes

  <output_root>/<band>MHz/<date>/<hour>/hour_images/<band>MHz-<product>[-tNNNN]-*.fits
  <output_root>/<band>MHz/<date>/<hour>/hour_images/<band>MHz-summary.json   (imager)
  <output_root>/<band>MHz/<date>/<hour>/hour_images/<band>MHz-hour-timing.json
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .config import Config

log = logging.getLogger("gsi.hour_image")

MANIFEST_PREFIX = "HOUR_IMAGE:"


def kept_dir(cfg: Config, band: int, hour: str, root: Optional[str] = None) -> Path:
    return (Path(root or cfg.pipeline.keep_ms_root) / f"{band}MHz" / cfg.pipeline.date / hour)


def kept_dirs(cfg: Config, band: int, hour: str) -> List[Path]:
    """Node-local keep dir (keep_ms_local) first, then the shared one (keep_ms_root)."""
    roots = [getattr(cfg.pipeline, "keep_ms_local", None), cfg.pipeline.keep_ms_root]
    return [kept_dir(cfg, band, hour, r) for r in roots if r]


def out_dir(cfg: Config, band: int, hour: str) -> Path:
    return (Path(cfg.pipeline.output_root) / f"{band}MHz" / cfg.pipeline.date / hour /
            "hour_images")


def list_kept(cfg: Config, band: int, hour: str) -> List[str]:
    """Kept MSes of a band-hour from every keep dir, in time (file name) order."""
    found = [p for d in kept_dirs(cfg, band, hour) if d.is_dir() for p in d.iterdir()
             if p.is_dir() and p.name.endswith(".ms") and not p.name.startswith(".")]
    return [str(p) for p in sorted(found, key=lambda p: p.name)]


def build_cmd(cfg: Config, band: int, list_file: Path, out_prefix: Path) -> List[str]:
    h = cfg.pipeline.hour_imaging
    return ([h.ttcalx_image, "hour", f"@{list_file}", "--band", str(band),
             "--out", str(out_prefix), "--products", ",".join(h.products),
             "--qa", h.qa, "--nproc", str(h.nproc), "-v"] + list(h.extra_args))


def imager_env(cfg: Config, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Environment for the imager venv: no conda PYTHONPATH, no numpy THP stalls."""
    env = dict(os.environ if base is None else base)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.update(cfg.pipeline.hour_imaging.env)
    return env


def run_band(cfg: Config, band: int, hour: str, gpu: Optional[int] = None) -> Dict:
    """Image one band-hour on this node (CUDA_VISIBLE_DEVICES set by the launcher)."""
    if not cfg.pipeline.keep_ms_root:
        raise RuntimeError("hour imaging needs pipeline.keep_ms_root")
    if getattr(cfg.pipeline, "layout", "utc") == "lst":
        from . import hour_lst
        return hour_lst.run_band(cfg, band, hour, gpu)
    ms = list_kept(cfg, band, hour)
    od = out_dir(cfg, band, hour)
    od.mkdir(parents=True, exist_ok=True)
    timing_path = od / f"{band}MHz-hour-timing.json"
    rec: Dict = dict(band=band, date=cfg.pipeline.date, hour=hour, n_ms=len(ms),
                     host=os.uname().nodename, gpu=gpu,
                     products=list(cfg.pipeline.hour_imaging.products))
    if not ms:
        log.warning("hour-image %sMHz %s: no kept MSes under %s", band, hour,
                    kept_dir(cfg, band, hour))
        rec.update(status="no-data")
        timing_path.write_text(json.dumps(rec, indent=1))
        return rec
    list_file = od / f"{band}MHz-ms-list.txt"
    list_file.write_text("\n".join(ms) + "\n")
    prefix = od / f"{band}MHz"
    cmd = build_cmd(cfg, band, list_file, prefix)
    log.info("hour-image %sMHz %s: %d MSes: %s", band, hour, len(ms), " ".join(cmd))
    rec["cmd"] = cmd
    rec["start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()
    with open(od / f"{band}MHz-ttcalx-image.log", "w") as fh:
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=imager_env(cfg),
                               timeout=cfg.pipeline.hour_imaging.timeout_seconds)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    rec["wall_s"] = time.perf_counter() - t0
    rec["rc"] = rc
    rec["status"] = "ok" if rc == 0 else "failed"
    summary = Path(f"{prefix}-summary.json")
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            rec["imager_timing_s"] = s.get("timing", {})
            rec["n_loaded"] = s.get("n_loaded")
            rec["rejected"] = s.get("rejected")
            rec["files"] = {k: [f for o in v for f in o.get("files", [])]
                            for k, v in s.get("products", {}).items()}
        except ValueError as e:
            log.warning("cannot parse %s: %s", summary, e)
    timing_path.write_text(json.dumps(rec, indent=1, default=float))
    log.info("timing %sMHz hour=%s phase=hour_image elapsed=%.2f s rc=%s n_ms=%d",
             band, hour, rec["wall_s"], rc, len(ms))
    if rc != 0:
        raise RuntimeError(f"ttcalx-image hour failed for {band}MHz {hour} (rc={rc})")
    return rec


def _hours(cfg: Config) -> List[str]:
    if getattr(cfg.pipeline, "layout", "utc") == "lst":
        from . import lst
        return [lst.label(h) for h in cfg.pipeline.lst_hours]
    if cfg.pipeline.hours == "all":
        root = Path(cfg.pipeline.keep_ms_root)
        found = set()
        for band in cfg.pipeline.bands:
            d = root / f"{band}MHz" / cfg.pipeline.date
            if d.is_dir():
                found |= {p.name for p in d.iterdir() if p.is_dir() and p.name.isdigit()}
        return sorted(found)
    return [str(h).zfill(2) for h in cfg.pipeline.hours]


def plan(cfg: Config, bands: Optional[List[int]] = None,
         hours: Optional[List[str]] = None) -> List[Tuple[int, str]]:
    """Band-hours with kept MSes, largest images first (they take longest)."""
    hrs = hours or _hours(cfg)
    jobs = [(b, h) for b in (bands or cfg.pipeline.bands) for h in hrs
            if list_kept(cfg, b, h)]
    return sorted(jobs, key=lambda j: (-cfg.subbands.geom(j[0]).pixels, j[0], j[1]))


def run_all(cfg: Config, config_dir: str, worker_sh: str, env: Mapping[str, str],
            bands: Optional[List[int]] = None, hours: Optional[List[str]] = None) -> Dict:
    """Run every band-hour, one per GPU slot; return {(band, hour): ok}."""
    from . import dispatch
    jobs = plan(cfg, bands, hours)
    log.info("hour-image: %d band-hour(s) on %d slot(s): %s", len(jobs),
             len(cfg.cluster.slots()), jobs)
    q: "queue.Queue[Tuple[int, str]]" = queue.Queue()
    for j in jobs:
        q.put(j)
    results: Dict = {}
    lock = threading.Lock()
    timeout = cfg.pipeline.hour_imaging.timeout_seconds

    def slot_loop(slot) -> None:
        while True:
            try:
                band, hour = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            log.info("hour-image %sMHz %s -> %s", band, hour, slot)
            r = dispatch.run_worker(slot.node, slot.gpu, worker_sh, config_dir, band, -2,
                                    f"{MANIFEST_PREFIX}{hour}", timeout=timeout, env=env)
            with lock:
                results[(band, hour)] = r.ok
            log.info("hour-image %sMHz %s on %s %s in %.1f s", band, hour, slot,
                     "done" if r.ok else f"FAILED rc={r.rc}", time.perf_counter() - t0)
            if not r.ok:
                log.warning("hour-image tail %sMHz %s:\n%s", band, hour, r.tail)

    threads = [threading.Thread(target=slot_loop, args=(s,), daemon=True)
               for s in cfg.cluster.slots()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
