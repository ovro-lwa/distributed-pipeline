"""Process one subband batch on a GPU worker.

CPU preparation, GPU peeling, and image archiving overlap. Existing FITS
products are skipped, and local working data is removed after each batch.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from . import lst
from .config import Config, v3_params
from .stages import (average, calibrate, compress, fastcal, flag, image, movie, quality,
                     stage_ms, uvw)
from .stages.peel import ZestDaemon

log = logging.getLogger("gsi.worker")


def _mjd(stamp: str) -> float:
    from astropy.time import Time
    s = stamp.split("_")
    iso = f"{s[0][:4]}-{s[0][4:6]}-{s[0][6:]}T{s[1][:2]}:{s[1][2:4]}:{s[1][4:]}"
    return float(Time(iso).mjd)


def _stamp(input_path: str) -> str:
    name = Path(input_path).name
    for suffix in (".ms.tar", ".ms"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError(f"unsupported measurement-set input: {input_path}")


def _hour(stamp: str) -> str:
    return stamp.split("_")[1][:2]


def _require_batch_outputs(band: int, batch: int, n_input: int, n_ok: int) -> None:
    if n_input and not n_ok:
        raise RuntimeError(
            f"batch {band}MHz/{batch} produced no outputs from {n_input} inputs"
        )


class Worker:
    def __init__(self, cfg: Config, band: int, batch: int, gpu: int):
        self.cfg, self.band, self.batch, self.gpu = cfg, band, batch, gpu
        pl = cfg.pipeline
        self.work = Path(cfg.cluster.work_root) / f"b{band}_{batch}"
        self.out = Path(pl.output_root) / f"{band}MHz" / pl.date
        self.geom = cfg.subbands.geom(band)
        self._badants = ""
        self._bp = pl.cal.bp
        self._xy = pl.cal.xy
        self._lst: dict = {}
        self._qa: List[dict] = []      # pilot V statistics of this batch (layout=lst)
        self._gpu_imager = None        # stages.gpu_image.SnapshotDaemon with imaging.imager=ttcalx
        # gsi v3 in-memory prep (v3.prep_inmem): spawn process pool and the batch's
        # mstransform template (v3.prep_inmem=on)
        self._v3_pool = None
        self._v3_lock = threading.Lock()
        self._v3_tpl_lock = threading.Lock()
        self._v3_template: Optional[Path] = None
        self._v3_template_failed = False
        self._archives: dict = {}      # v3.archive: group -> hourstore.ArchiveClient
        self._archive_lock = threading.Lock()
        self._archive_failed = False   # a record of this batch failed (logged loudly once)

    def _start_gpu_imager(self):
        from .stages import gpu_image
        p, h = self.cfg.pipeline.imaging, self.cfg.pipeline.hour_imaging
        python = p.gpu_python or str(Path(h.ttcalx_image).parent / "python")
        extra = {k: v for k, v in h.env.items() if k == "CUPY_CACHE_DIR"}
        extra.update(p.gpu_env)
        return gpu_image.start(python, extra_env=extra)

    # -- layout ------------------------------------------------------------------
    # layout=utc keys everything by the UTC hour of the stamp, layout=lst by its
    # LST hour ("HHh") under the main-pipeline tree (lst.py).

    @property
    def lst_layout(self) -> bool:
        return getattr(self.cfg.pipeline, "layout", "utc") == "lst"

    def lst_hour(self, stamp: str) -> int:
        if stamp not in self._lst:
            self._lst[stamp] = lst.lst_hours([stamp])[0]
        return self._lst[stamp]

    def group(self, stamp: str) -> str:
        return lst.label(self.lst_hour(stamp)) if self.lst_layout else _hour(stamp)

    def band_dir(self, stamp: str) -> Path:
        pl = self.cfg.pipeline
        return lst.band_dir(pl.output_root, self.lst_hour(stamp), pl.date, pl.run_label, self.band)

    def hour_dir(self, hour: str) -> Path:
        return self.out / hour

    def fits_dir(self, hour: str) -> Path:
        return self.hour_dir(hour) / "fits"

    def frame_dir(self, hour: str) -> Path:
        return self.hour_dir(hour) / "images"

    def _frame_dir_for_stamp(self, stamp: str) -> Path:
        if self.lst_layout:
            return self.band_dir(stamp) / "Movies" / "frames"
        return self.frame_dir(_hour(stamp))

    def _archive_paths(self, stamp: str) -> Tuple[Path, Path]:
        """Archive paths (uncompressed names) of the I and V snapshot images."""
        if self.lst_layout:
            d, ts = self.band_dir(stamp), lst.timestamp(stamp)
            return (d / "snapshots_clean" / lst.clean_name(self.band, ts),
                    d / "snapshots" / lst.pilot_name(self.band, ts))
        d = self.fits_dir(_hour(stamp))
        return d / f"img_{stamp}-I-image.fits", d / f"img_{stamp}-V-image.fits"

    def _fits_path_for_stamp(self, stamp: str) -> Path:
        return Path(f"{self._archive_paths(stamp)[0]}.fz")

    def _stage_cal(self, src: str) -> str:
        """Cache one calibration table on local NVMe using an atomic rename."""
        import os
        cache = Path(self.cfg.cluster.work_root) / "_calcache"
        dst = cache / Path(src).name
        if not dst.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = cache / f".{Path(src).name}.tmp{os.getpid()}"
            shutil.copytree(src, str(tmp))
            try:
                os.rename(str(tmp), str(dst))
            except OSError:
                shutil.rmtree(str(tmp), ignore_errors=True)
                if not dst.exists():
                    raise
        return str(dst)

    # -- CPU preprocessing (feeder-parallel) --------------------------------

    def _preprocess(self, tar: str) -> Optional[Tuple[str, Path]]:
        stamp = _stamp(tar)
        if self._fits_path_for_stamp(stamp).exists():
            return None  # Already archived.
        d = self.work / stamp
        udir = self._untar_dir(stamp)
        try:
            t0 = time.perf_counter()
            ms = self._stage(tar, d, stamp)
            log.info("timing %sMHz batch%d stamp=%s phase=stage_ms elapsed=%.2f s",
                     self.band, self.batch, stamp, time.perf_counter() - t0)

            stats = None            # set by the v3 in-memory prep (prep_inmem=on)
            shadow = None           # v3 result to compare with v2 (prep_inmem=shadow)
            if getattr(self.cfg.pipeline, 'do_aoflag', True):
                t0 = time.perf_counter()
                if self._badants:
                    flag.flag_badants(ms, self._badants)
                    log.info("timing %sMHz batch%d stamp=%s phase=flag_badants elapsed=%.2f s",
                             self.band, self.batch, stamp, time.perf_counter() - t0)

                t0 = time.perf_counter()
                calibrate.applycal(ms, self._bp, self._xy)
                log.info("timing %sMHz batch%d stamp=%s phase=applycal elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)

                t0 = time.perf_counter()
                flag.aoflag(ms, self.cfg.pipeline.cal.aoflagger_strategy,
                            self.cfg.cluster.aoflagger_threads)
                log.info("timing %sMHz batch%d stamp=%s phase=aoflag elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)

                t0 = time.perf_counter()
                avg = average.average(ms, d / f"{stamp}_avg.ms", self.cfg.pipeline.chanbin)
                log.info("timing %sMHz batch%d stamp=%s phase=average elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)
            elif getattr(self.cfg.pipeline, "prep", "casa") == "numpy":
                mode = self._prep_inmem()
                done = self._v3_prep_on(stamp, ms, d) if mode == "on" else None
                if done is not None:
                    avg, stats = done
                else:
                    if mode == "shadow":    # before flag_calibrate rewrites the raw MS
                        shadow = self._v3_prep_shadow(stamp, ms)
                    t0 = time.perf_counter()
                    fastcal.flag_calibrate(ms, self._bp, self._xy, self._badants)
                    t1 = time.perf_counter()
                    avg = fastcal.average(ms, d / f"{stamp}_avg.ms", self.cfg.pipeline.chanbin)
                    log.info("timing %sMHz batch%d stamp=%s phase=flag_cal elapsed=%.2f s",
                             self.band, self.batch, stamp, t1 - t0)
                    log.info("timing %sMHz batch%d stamp=%s phase=average elapsed=%.2f s",
                             self.band, self.batch, stamp, time.perf_counter() - t1)
                    log.info("timing %sMHz batch%d stamp=%s phase=flag_cal_avg elapsed=%.2f s",
                             self.band, self.batch, stamp, time.perf_counter() - t0)
            else:
                t0 = time.perf_counter()
                avg = calibrate.flag_cal_average(
                    ms, d / f"{stamp}_avg.ms", self._bp, self._xy,
                    self._badants, self.cfg.pipeline.chanbin)
                log.info("timing %sMHz batch%d stamp=%s phase=flag_cal_avg elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)

            peel = self.cfg.pipeline.peel_for_band(self.band)
            if stats is None:
                self._fix_uvw(stamp, avg)
                stats = quality.visibility_stats(
                    avg, "DATA", peel.quality_sample_rows,
                    peel.min_visibility_amplitude)
                if shadow is not None:
                    self._v3_shadow_compare(stamp, shadow, avg, stats)
            log.info(
                "input_quality %sMHz stamp=%s sampled=%d usable=%d "
                "nonzero_fraction=%.6g median_abs=%.6g max_abs=%.6g",
                self.band, stamp, stats.sampled_values, stats.usable_values,
                stats.nonzero_fraction, stats.median_amplitude,
                stats.max_amplitude)
            if stats.nonzero_fraction < peel.min_visibility_nonzero_fraction:
                log.warning(
                    "skipping %sMHz stamp=%s before peeling: visibility "
                    "nonzero_fraction %.6g is below %.6g",
                    self.band, stamp, stats.nonzero_fraction,
                    peel.min_visibility_nonzero_fraction)
                shutil.rmtree(d, ignore_errors=True)
                return None

            shutil.rmtree(ms, ignore_errors=True)
            return stamp, avg
        except Exception as e:
            log.error("preprocess %s failed: %s", stamp, e)
            shutil.rmtree(d, ignore_errors=True)
            return None
        finally:
            if udir is not None:
                shutil.rmtree(udir, ignore_errors=True)

    # -- gsi v3 steps 1-2: tmpfs untar, in-memory prep (docs/v3_design.md) -------

    UNTAR_SPACE_MARGIN = 1.5   # free space wanted in untar_to, in units of the tar size

    def _untar_root(self) -> Optional[Path]:
        """This process's directory under v3.untar_to: <untar_to>/.gsi3w_<uid>_<host>_<pid>.

        Named so that a later batch can tell a crashed worker's leftovers (same user and
        host, dead pid) from everything else (_v3_sweep_untar).
        """
        import os
        v3 = getattr(self.cfg.pipeline, "v3", None)
        if v3 is None or not v3.untar_to:
            return None
        return Path(v3.untar_to) / f".gsi3w_{os.getuid()}_{os.uname().nodename}_{os.getpid()}"

    def _untar_dir(self, stamp: str) -> Optional[Path]:
        """<untar root>/<stamp> when the tar is to be streamed into tmpfs, else None."""
        root = self._untar_root()
        return None if root is None else root / stamp

    def _v3_sweep_untar(self) -> None:
        """Remove untar dirs of crashed workers: ours (uid, host) and their pid is dead."""
        import os
        import re
        import stat
        root = self._untar_root()
        if root is None:
            return
        base, uid, host = root.parent, os.getuid(), os.uname().nodename
        pat = re.compile(r"\.gsi3w_(\d+)_(.+)_(\d+)")
        try:
            entries = list(base.iterdir())
        except OSError:
            return
        for e in entries:
            m = pat.fullmatch(e.name)
            if not m or int(m.group(1)) != uid or m.group(2) != host:
                continue
            pid = int(m.group(3))
            try:
                st = os.lstat(e)
            except OSError:
                continue
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid or pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
                continue                # alive (ours or a reused pid): keep
            except ProcessLookupError:
                pass
            except OSError:
                continue                # e.g. EPERM: a live process of another user
            shutil.rmtree(e, ignore_errors=True)
            log.warning("V3_SWEEP removed %s (pid %d is gone)", e, pid)

    def _stage(self, tar: str, d: Path, stamp: str) -> Path:
        """Stage the raw MS: streamed into tmpfs (v3.untar_to) when it fits and works,
        else the v2 way into work_root (logged V3_FALLBACK)."""
        udir = self._untar_dir(stamp)
        if udir is None:
            return stage_ms.stage(tar, d)
        try:
            need = Path(tar).stat().st_size * self.UNTAR_SPACE_MARGIN
            udir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if udir.parent.is_symlink():
                raise FileExistsError(f"{udir.parent} is a symlink")
            free = shutil.disk_usage(str(udir.parent)).free
            if free < need:
                raise OSError(f"{udir.parent}: {free / 1e9:.2f} GB free, want {need / 1e9:.2f} GB")
            return stage_ms.stage(tar, d, untar_to=udir)
        except FileNotFoundError:
            if not Path(tar).exists():
                raise                   # the input is gone: v2 would fail as well
            err = "untar dir vanished"
        except Exception as e:
            err = str(e)
        if udir.is_symlink():
            udir.unlink()
        else:
            shutil.rmtree(udir, ignore_errors=True)
        log.warning("V3_FALLBACK %sMHz stamp=%s untar to tmpfs failed, staging in work_root: %s",
                    self.band, stamp, err)
        return stage_ms.stage(tar, d)

    def _prep_inmem(self) -> str:
        pl = self.cfg.pipeline
        v3 = getattr(pl, "v3", None)
        if (v3 is None or getattr(pl, "prep", "casa") != "numpy"
                or getattr(pl, "do_aoflag", True)):
            return "off"
        return v3.prep_inmem

    def _v3_start_pool(self) -> None:
        if self._prep_inmem() == "off":
            return
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        n = max(1, int(self.cfg.pipeline.v3.prep_procs))
        # spawn, not fork: the parent has threads (feeders, archiver) and casacore state;
        # processes, not threads: python-casacore holds the GIL while it reads
        self._v3_pool = ProcessPoolExecutor(max_workers=n,
                                            mp_context=multiprocessing.get_context("spawn"))
        log.info("batch %sMHz/%d: v3 prep_inmem=%s in %d spawn processes",
                 self.band, self.batch, self._prep_inmem(), n)

    @staticmethod
    def _v3_discard_pool(pool, kill: bool, wait_s: float = 30.0) -> None:
        """Shut a pool down without ever hanging: optionally kill its processes first
        (a broken or timed-out pool), give shutdown ``wait_s``, then kill what is left."""
        procs = list((getattr(pool, "_processes", None) or {}).values())
        if kill:
            for p in procs:
                try:
                    p.kill()
                except Exception:
                    pass
        # with the workers dead the management thread winds down by itself; shutdown in a
        # daemon thread so a stuck queue cannot block the batch
        t = threading.Thread(target=pool.shutdown, kwargs={"wait": True}, daemon=True)
        t.start()
        t.join(wait_s)
        for p in procs:
            try:
                p.join(5)
            except Exception:
                pass
        for p in procs:                 # still running (a stuck task): kill
            try:
                if p.exitcode is None:
                    p.kill()
                    p.join(5)
            except Exception:
                pass

    def _v3_stop_pool(self) -> None:
        with self._v3_lock:
            pool, self._v3_pool = self._v3_pool, None
        if pool is not None:
            self._v3_discard_pool(pool, kill=False)

    def _v3_replace_pool(self, pool) -> None:
        """Kill a broken / timed-out pool and start a fresh one (once per bad pool)."""
        with self._v3_lock:
            if self._v3_pool is not pool:
                return                  # another feeder already replaced it
            self._v3_pool = None
            self._v3_start_pool()
        log.warning("batch %sMHz/%d: v3 prep pool replaced", self.band, self.batch)
        self._v3_discard_pool(pool, kill=True)

    def _v3_run(self, fn, *args):
        """fn(*args) in the prep process pool, at most v3.prep_timeout_s.

        A broken pool (a child died) or a timeout kills and replaces the pool and
        re-raises, so the caller falls back to the v2 prep for that file.
        """
        from concurrent.futures import TimeoutError as FutureTimeout
        from concurrent.futures.process import BrokenProcessPool
        with self._v3_lock:
            pool = self._v3_pool
        if pool is None:
            raise RuntimeError("v3 prep pool is not running")
        timeout = float(getattr(self.cfg.pipeline.v3, "prep_timeout_s", 900.0))
        try:
            fut = pool.submit(fn, *args)
        except (BrokenProcessPool, RuntimeError):
            self._v3_replace_pool(pool)
            raise
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            self._v3_replace_pool(pool)
            raise TimeoutError(f"v3 prep task exceeded {timeout:.0f} s") from None
        except BrokenProcessPool:
            self._v3_replace_pool(pool)
            raise

    def _v3_prep_args(self, ms: Path) -> tuple:
        peel = self.cfg.pipeline.peel_for_band(self.band)
        return (str(ms), self._bp, self._xy, self._badants, self.cfg.pipeline.chanbin,
                self.cfg.pipeline.fix_uvw, peel.quality_sample_rows,
                peel.min_visibility_amplitude)

    def _v3_template_for(self, ms: Path) -> Optional[Path]:
        """The batch's mstransform template (made once, from the first raw MS)."""
        from .v3 import fastprep
        with self._v3_tpl_lock:
            if self._v3_template is not None or self._v3_template_failed:
                return self._v3_template
            tpl = self.work / "_v3_template_avg.ms"
            t0 = time.perf_counter()
            try:
                self.work.mkdir(parents=True, exist_ok=True)
                fastprep.make_template(str(ms), str(tpl), self.cfg.pipeline.chanbin)
                self._v3_template = tpl
            except Exception as e:
                log.error("V3_TEMPLATE %sMHz batch%d: mstransform template failed, "
                          "v2 prep for this batch: %s", self.band, self.batch, e)
                self._v3_template_failed = True
            log.info("timing %sMHz batch%d phase=inmem_template elapsed=%.2f s",
                     self.band, self.batch, time.perf_counter() - t0)
            return self._v3_template

    def _log_v3_timings(self, stamp: str, meta: dict, elapsed: float) -> None:
        for name, t in meta.get("timings", {}).items():
            log.info("timing %sMHz batch%d stamp=%s phase=inmem_%s elapsed=%.2f s cpu=%.2f s",
                     self.band, self.batch, stamp, name, t["wall"], t["cpu"])
        log.info("timing %sMHz batch%d stamp=%s phase=inmem_prep elapsed=%.2f s",
                 self.band, self.batch, stamp, elapsed)

    def _v3_prep_on(self, stamp: str, ms: Path, d: Path):
        """(avg MS written by write_ms_like, VisibilityStats), or None: use the v2 prep."""
        from .v3 import fastprep
        tpl = self._v3_template_for(ms)
        if tpl is None:
            return None
        avg = d / f"{stamp}_avg.ms"
        try:        # a wrong template (rows, channels, frequencies): before any work
            fastprep.check_template(str(tpl), str(ms), self.cfg.pipeline.chanbin)
        except Exception as e:
            log.warning("V3_FALLBACK %sMHz stamp=%s template does not fit, v2 prep: %s",
                        self.band, stamp, e)
            return None
        t0 = time.perf_counter()
        try:
            res = self._v3_run(fastprep.prep_task, *self._v3_prep_args(ms), str(tpl), str(avg))
        except Exception as e:
            log.warning("V3_FALLBACK %sMHz stamp=%s in-memory prep failed, v2 prep: %s",
                        self.band, stamp, e)
            shutil.rmtree(avg, ignore_errors=True)
            return None
        elapsed = time.perf_counter() - t0
        meta = res["meta"]
        self._log_v3_timings(stamp, meta, elapsed)
        # the v2 phase names, from the child's own clocks (queue wait excluded)
        tm = meta.get("timings", {})
        wall = lambda *names: sum(tm.get(n, {}).get("wall", 0.0) for n in names)  # noqa: E731
        task = float(res.get("task_wall", elapsed))
        for phase, value in (("flag_cal", wall("read", "cal")),
                             ("average", wall("average", "write")),
                             ("flag_cal_avg", task),
                             ("inmem_wait", max(0.0, elapsed - task))):
            log.info("timing %sMHz batch%d stamp=%s phase=%s elapsed=%.2f s",
                     self.band, self.batch, stamp, phase, value)
        mode = self.cfg.pipeline.fix_uvw
        if mode != "off":
            if meta.get("fix_uvw_error"):
                log.warning("fix_uvw %sMHz stamp=%s failed, stored UVW kept: %s",
                            self.band, stamp, meta["fix_uvw_error"])
            elif meta.get("fix_uvw"):
                st = meta["fix_uvw"]
                log.info("fix_uvw %sMHz stamp=%s epoch=%s rot_deg=%.4f rms_duv_m=%.3f "
                         "max_dw_m=%.2e rephased=%s", self.band, stamp, mode, st["rot_deg"],
                         st["rms_duv_m"], st["max_dw_m"], st["rephased"])
                log.info("timing %sMHz batch%d stamp=%s phase=fix_uvw elapsed=%.2f s",
                         self.band, self.batch, stamp, wall("uvw"))
        return avg, quality.VisibilityStats(**res["stats"])

    def _v3_prep_shadow(self, stamp: str, ms: Path) -> Optional[dict]:
        from .v3 import fastprep
        t0 = time.perf_counter()
        try:
            res = self._v3_run(fastprep.prep_task, *self._v3_prep_args(ms))
        except Exception as e:
            log.warning("V3_SHADOW %sMHz stamp=%s in-memory prep failed: %s",
                        self.band, stamp, e)
            return None
        self._log_v3_timings(stamp, res["meta"], time.perf_counter() - t0)
        return res

    def _v3_shadow_compare(self, stamp: str, shadow: dict, avg: Path,
                           stats: "quality.VisibilityStats") -> None:
        """Log how far the in-memory prep is from v2's averaged MS (one V3_SHADOW line)."""
        from .v3 import fastprep
        try:
            c = fastprep.compare_snap_ms(shadow["snap"], str(avg))
        except Exception as e:
            log.warning("V3_SHADOW %sMHz stamp=%s compare failed: %s", self.band, stamp, e)
            return
        if "shape_mismatch" in c:
            log.warning("V3_SHADOW %sMHz stamp=%s identical=False shape %s",
                        self.band, stamp, c["shape_mismatch"])
            return
        tot = shadow["meta"].get("timings", {}).get("total", {})
        log.info("V3_SHADOW %sMHz stamp=%s identical=%s data_max_abs=%.3g data_rel_rms=%.3g "
                 "data_ndiff=%d flag_ndiff=%d flag_row_ndiff=%d weight_ndiff=%d "
                 "sigma_ndiff=%d uvw_ndiff=%d uvw_max_abs=%.3g stats_equal=%s "
                 "v3_wall=%.2f v3_cpu=%.2f", self.band, stamp, c["identical"],
                 c["data_max_abs"], c["data_rel_rms"], c["data_ndiff"], c["flag_ndiff"],
                 c["flag_row_ndiff"], c["weight_ndiff"], c["sigma_ndiff"], c["uvw_ndiff"],
                 c.get("uvw_max_abs", 0.0), shadow["stats"] == stats.__dict__,
                 tot.get("wall", 0.0), tot.get("cpu", 0.0))

    def _fix_uvw(self, stamp: str, ms: Path) -> None:
        """Replace the raw (rotated) UVW with J2000 UVW; keep the MS if it fails."""
        mode = self.cfg.pipeline.fix_uvw
        if mode == "off":
            return
        t0 = time.perf_counter()
        try:
            st = uvw.fix_uvw(ms, epoch=mode)
        except Exception as e:  # image with the stored UVW rather than lose the snapshot
            log.warning("fix_uvw %sMHz stamp=%s failed, stored UVW kept: %s",
                        self.band, stamp, e)
            return
        log.info("fix_uvw %sMHz stamp=%s epoch=%s rot_deg=%.4f rms_duv_m=%.3f "
                 "max_dw_m=%.2e rephased=%s", self.band, stamp, mode, st.rot_deg,
                 st.rms_duv_m, st.max_dw_m, st.rephased)
        log.info("timing %sMHz batch%d stamp=%s phase=fix_uvw elapsed=%.2f s",
                 self.band, self.batch, stamp, time.perf_counter() - t0)

    # -- v3 step 5: one GSIV record file per band-hour (v3.archive) ----------------

    def _archive_python_env(self):
        """(python, env) of the archive writer: v3.archive_python (else imaging.gpu_python, else the
        python beside hour_imaging.ttcalx_image), the imager env (hour_imaging.env may set the
        PYTHONPATH of the imager to use) with this pipeline's src in front."""
        from . import hour_image
        pl = self.cfg.pipeline
        python = (v3_params(pl).archive_python or getattr(pl.imaging, "gpu_python", None) or
                  str(Path(pl.hour_imaging.ttcalx_image).parent / "python"))
        env = hour_image.imager_env(self.cfg)
        env["CUDA_VISIBLE_DEVICES"] = ""          # the writer never touches the GPU
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return python, env

    def _check_archive_imager(self) -> None:
        """Batch start: the archive writer's imager must have gsiv (fail fast, not per snapshot)."""
        from .v3 import hourstore
        python, env = self._archive_python_env()
        try:
            where = hourstore.probe_imager(python, env)
        except hourstore.GSIVError as e:
            raise RuntimeError(f"v3.archive={v3_params(self.cfg.pipeline).archive}: {e} -- set v3.archive_python "
                               "or hour_imaging.env PYTHONPATH to an imager with ttcalx_imager.gsiv") from None
        log.info("archive %sMHz batch%d: writer imager %s", self.band, self.batch, where)

    def _archive_client(self, stamp: str):
        """The archive writer of this stamp's band-hour (started on first use): an
        ``hourstore serve`` process in the imager venv, appending to one file per batch."""
        from .v3 import hourstore
        pl = self.cfg.pipeline
        group = self.group(stamp)
        with self._archive_lock:
            c = self._archives.get(group)
            if c is not None and not c.alive:
                self._archives.pop(group, None)
                c = None
            if c is None:
                python, env = self._archive_python_env()
                pc = lst.phase_centre(self.lst_hour(stamp)) if self.lst_layout else None
                out = hourstore.archive_path(self.cfg, self.band, group, self.batch)
                hdr = hourstore.header_for(self.band, pl.date, group, pc, pl.run_label, batch=self.batch,
                                           layout=getattr(pl, "layout", "utc"), cal_bp=str(pl.cal.bp),
                                           cal_xy=str(pl.cal.xy))
                c = hourstore.ArchiveClient(python, out, pc, hdr, env=env)
                log.info("archive %sMHz batch%d group=%s: %s (writer pid %s, %s records, imager %s)",
                         self.band, self.batch, group, out, c.info.get("pid"), c.info.get("n"),
                         c.info.get("imager"))
                self._archives[group] = c
        return c

    def _archive_expected(self, stamp: str) -> None:
        """The snapshot's FITS are archived: list it as expected in the band-hour archive (the hour
        job checks the archive against these lists and never images a reduced set)."""
        from .v3 import hourstore
        out = hourstore.archive_path(self.cfg, self.band, self.group(stamp), self.batch)
        hourstore.add_expected(out, self._kept_stem(stamp), batch=self.batch)

    def _kept_stem(self, stamp: str) -> str:
        return self.kept_ms_path(stamp, ".").name[:-len(".ms")]

    def _archive_record(self, stamp: str, ms: Path, kept: Optional[Path] = None) -> None:
        """Append the compact record of the peeled MS (the kept MS if there is one) to the
        band-hour archive, named as the kept MS (stem, path), not as the work-dir <stamp>_avg.ms.

        A failed writer is killed and dropped (the next snapshot starts a new one, which resumes
        the file).  hour_source=archive: the error fails the snapshot (the hour job then sees it
        missing from the archive and falls back to the kept MSes or fails); hour_source=kept_ms:
        logged loudly once per batch, the v2 hour job does not need the archive."""
        t0 = time.perf_counter()
        group = self.group(stamp)
        try:
            r = self._archive_client(stamp).add(ms, stem=self._kept_stem(stamp),
                                                path=str(kept) if kept else f"{self._kept_stem(stamp)}.ms")
        except Exception as e:
            with self._archive_lock:
                c = self._archives.pop(group, None)
            if c is not None:
                c.close()
            if v3_params(self.cfg.pipeline).hour_source == "archive":
                log.error("ARCHIVE_RECORD_FAILED %sMHz stamp=%s: %s", self.band, stamp, e)
                raise
            if not self._archive_failed:
                log.error("ARCHIVE_RECORD_FAILED %sMHz batch%d stamp=%s: %s (hour_source=kept_ms: the hour "
                          "job is unaffected; the writer is restarted for the next snapshot; further "
                          "failures of this batch at warning level)", self.band, self.batch, stamp, e)
            else:
                log.warning("archive record %sMHz stamp=%s failed: %s", self.band, stamp, e)
            self._archive_failed = True
            return
        log.info("timing %sMHz batch%d stamp=%s phase=archive_record elapsed=%.2f s nbytes=%s n_kept=%s%s",
                 self.band, self.batch, stamp, time.perf_counter() - t0, r.get("nbytes"), r.get("n_kept"),
                 " (supersedes an older record)" if r.get("superseded") else "")

    def _close_archives(self) -> None:
        with self._archive_lock:
            for c in self._archives.values():
                try:
                    c.close()
                except Exception as e:
                    log.warning("archive writer close: %s", e)
            self._archives.clear()

    # -- image + archive (background thread; GPU moves on to the next zest) --

    def _image_archive(self, stamp: str, avg: Path) -> None:
        """Image, render frame, compress+archive FITS for one peeled MS."""
        vmax = self.cfg.pipeline.movie.fixed_vmax
        t0 = time.perf_counter()
        prefix = self.work / stamp / f"img_{stamp}"
        steps: dict = {}
        src = avg
        if getattr(self.cfg.pipeline, "snapshot_phase_centre", "zenith") == "lst":
            # chgcentre a copy: the kept MS stays zenith-phased for the hour imager
            t1 = time.perf_counter()
            src = self.work / stamp / f"{stamp}_lst.ms"
            shutil.rmtree(src, ignore_errors=True)
            shutil.copytree(str(avg), str(src))
            image.chgcentre(src, lst.phase_centre(self.lst_hour(stamp)), self.cfg.pipeline.chgcentre)
            steps["chgcentre"] = time.perf_counter() - t1
        if self._gpu_imager is not None:
            fits_files = self._gpu_imager.image(src, prefix, self.geom, self.cfg.pipeline.imaging,
                                                timings=steps)
        else:
            fits_files = image.image(src, prefix, self.geom, self.cfg.pipeline.imaging,
                                     self.cfg.cluster.wsclean_threads, timings=steps)
        if src != avg:
            shutil.rmtree(src, ignore_errors=True)
        for name, dt in steps.items():
            log.info("timing %sMHz batch%d stamp=%s phase=%s elapsed=%.2f s",
                     self.band, self.batch, stamp, name, dt)
        log.info("timing %sMHz batch%d stamp=%s phase=image elapsed=%.2f s",
                 self.band, self.batch, stamp, time.perf_counter() - t0)

        if (self.cfg.pipeline.movie.enabled and
                self.cfg.pipeline.movie.scale_mode == "adaptive_i"):
            t0 = time.perf_counter()
            vmax = movie.band_vmax([str(fits_files[0])], "adaptive_i", vmax,
                                   self.cfg.pipeline.movie.max_px,
                                   k=self.cfg.pipeline.movie.rms_scale,
                                   horizon_mask=self.cfg.pipeline.movie.horizon_mask,
                                   horizon_radius_fraction=(
                                       self.cfg.pipeline.movie.horizon_radius_fraction))
            log.info("timing %sMHz batch%d stamp=%s phase=band_vmax elapsed=%.2f s",
                     self.band, self.batch, stamp, time.perf_counter() - t0)

        if self.cfg.pipeline.movie.enabled:
            try:
                t0 = time.perf_counter()
                movie.render_frame(str(fits_files[0]), self._frame_dir_for_stamp(stamp) / f"{stamp}.png",
                                   vmax, self.cfg.pipeline.movie.max_px,
                                   horizon_mask=self.cfg.pipeline.movie.horizon_mask,
                                   horizon_radius_fraction=(
                                       self.cfg.pipeline.movie.horizon_radius_fraction))
                log.info("timing %sMHz batch%d stamp=%s phase=render_frame elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)
            except Exception as e:
                log.error("render_frame %s failed: %s", stamp, e)

        if self.lst_layout:
            self._qa.append(dict(stamp=stamp, ts=lst.timestamp(stamp),
                                 file=self._archive_paths(stamp)[1].name,
                                 **image.v_stats(fits_files[1])))
        for f, target in zip(fits_files, self._archive_paths(stamp)):
            t0 = time.perf_counter()
            f = f.rename(f.parent / target.name)
            fz = compress.fpack(f)
            log.info("timing %sMHz batch%d stamp=%s phase=fpack file=%s elapsed=%.2f s",
                     self.band, self.batch, stamp, f.name, time.perf_counter() - t0)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fz), str(target.parent / fz.name))
        archive = v3_params(self.cfg.pipeline).archive != "none"
        if archive:            # the snapshot is archived: the hour job must find its record
            self._archive_expected(stamp)
        kept = None
        if (self.cfg.pipeline.keep_ms_root or getattr(self.cfg.pipeline, "keep_ms_local", None)) and \
                v3_params(self.cfg.pipeline).write_kept_ms:
            t0 = time.perf_counter()
            kept = self.keep_ms(stamp, avg)
            log.info("timing %sMHz batch%d stamp=%s phase=keep_ms elapsed=%.2f s dst=%s",
                     self.band, self.batch, stamp, time.perf_counter() - t0, kept)
        if archive:            # after the FITS and keep_ms: the archive's set is v2's kept set
            self._archive_record(stamp, kept or avg, kept)
        shutil.rmtree(self.work / stamp, ignore_errors=True)

    def keep_root(self) -> str:
        """keep_ms_local while it has keep_ms_local_min_free_gb free, else keep_ms_root."""
        pl = self.cfg.pipeline
        local = getattr(pl, "keep_ms_local", None)
        if not local:
            return pl.keep_ms_root
        Path(local).mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(local).free / 1e9
        if free_gb >= pl.keep_ms_local_min_free_gb or not pl.keep_ms_root:
            return local
        log.warning("keep_ms_local %s has %.0f GB free (< %.0f GB); keeping on %s",
                    local, free_gb, pl.keep_ms_local_min_free_gb, pl.keep_ms_root)
        return pl.keep_ms_root

    def kept_ms_path(self, stamp: str, root: Optional[str] = None) -> Path:
        """<keep root>/<band>MHz/<date>/<hour>/<stamp>.ms (stamp ends in _<band>MHz)."""
        name = stamp if stamp.endswith(f"_{self.band}MHz") else f"{stamp}_{self.band}MHz"
        return (Path(root or self.cfg.pipeline.keep_ms_root) / f"{self.band}MHz" /
                self.cfg.pipeline.date / self.group(stamp) / f"{name}.ms")

    def keep_ms(self, stamp: str, avg: Path) -> Path:
        """Move the peeled MS to the keep root (atomic rename into place)."""
        import os
        pl = self.cfg.pipeline
        root = self.keep_root()
        dst = self.kept_ms_path(stamp, root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".{dst.name}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(str(avg), str(tmp))
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(str(tmp), str(dst))
        for other in (getattr(pl, "keep_ms_local", None), pl.keep_ms_root):
            if other and other != root:     # a re-run must not leave a second copy
                shutil.rmtree(self.kept_ms_path(stamp, other), ignore_errors=True)
        return dst

    def run_batch(self, tars: List[str]) -> None:
        todo = [t for t in tars
                if not self._fits_path_for_stamp(_stamp(t)).exists()]
        if not todo:
            log.info("batch %sMHz/%d: nothing to do", self.band, self.batch)
            return
        if v3_params(self.cfg.pipeline).archive != "none":
            self._check_archive_imager()
        self.work.mkdir(parents=True, exist_ok=True)
        batch_start = time.perf_counter()
        t0 = time.perf_counter()
        self._bp = self._stage_cal(self.cfg.pipeline.cal.bp)
        self._xy = self._stage_cal(self.cfg.pipeline.cal.xy)
        log.info("timing %sMHz batch%d phase=stage_cal elapsed=%.2f s",
                 self.band, self.batch, time.perf_counter() - t0)
        if getattr(self.cfg.pipeline, 'do_badants', True):
            t0 = time.perf_counter()
            dev_python = f"{self.cfg.pipeline.env.dev_env}/bin/python"
            self._badants = flag.badants(_mjd(_stamp(todo[0])), dev_python)
            log.info("timing %sMHz batch%d phase=badants(batch) n=%s elapsed=%.2f s",
                     self.band, self.batch,
                     len(self._badants.split(",")) if self._badants else 0,
                     time.perf_counter() - t0)
        daemon = self._start_daemon()  # Julia JIT overlaps preprocessing.
        if getattr(self.cfg.pipeline.imaging, "imager", "wsclean") == "ttcalx":
            self._gpu_imager = self._start_gpu_imager()   # started at the first snapshot
        n_ok, n_fail = 0, 0
        try:
            self._v3_sweep_untar()
            self._v3_start_pool()
            with ThreadPoolExecutor(max_workers=self.cfg.cluster.max_feeders) as pool, \
                 ThreadPoolExecutor(max_workers=max(1, self.cfg.cluster.archive_workers)) as archiver:
                # test copy: GSI_PREP_LOOKAHEAD bounds how many files the feeders run ahead
                # of zest (0 = all at once, as production)
                import os
                ahead = int(os.environ.get("GSI_PREP_LOOKAHEAD", "0")) or len(todo)
                futs = [pool.submit(self._preprocess, t) for t in todo[:ahead]]
                archive_futs = []
                n_zested = 0
                for i in range(len(todo)):
                    fut = futs[i]
                    if i + ahead < len(todo):
                        futs.append(pool.submit(self._preprocess, todo[i + ahead]))
                    r = fut.result()
                    if not r:
                        continue
                    stamp, avg = r
                    recycle = getattr(self.cfg.cluster, "zest_recycle_files", 25)
                    if recycle and n_zested and n_zested % recycle == 0:
                        # Recycling avoids a repeatable CUDA stall near file 31.
                        log.info("batch %sMHz/%d: recycling zest daemon after %d files",
                                 self.band, self.batch, n_zested)
                        daemon.close()
                        daemon = self._start_daemon()
                    try:
                        t0 = time.perf_counter()
                        daemon.zest(avg)
                        n_zested += 1
                        log.info("timing %sMHz batch%d stamp=%s phase=zest elapsed=%.2f s",
                                 self.band, self.batch, stamp, time.perf_counter() - t0)
                    except Exception as e:
                        log.error("zest %s failed: %s", stamp, e)
                        shutil.rmtree(self.work / stamp, ignore_errors=True)
                        n_fail += 1
                        continue
                    archive_futs.append(archiver.submit(self._image_archive, stamp, avg))
                for f in archive_futs:
                    try:
                        f.result()
                        n_ok += 1
                    except Exception as e:
                        log.error("image/archive failed: %s", e)
                        n_fail += 1
        finally:
            daemon.close()
            if self._gpu_imager is not None:
                self._gpu_imager.close()
            self._v3_stop_pool()
            self._close_archives()
            shutil.rmtree(self.work, ignore_errors=True)
            root = self._untar_root()
            if root is not None and not root.is_symlink():
                shutil.rmtree(root, ignore_errors=True)
            self._write_qa_stats()
        log.info("batch %sMHz/%d: total elapsed %.2f s for %d files (%d failed)",
                 self.band, self.batch, time.perf_counter() - batch_start, n_ok, n_fail)
        if n_fail:
            log.warning("batch %sMHz/%d completed with %d failed file(s); continuing",
                        self.band, self.batch, n_fail)
        _require_batch_outputs(self.band, self.batch, len(todo), n_ok)

    def _write_qa_stats(self) -> None:
        """Pilot V statistics of this batch -> QA/.snapshot_stats/<batch>.jsonl (hour step reads them)."""
        if not self._qa:
            return
        import json
        import os
        d = self.band_dir(self._qa[0]["stamp"]) / "QA" / ".snapshot_stats"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"b{self.batch}_{os.uname().nodename}_{os.getpid()}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in self._qa))
        log.info("batch %sMHz/%d: wrote %d pilot QA rows to %s", self.band, self.batch,
                 len(self._qa), path)

    def _zest_env(self) -> dict:
        import os
        env = dict(os.environ)
        if getattr(self.cfg.pipeline.imaging, "imager", "wsclean") == "ttcalx":
            # the GPU snapshot imager shares this GPU: make CUDA.jl hand cached memory back
            env.setdefault("JULIA_CUDA_SOFT_MEMORY_LIMIT", "4GiB")
        return env

    def _start_daemon(self) -> ZestDaemon:
        import os
        workers_dir = os.environ["GSI_WORKERS"]
        return ZestDaemon(
            julia=f"{self.cfg.pipeline.env.julia_bin}/julia",
            project=self.cfg.pipeline.env.ttcalx,
            daemon_jl=str(Path(workers_dir) / "zest_daemon.jl"),
            sources=self.cfg.pipeline.cal.sources, env=self._zest_env(),
            zest_env={"ZEST_COLUMN": "DATA",
                      **self.cfg.pipeline.peel_for_band(self.band).daemon_env(
                          self.cfg.pipeline.cal.rfi_sources)})

    def stitch(self) -> None:
        if not self.cfg.pipeline.movie.enabled:
            log.info("movie stitching disabled")
            return
        if not self.out.exists():
            log.info("no output directory for %sMHz", self.band)
            return
        hour_dirs = sorted(p for p in self.out.iterdir()
                           if p.is_dir() and p.name.isdigit())
        for hour_dir in hour_dirs:
            hour = hour_dir.name
            frames = self.frame_dir(hour)
            out = hour_dir / f"{self.band}MHz_{self.cfg.pipeline.date}_{hour}UTC.mp4"
            try:
                m = movie.stitch(frames, out, self.cfg.pipeline.movie.fps,
                                 self.cfg.pipeline.movie.crf)
                log.info("stitched %s", m or f"(no frames for {hour}UTC)")
                if m and not self.cfg.pipeline.movie.keep_frames:
                    shutil.rmtree(frames, ignore_errors=True)
                    log.info("deleted frame directory %s", frames)
            except Exception as e:
                log.error("stitch %sMHz %sUTC failed: %s", self.band, hour, e)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir")
    ap.add_argument("band", type=int)
    ap.add_argument("batch", type=int)
    ap.add_argument("manifest")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = Config.load(a.config_dir)
    from . import hour_image
    if a.manifest.startswith(hour_image.MANIFEST_PREFIX):
        hour_image.run_band(cfg, a.band, a.manifest[len(hour_image.MANIFEST_PREFIX):], a.gpu)
        return 0
    w = Worker(cfg, a.band, a.batch, a.gpu)
    if a.manifest == "STITCH" or a.batch < 0:
        w.stitch()
        return 0
    tars = [ln.strip() for ln in Path(a.manifest).read_text().splitlines() if ln.strip()]
    w.run_batch(tars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
