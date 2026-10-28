"""Process one subband batch on a GPU worker.

CPU preparation, GPU peeling, and image archiving overlap. Existing FITS
products are skipped, and local working data is removed after each batch.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from . import lst
from .config import Config
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
        try:
            t0 = time.perf_counter()
            ms = stage_ms.stage(tar, d)
            log.info("timing %sMHz batch%d stamp=%s phase=stage_ms elapsed=%.2f s",
                     self.band, self.batch, stamp, time.perf_counter() - t0)

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

            self._fix_uvw(stamp, avg)

            peel = self.cfg.pipeline.peel_for_band(self.band)
            stats = quality.visibility_stats(
                avg, "DATA", peel.quality_sample_rows,
                peel.min_visibility_amplitude)
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
        if self.cfg.pipeline.keep_ms_root or getattr(self.cfg.pipeline, "keep_ms_local", None):
            t0 = time.perf_counter()
            kept = self.keep_ms(stamp, avg)
            log.info("timing %sMHz batch%d stamp=%s phase=keep_ms elapsed=%.2f s dst=%s",
                     self.band, self.batch, stamp, time.perf_counter() - t0, kept)
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
            with ThreadPoolExecutor(max_workers=self.cfg.cluster.max_feeders) as pool, \
                 ThreadPoolExecutor(max_workers=max(1, self.cfg.cluster.archive_workers)) as archiver:
                futs = [pool.submit(self._preprocess, t) for t in todo]
                archive_futs = []
                n_zested = 0
                for fut in futs:
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
            shutil.rmtree(self.work, ignore_errors=True)
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

    def _start_daemon(self) -> ZestDaemon:
        import os
        workers_dir = os.environ["GSI_WORKERS"]
        return ZestDaemon(
            julia=f"{self.cfg.pipeline.env.julia_bin}/julia",
            project=self.cfg.pipeline.env.ttcalx,
            daemon_jl=str(Path(workers_dir) / "zest_daemon.jl"),
            sources=self.cfg.pipeline.cal.sources, env=dict(os.environ),
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
