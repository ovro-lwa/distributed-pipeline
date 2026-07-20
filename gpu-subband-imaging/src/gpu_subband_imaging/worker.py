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

from .config import Config
from .stages import average, calibrate, compress, flag, image, movie, stage_ms
from .stages.peel import ZestDaemon

log = logging.getLogger("gsi.worker")


def _mjd(stamp: str) -> float:
    from astropy.time import Time
    s = stamp.split("_")
    iso = f"{s[0][:4]}-{s[0][4:6]}-{s[0][6:]}T{s[1][:2]}:{s[1][2:4]}:{s[1][4:]}"
    return float(Time(iso).mjd)


def _stamp(tar: str) -> str:
    return Path(tar).name[:-len(".ms.tar")]


def _hour(stamp: str) -> str:
    return stamp.split("_")[1][:2]


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

    def hour_dir(self, hour: str) -> Path:
        return self.out / hour

    def fits_dir(self, hour: str) -> Path:
        return self.hour_dir(hour) / "fits"

    def frame_dir(self, hour: str) -> Path:
        return self.hour_dir(hour) / "images"

    def _fits_path_for_stamp(self, stamp: str) -> Path:
        return self.fits_dir(_hour(stamp)) / f"img_{stamp}-I-image.fits.fz"

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
            else:
                t0 = time.perf_counter()
                avg = calibrate.flag_cal_average(
                    ms, d / f"{stamp}_avg.ms", self._bp, self._xy,
                    self._badants, self.cfg.pipeline.chanbin)
                log.info("timing %sMHz batch%d stamp=%s phase=flag_cal_avg elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)

            shutil.rmtree(ms, ignore_errors=True)
            return stamp, avg
        except Exception as e:
            log.error("preprocess %s failed: %s", stamp, e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # -- image + archive (background thread; GPU moves on to the next zest) --

    def _image_archive(self, stamp: str, avg: Path) -> None:
        """Image, render frame, compress+archive FITS for one peeled MS."""
        vmax = self.cfg.pipeline.movie.fixed_vmax
        hour = _hour(stamp)
        t0 = time.perf_counter()
        prefix = self.work / stamp / f"img_{stamp}"
        fits_files = image.image(avg, prefix, self.geom, self.cfg.pipeline.imaging,
                                 self.cfg.cluster.wsclean_threads)
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
                movie.render_frame(str(fits_files[0]), self.frame_dir(hour) / f"{stamp}.png",
                                   vmax, self.cfg.pipeline.movie.max_px,
                                   horizon_mask=self.cfg.pipeline.movie.horizon_mask,
                                   horizon_radius_fraction=(
                                       self.cfg.pipeline.movie.horizon_radius_fraction))
                log.info("timing %sMHz batch%d stamp=%s phase=render_frame elapsed=%.2f s",
                         self.band, self.batch, stamp, time.perf_counter() - t0)
            except Exception as e:
                log.error("render_frame %s failed: %s", stamp, e)

        for f in fits_files:
            t0 = time.perf_counter()
            fz = compress.fpack(f)
            log.info("timing %sMHz batch%d stamp=%s phase=fpack file=%s elapsed=%.2f s",
                     self.band, self.batch, stamp, f.name, time.perf_counter() - t0)
            dst = self.fits_dir(hour)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fz), str(dst / fz.name))
        shutil.rmtree(self.work / stamp, ignore_errors=True)

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
        n_ok, n_fail = 0, 0
        try:
            with ThreadPoolExecutor(max_workers=self.cfg.cluster.max_feeders) as pool, \
                 ThreadPoolExecutor(max_workers=1) as archiver:
                futs = [pool.submit(self._preprocess, t) for t in todo]
                archive_futs = []
                n_zested = 0
                for fut in futs:
                    r = fut.result()
                    if not r:
                        continue
                    stamp, avg = r
                    if n_zested and n_zested % 25 == 0:
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
            shutil.rmtree(self.work, ignore_errors=True)
        log.info("batch %sMHz/%d: total elapsed %.2f s for %d files (%d failed)",
                 self.band, self.batch, time.perf_counter() - batch_start, n_ok, n_fail)
        if n_fail:
            log.warning("batch %sMHz/%d completed with %d failed file(s); continuing",
                        self.band, self.batch, n_fail)

    def _start_daemon(self) -> ZestDaemon:
        import os
        workers_dir = os.environ["GSI_WORKERS"]
        return ZestDaemon(
            julia=f"{self.cfg.pipeline.env.julia_bin}/julia",
            project=self.cfg.pipeline.env.ttcalx,
            daemon_jl=str(Path(workers_dir) / "zest_daemon.jl"),
            sources=self.cfg.pipeline.cal.sources, env=dict(os.environ),
            peel_opts=dict(
                column="DATA", **vars(self.cfg.pipeline.peel_for_band(self.band))))

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
    w = Worker(cfg, a.band, a.batch, a.gpu)
    if a.manifest == "STITCH" or a.batch < 0:
        w.stitch()
        return 0
    tars = [ln.strip() for ln in Path(a.manifest).read_text().splitlines() if ln.strip()]
    w.run_batch(tars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
