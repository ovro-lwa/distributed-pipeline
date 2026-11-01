"""Hour step of one band and LST hour for layout=lst (worker side).

1. Snapshot QA as the main pipeline (orca analyze_snapshot_quality on the pilot V
   snapshots: std in the central 1024 x 1024 box, rejected if > median + 3 MAD-std
   or < 0.5 x median) from the statistics the snapshot workers wrote to
   QA/.snapshot_stats/; writes QA/snapshot_qa_<band>MHz.csv, QA/pilot_qa_rejected.txt
   and QA/snapshot_qa_<band>MHz.png.
2. ``ttcalx-image hour`` on the kept (zenith-phased) peeled MSes at the LST-hour
   phase centre, the QA-rejected snapshots excluded.
3. Products moved to I/deep, I/10min, V/deep, V/10min with the main pipeline's
   names (deep: <band>MHz-<product>-<kind>-<mid>.fits, 10-min:
   <band>MHz-<product>-<mid>-<kind>.fits, <mid> = YYYYMMDD_HHMMSS of the interval).
4. Movie of the CLEAN I snapshots -> Movies/<band>MHz_I_Clean.mp4.
5. provenance.json in the band directory; kept MSes deleted on success
   (hour_imaging.delete_kept_ms).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import lst
from .config import Config, v3_params

log = logging.getLogger("gsi.hour_lst")


def _band_dir(cfg: Config, band: int, lst_hour: int) -> Path:
    pl = cfg.pipeline
    return lst.band_dir(pl.output_root, lst_hour, pl.date, pl.run_label, band)


# ---- 1. snapshot QA ------------------------------------------------------------

def qa_reject(rms: List[float]) -> Tuple[List[int], float, float]:
    """orca analyze_snapshot_quality rule: indices with rms > med + 3 mad_std or < 0.5 med."""
    import numpy as np
    from astropy.stats import mad_std
    r = np.asarray(rms, dtype=float)
    med = float(np.nanmedian(r))
    sd = float(mad_std(r, ignore_nan=True))
    bad = [i for i, x in enumerate(r) if x > med + 3.0 * sd or x < 0.5 * med]
    return bad, med, sd


def read_stats(qa_dir: Path) -> List[Dict]:
    rows: Dict[str, Dict] = {}
    d = qa_dir / ".snapshot_stats"
    if d.is_dir():
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["ts"]] = r            # a re-run of a snapshot replaces it
    return [rows[k] for k in sorted(rows)]


def write_qa(qa_dir: Path, band: int, rows: List[Dict]) -> Tuple[set, Dict]:
    """Write the main-pipeline QA files; return (rejected timestamps, summary)."""
    if not rows:
        return set(), {"n": 0}
    bad, med, sd = qa_reject([r["rms"] for r in rows])
    bad_set = set(bad)
    with open(qa_dir / f"snapshot_qa_{band}MHz.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["snapshot_index", "scan_number", "timestamp_utc",
                                           "rms_jy_per_beam", "peak_jy_per_beam", "flagged",
                                           "filename"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"snapshot_index": i, "scan_number": i + 1,
                        "timestamp_utc": lst.frame_time(r["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "rms_jy_per_beam": r["rms"], "peak_jy_per_beam": r["peak"],
                        "flagged": "true" if i in bad_set else "false", "filename": r["file"]})
    with open(qa_dir / "pilot_qa_rejected.txt", "w") as fh:
        fh.write(f"# pilot V snapshot QA rejections (analyze_snapshot_quality): {len(bad)}/{len(rows)}; "
                 "snapshot_index scan_number timestamp_utc rms filename\n")
        for i in bad:
            r = rows[i]
            fh.write(f"{i} {i + 1} {lst.frame_time(r['ts']).strftime('%Y-%m-%dT%H:%M:%SZ')} "
                     f"{r['rms']} {r['file']}\n")
    try:
        _plot_qa(qa_dir / f"snapshot_qa_{band}MHz.png", band, rows, bad_set, med, sd)
    except Exception as e:                                   # the plot is optional
        log.warning("QA plot failed: %s", e)
    return {rows[i]["ts"] for i in bad}, {"n": len(rows), "rejected": len(bad),
                                          "median_rms": med, "mad_std_rms": sd}


def _plot_qa(path: Path, band: int, rows: List[Dict], bad: set, med: float, sd: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [lst.frame_time(r["ts"]) for r in rows]
    y = [r["rms"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, ".", ms=3, color="#3366aa", label="pilot V rms")
    bt = [t[i] for i in sorted(bad)]
    ax.plot(bt, [y[i] for i in sorted(bad)], "x", color="#cc3311", label=f"rejected ({len(bad)})")
    ax.axhline(med, color="grey", lw=0.8)
    ax.axhline(med + 3 * sd, color="grey", lw=0.8, ls="--")
    ax.axhline(0.5 * med, color="grey", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("V rms, central 1024 px box (Jy/beam)")
    ax.set_title(f"{band} MHz snapshot QA ({len(rows)} snapshots)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---- 3. product names -----------------------------------------------------------

_DEEP = re.compile(r"^(?P<band>\d+MHz)-(?P<prod>[IV]-(?:Deep-[A-Za-z0-9.-]+?|Taper-Deep))-(?P<kind>image|residual|model|psf|dirty)\.fits$")
_TEN = re.compile(r"^(?P<band>\d+MHz)-(?P<prod>[IV]-Taper-10min)-t(?P<k>\d{4})-(?P<kind>image|residual|model|psf|dirty)\.fits$")


def _mid(times: List[datetime]) -> str:
    a, b = min(times), max(times)
    return (a + (b - a) / 2).strftime("%Y%m%d_%H%M%S")


def place_products(work: Path, band_dir: Path, stamps: List[str], n_intervals: int = 6) -> List[str]:
    """Move the imager's files into I/V deep/10min with timestamped main-pipeline names."""
    times = sorted(lst.frame_time(s) + timedelta(seconds=5) for s in stamps)   # TIME_CENTROID
    n = len(times)
    deep_ts = _mid(times)
    bnd = [int(math.floor(k * n / n_intervals)) for k in range(n_intervals + 1)]
    ten_ts = [_mid(times[bnd[k]:bnd[k + 1]]) if bnd[k + 1] > bnd[k] else deep_ts
              for k in range(n_intervals)]
    moved = []
    for f in sorted(work.glob("*.fits")):
        m = _DEEP.match(f.name)
        if m:
            pol = m["prod"][0]
            dst = band_dir / pol / "deep" / f"{m['band']}-{m['prod']}-{m['kind']}-{deep_ts}.fits"
        else:
            m = _TEN.match(f.name)
            if not m:
                continue
            pol = m["prod"][0]
            dst = (band_dir / pol / "10min" /
                   f"{m['band']}-{m['prod']}-{ten_ts[int(m['k'])]}-{m['kind']}.fits")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))
        moved.append(str(dst))
    return moved


# ---- the band-hour job ------------------------------------------------------------

def _git_rev(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def qa_step(qa_dir: Path, band: int, rows: List[Dict]) -> Tuple[set, Dict]:
    """Step 1 (shared by the hour job and the gsi v3 in-batch hour stage): the main-pipeline QA
    files of the band-hour's pilot V rows; returns (rejected timestamps, QA summary)."""
    return write_qa(qa_dir, band, rows)


def movie_step(cfg: Config, band: int, group: str, bdir: Path) -> Tuple[Optional[str], float]:
    """Step 4: Movies/<band>MHz_I_Clean.mp4 from the frames; (movie path or None, seconds)."""
    from .stages import movie
    pl = cfg.pipeline
    t0 = time.perf_counter()
    frames = bdir / "Movies" / "frames"
    out = None
    try:
        m = movie.stitch(frames, bdir / "Movies" / f"{band}MHz_I_Clean.mp4", pl.movie.fps, pl.movie.crf)
        out = str(m) if m else None
        if m and not pl.movie.keep_frames:
            shutil.rmtree(frames, ignore_errors=True)
    except Exception as e:
        log.error("movie %sMHz %s failed: %s", band, group, e)
    return out, time.perf_counter() - t0


def write_provenance(cfg: Config, band: int, group: str, gpu: Optional[int], rec: Dict,
                     v3: Optional[Dict] = None) -> Dict:
    """Step 5: provenance.json of the band directory and QA/<band>MHz-hour-timing.json."""
    from .metadata import _code_revision, software_versions
    pl, h = cfg.pipeline, cfg.pipeline.hour_imaging
    lst_hour = int(group.rstrip("h"))
    bdir = _band_dir(cfg, band, lst_hour)
    qa_dir = bdir / "QA"
    snaps = sorted((bdir / "snapshots_clean").glob("*.fits.fz")) if (bdir / "snapshots_clean").is_dir() else []
    prov = {
        "pipeline": "gpu-subband-imaging", "layout": "lst",
        "pipeline_version": _code_revision(), "ttcalx_version": _git_rev(pl.env.ttcalx),
        "worker_node": os.uname().nodename, "gpu": gpu, "subband": f"{band}MHz",
        "obs_date": pl.date, "lst_label": group, "run_label": pl.run_label,
        "phase_centre": lst.phase_centre(lst_hour),
        "snapshot_phase_centre": pl.snapshot_phase_centre,
        "bp_table": pl.cal.bp, "xy_table": pl.cal.xy, "sky_model": pl.cal.sources,
        "rfi_model": pl.cal.rfi_sources if pl.peel_for_band(band).rfi else None,
        "peel": vars(pl.peel_for_band(band)), "imaging": vars(pl.imaging),
        "hour_imaging": {"products": list(h.products), "imager": h.ttcalx_image},
        "software": software_versions(cfg),
        "sun_cut": pl.sun_cut, "n_snapshots_clean": len(snaps), "hour": rec,
    }
    if v3 is not None:
        prov["v3"] = v3
    (bdir / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    (qa_dir / f"{band}MHz-hour-timing.json").write_text(json.dumps(rec, indent=1, default=str))
    return prov


def run_band(cfg: Config, band: int, group: str, gpu: Optional[int] = None) -> Dict:
    from . import hour_image

    pl, h = cfg.pipeline, cfg.pipeline.hour_imaging
    lst_hour = int(group.rstrip("h"))
    bdir = _band_dir(cfg, band, lst_hour)
    qa_dir = bdir / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)
    ms = hour_image.list_kept(cfg, band, group)
    archives: List[Path] = []
    source = "kept_ms"
    rec: Dict = dict(band=band, date=pl.date, lst_label=group, run_label=pl.run_label,
                     n_ms=len(ms), host=os.uname().nodename, gpu=gpu,
                     products=list(h.products), start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if v3_params(pl).hour_source == "archive":    # gsi v3 step 5: the band-hour's GSIV record file(s)
        source, archives, why = _archive_or_fallback(cfg, band, group, lst_hour, ms)
        rec.update(hour_source=source, archive=[str(a) for a in archives],
                   archive_bytes=sum(a.stat().st_size for a in archives if a.exists()))
        if why:
            rec["archive_problems"] = why
    if source == "archive":
        from .v3 import hourstore
        stamps = hourstore.archive_inventory(archives)["stems"]
        rec["n_records"] = len(stamps)
        ms_for_cmd: List[str] = []
    elif source == "failed":
        stamps, ms_for_cmd = [], []
    else:
        stamps = [Path(p).name[:-3] for p in ms]
        ms_for_cmd = ms
    t_all = time.perf_counter()

    # 1. QA
    t0 = time.perf_counter()
    bad_ts, rec["qa"] = qa_step(qa_dir, band, read_stats(qa_dir))
    exclude = [i for i, s in enumerate(stamps) if lst.timestamp(s) in bad_ts]
    rec["qa"]["excluded_from_hour"] = len(exclude)
    rec["timing_s"] = {"qa": time.perf_counter() - t0}
    log.info("hour %sMHz %s: %d %s, QA %s", band, group, len(stamps),
             f"records in {len(archives)} archive file(s)" if source == "archive" else "kept MSes", rec["qa"])

    rc = 0
    # hour_source=archive: the stricter rules (the job must load exactly the given stems; only those
    # kept MSes are deleted).  Defaults (kept_ms): exactly 13405ce -- no such check, and on success
    # the whole kept dirs are removed (QA-rejected MSes and leftovers included).
    strict = v3_params(pl).hour_source == "archive"
    imaged: set = set()          # stems the imager loaded in this job
    if source == "failed":
        rec["status"] = "failed"
        rc = -2
    elif not stamps:
        rec["status"] = "no-data"
    else:
        # 2. imager
        work = bdir / ".hour_work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        if source == "archive":
            # QA rejections by stem: the record order of the archive need not be the MS order
            xs = [s for s in stamps if lst.timestamp(s) in bad_ts]
            list_file = work / "archive-list.txt"
            list_file.write_text("\n".join(map(str, archives)) + "\n")
            src = ["--archive", f"@{list_file}"]
            if xs:
                (work / "exclude-stems.txt").write_text("\n".join(xs) + "\n")
            excl = ["--exclude-stems", f"@{work / 'exclude-stems.txt'}"] if xs else []
        else:
            list_file = work / "ms-list.txt"
            list_file.write_text("\n".join(ms_for_cmd) + "\n")
            src = [f"@{list_file}"]
            excl = ["--exclude", ",".join(map(str, exclude))] if exclude else []
        cmd = ([h.ttcalx_image, "hour"] + src + ["--band", str(band),
                "--out", str(work / f"{band}MHz"), "--products", ",".join(h.products),
                "--lst-hour", str(lst_hour), "--qa", "none", "--nproc", str(h.nproc), "-v"]
               + excl + list(h.extra_args))
        rec["cmd"] = cmd
        log.info("hour %sMHz %s: %s", band, group, " ".join(cmd))
        t0 = time.perf_counter()
        with open(qa_dir / f"{band}MHz-ttcalx-image.log", "w") as fh:
            try:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=hour_image.imager_env(cfg), timeout=h.timeout_seconds).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        rec["timing_s"]["hour_image"] = time.perf_counter() - t0
        rec["rc"] = rc
        summary = work / f"{band}MHz-summary.json"
        if summary.exists():
            try:
                s = json.loads(summary.read_text())
                rec["imager_timing_s"] = s.get("timing", {})
                rec["n_loaded"] = s.get("n_loaded")
                rec["n_rejected_hour"] = len(s.get("rejected", []))
                rec["phase_centre_deg"] = s.get("phase_centre")
                if rc == 0:
                    imaged = {q["stem"] for q in s.get("qa", [])}
                shutil.move(str(summary), str(qa_dir / summary.name))
            except (ValueError, KeyError, TypeError) as e:
                log.warning("cannot parse %s: %s", summary, e)
        if strict and rc == 0 and imaged != set(stamps):
            # the job did not load exactly the snapshots it was given: fail, delete nothing
            rec["imaged_mismatch"] = dict(missing=sorted(set(stamps) - imaged), extra=sorted(imaged - set(stamps)))
            log.error("hour %sMHz %s: the imager loaded %d snapshots, expected %d: %s", band, group,
                      len(imaged), len(stamps), rec["imaged_mismatch"])
            rc, imaged = -3, set()
        # 3. products
        t0 = time.perf_counter()
        rec["files"] = place_products(work, bdir, stamps)
        rec["timing_s"]["place_products"] = time.perf_counter() - t0
        shutil.rmtree(work, ignore_errors=True)
        rec["status"] = "ok" if rc == 0 else "failed"

    # 4. movie
    if pl.movie.enabled:
        rec["movie"], rec["timing_s"]["movie"] = movie_step(cfg, band, group, bdir)

    # 5. provenance, kept MSes
    rec["timing_s"]["total"] = time.perf_counter() - t_all
    v3b = None
    if v3_params(pl).hour_source == "archive":
        v3b = {"hour_source": "archive", "used": source, "archive": v3_params(pl).archive,
               "files": [{"path": str(a), "bytes": a.stat().st_size} for a in archives if a.exists()],
               "n_records": len(stamps) if source == "archive" else None,
               "problems": rec.get("archive_problems")}
    write_provenance(cfg, band, group, gpu, rec, v3b)
    log.info("timing %sMHz lst=%s phase=hour_total elapsed=%.2f s rc=%s n_ms=%d",
             band, group, rec["timing_s"]["total"], rc, len(ms))
    if source == "failed":
        raise RuntimeError(f"hour {band}MHz {group}: the archive is unusable and there are no complete kept "
                           f"MSes to fall back to: {rec.get('archive_problems')}")
    if rc != 0:
        raise RuntimeError(f"ttcalx-image hour failed for {band}MHz {group} (rc={rc})")
    if strict:
        if imaged and h.delete_kept_ms:
            _delete_kept(cfg, band, group, imaged)
    elif ms and h.delete_kept_ms:            # v2 (13405ce), unchanged
        for kd in hour_image.kept_dirs(cfg, band, group):
            if kd.is_dir():
                shutil.rmtree(kd, ignore_errors=True)
                log.info("deleted kept MSes %s", kd)
    return rec


def _delete_kept(cfg: Config, band: int, group: str, imaged: set) -> None:
    """Delete the kept MSes whose stems this hour job imaged -- never any other (a no-data job,
    a failed one or a reduced archive deletes nothing it did not image)."""
    from . import hour_image
    for kd in hour_image.kept_dirs(cfg, band, group):
        if not kd.is_dir():
            continue
        n = 0
        for p in sorted(kd.iterdir()):
            if p.name.endswith(".ms") and not p.name.startswith(".") and p.name[:-3] in imaged:
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        left = [p for p in kd.iterdir() if not p.name.startswith(".")]
        if not left:
            shutil.rmtree(kd, ignore_errors=True)
        log.info("deleted %d kept MSes in %s%s", n, kd, f" ({len(left)} not imaged here kept)" if left else "")


def _imager_has_archive(cfg: Config) -> bool:
    from . import hour_image
    try:
        p = subprocess.run([cfg.pipeline.hour_imaging.ttcalx_image, "hour", "--help"], capture_output=True,
                           text=True, timeout=120, env=hour_image.imager_env(cfg))
        return "--archive" in p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _archive_or_fallback(cfg: Config, band: int, group: str, lst_hour: int, ms: List[str]):
    """hour_source=archive: ("archive", files, []) if the archive is complete and sound, else
    ("kept_ms", files, problems) when the kept MSes cover the expected snapshots, else
    ("failed", files, problems); ("kept_ms", [], []) when there is nothing at all (no-data).

    Expected snapshots = the worker lists (<archive>.expected.jsonl: FITS archived) and the CLEAN
    snapshot FITS of the band-hour; the archive must hold exactly those (newest record per
    stem), structurally sound (no torn bytes, dropped index lines, unindexed records), of this
    band and phase centre, every record's crc good (checked here, so that a bad record falls back
    to the MSes; the imager checks the crc and the freqs again)."""
    from .v3 import hourstore
    files = hourstore.archive_files(cfg, band, group)
    expected = hourstore.expected_stems(cfg, band, group) or set()
    expected |= _fits_stems(cfg, band, lst_hour)
    kept = {Path(p).name[:-3] for p in ms}
    if not files and not expected and not kept:
        return "kept_ms", [], []                         # nothing to image (no-data, deletes nothing)
    problems: List[str] = []
    if not files:
        problems.append("no archive files")
    inv = hourstore.archive_inventory(files, band=band, phase_centre=lst.phase_centre(lst_hour), verify=True)
    problems += inv["problems"]
    got = set(inv["stems"])
    if files and got != expected:
        problems.append(f"archive has {len(got)} snapshots, expected {len(expected)}: missing "
                        f"{sorted(expected - got)[:10]}, unexpected {sorted(got - expected)[:10]}")
    if kept and kept != got:
        problems.append(f"kept MSes ({len(kept)}) and archive ({len(got)}) differ")
    if files and not problems and not _imager_has_archive(cfg):
        problems.append(f"{cfg.pipeline.hour_imaging.ttcalx_image} has no --archive (imager without gsiv)")
    if not problems:
        return "archive", files, []
    if kept and expected <= kept:
        log.error("ARCHIVE_UNUSABLE %sMHz %s: %s -- falling back to the %d kept MSes", band, group,
                  "; ".join(problems), len(kept))
        return "kept_ms", files, problems
    log.error("ARCHIVE_UNUSABLE %sMHz %s: %s -- and the kept MSes do not cover the %d expected snapshots",
              band, group, "; ".join(problems), len(expected))
    return "failed", files, problems


def _fits_stems(cfg: Config, band: int, lst_hour: int) -> set:
    """Stems of the band-hour's archived CLEAN snapshot FITS (<band>MHz-Clean-Snapshot-<ts>-image.fits.fz)."""
    d = _band_dir(cfg, band, lst_hour) / "snapshots_clean"
    if not d.is_dir():
        return set()
    out = set()
    for f in d.glob("*.fits.fz"):
        try:
            out.add(f"{lst.timestamp(f.name)}_{band}MHz")
        except ValueError:
            continue
    return out
