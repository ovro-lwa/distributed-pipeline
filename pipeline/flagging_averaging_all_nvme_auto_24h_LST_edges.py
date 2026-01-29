#!/usr/bin/env python3
"""Automated 24-hour flagging and averaging with NVMe acceleration.

This script processes a full day of slow-cadence data, applying AOFlagger
RFI flagging and frequency averaging. Uses NVMe for fast I/O and handles
LST edge cases for proper time binning.

The pipeline:
    1. Scans slow data directories for available dates.
    2. Filters by LST range to avoid problematic edge times.
    3. Copies to NVMe for fast processing.
    4. Runs AOFlagger and frequency averaging.
    5. Archives results to Lustre.
"""
import os
import sys
import glob
import time
import shutil
import logging
import random
import datetime
import math
from typing import List, Tuple, Optional

from celery import group

from astropy.utils.iers import conf
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from orca.tasks.pipeline_tasks import run_pipeline_slow_on_one_cpu_nvme


# ------------------------------------------------------------------
# PATH CONFIG
# ------------------------------------------------------------------
ROOT_SLOW      = "/lustre/pipeline/slow"
SLOW_AVG_BASE  = "/lustre/pipeline/slow-averaged"
NIGHT_AVG_BASE = "/lustre/pipeline/night-time/averaged"

FREQ_DIRS = [
    "13MHz","18MHz","23MHz","27MHz","32MHz","36MHz",
    "41MHz","46MHz","50MHz","55MHz","59MHz","64MHz",
    "69MHz","73MHz","78MHz","82MHz"
]

# Your "core" UTC window (same meaning as range(4, 13) => 04..12 inclusive)
BASE_UTC_START_HOUR = 4
BASE_UTC_END_HOUR   = 12  # inclusive

# We'll search one hour earlier/later to "complete" the LST edge-hours
EXT_UTC_START_HOUR = BASE_UTC_START_HOUR - 1  # 03
EXT_UTC_END_HOUR   = BASE_UTC_END_HOUR + 1    # 13

# Which subband to use for computing the LST boundaries (fallbacks if missing)
REFERENCE_FREQ = "73MHz"

# OVRO-LWA / OVRO site (approx; good to a few seconds of LST which is fine for 10s files)
OVRO_LOCATION = EarthLocation.from_geodetic(
    lon=-118.2817 * u.deg,
    lat=37.2398 * u.deg,
    height=1222 * u.m,
)

# Assume each MS represents ~10s integration; used only for a safe end_dt when needed
MS_STEP_SECONDS = 10


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
LOG_AUTO = "pipeline_auto.log"
LOG_ERR  = "pipeline_errors.log"

conf.auto_download = False
conf.iers_degraded_accuracy = "warn"

logging.basicConfig(
    filename=LOG_AUTO,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

err_logger = logging.getLogger("errors")
eh = logging.FileHandler(LOG_ERR)
eh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
err_logger.addHandler(eh)
err_logger.setLevel(logging.ERROR)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def _parse_dt_from_ms_filename(ms_path: str) -> datetime.datetime:
    """
    Parse UTC datetime from MS filename like:
        20251226_040003_13MHz.ms

    IMPORTANT: uses filename ONLY, per your request.
    """
    base = os.path.basename(ms_path)
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unrecognized MS filename (no _ parts): {base}")

    dt_str = f"{parts[0]}_{parts[1]}"  # YYYYMMDD_HHMMSS
    dt = datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)
    return dt


def _list_ms_records(freq: str, date_str: str, hours: range) -> List[Tuple[datetime.datetime, str]]:
    """
    Return [(dt_utc, ms_path), ...] sorted by dt_utc for the given freq/date/hours.
    """
    recs: List[Tuple[datetime.datetime, str]] = []
    for h in hours:
        path = os.path.join(ROOT_SLOW, freq, date_str, f"{h:02d}")
        for ms in glob.glob(os.path.join(path, "*.ms")):
            try:
                dt = _parse_dt_from_ms_filename(ms)
            except Exception as e:
                err_logger.error(f"Failed to parse datetime from {ms}: {e}")
                continue
            recs.append((dt, ms))

    recs.sort(key=lambda x: x[0])
    return recs


def _lst_hours(dt_utc: datetime.datetime) -> float:
    """
    Return apparent Local Sidereal Time at OVRO in hours in [0, 24).
    """
    t = Time(dt_utc, scale="utc", location=OVRO_LOCATION)
    lst = t.sidereal_time("apparent")
    # lst is an Angle; .hour gives hours in [0,24)
    return float(lst.hour)


def _unwrap_hours(hours_0_24: List[float]) -> List[float]:
    """
    Unwrap a sequence of LST hours so it is continuous/monotonic across 24h wrap.
    E.g. [23.9, 0.1, 0.2] -> [23.9, 24.1, 24.2]
    """
    if not hours_0_24:
        return []

    unwrapped = [hours_0_24[0]]
    offset = 0.0
    prev = unwrapped[0]

    for x in hours_0_24[1:]:
        x_adj = x + offset
        if x_adj < prev - 12.0:
            offset += 24.0
            x_adj = x + offset
        elif x_adj > prev + 12.0:
            offset -= 24.0
            x_adj = x + offset

        unwrapped.append(x_adj)
        prev = x_adj

    return unwrapped


def _pick_reference_freq(date_str: str) -> Optional[str]:
    """
    Prefer REFERENCE_FREQ if it has data in the base window, else pick first freq that does.
    """
    base_hours = range(BASE_UTC_START_HOUR, BASE_UTC_END_HOUR + 1)
    # Try preferred first
    if REFERENCE_FREQ in FREQ_DIRS:
        recs = _list_ms_records(REFERENCE_FREQ, date_str, base_hours)
        if recs:
            return REFERENCE_FREQ

    # Fallback: first freq with base-window data
    for f in FREQ_DIRS:
        recs = _list_ms_records(f, date_str, base_hours)
        if recs:
            return f

    return None


def _compute_full_lst_utc_window(date_str: str) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Compute a UTC [start_dt, end_dt) window (inclusive/exclusive) that corresponds to
    full integer LST-hour blocks covering the LST span of the base UTC window.

    Uses reference subband's files in EXT window to locate where LST crosses the boundaries.
    """
    ref = _pick_reference_freq(date_str)
    if ref is None:
        logging.warning(f"No reference freq found with data in base window for {date_str}")
        return None

    ext_hours = range(EXT_UTC_START_HOUR, EXT_UTC_END_HOUR + 1)
    recs_ext = _list_ms_records(ref, date_str, ext_hours)
    if not recs_ext:
        logging.warning(f"No MS files found for reference {ref} in ext window for {date_str}")
        return None

    # Identify first/last records that are in the base UTC window
    idxs_base = [
        i for i, (dt, _) in enumerate(recs_ext)
        if (dt.date().isoformat() == date_str) and (BASE_UTC_START_HOUR <= dt.hour <= BASE_UTC_END_HOUR)
    ]
    if not idxs_base:
        logging.warning(f"Reference {ref}: no files in BASE window for {date_str}")
        return None

    i0 = idxs_base[0]
    i1 = idxs_base[-1]

    # Compute LST for all ext records (reference only), then unwrap
    lst_raw = []
    for dt, _ in recs_ext:
        lst_raw.append(_lst_hours(dt))
    lst_unw = _unwrap_hours(lst_raw)

    lst_start_u = lst_unw[i0]
    lst_end_u   = lst_unw[i1]

    start_boundary = math.floor(lst_start_u)  # integer LST hour at/before start
    end_boundary   = math.ceil(lst_end_u)     # integer LST hour at/after end

    # Find first record at/after start_boundary
    start_idx = 0
    for i, v in enumerate(lst_unw):
        if v >= start_boundary:
            start_idx = i
            break

    # Find first record at/after end_boundary (exclusive end)
    end_idx = len(recs_ext)
    for i, v in enumerate(lst_unw):
        if v >= end_boundary:
            end_idx = i
            break

    start_dt = recs_ext[start_idx][0]
    if end_idx < len(recs_ext):
        end_dt = recs_ext[end_idx][0]
    else:
        # Not enough data to reach end boundary; include everything we have (best-effort)
        last_dt = recs_ext[-1][0]
        end_dt = last_dt + datetime.timedelta(seconds=MS_STEP_SECONDS)
        logging.warning(
            f"Reference {ref}: did not reach end LST boundary for {date_str}. "
            f"Using end_dt={end_dt.isoformat()} (best-effort)."
        )

    # Logging context (print boundaries modulo 24 for readability)
    logging.info(
        f"[{date_str}] Ref={ref} base LST start={lst_start_u:.3f}h end={lst_end_u:.3f}h "
        f"=> boundaries [{start_boundary % 24:02d}:00, {end_boundary % 24:02d}:00) "
        f"UTC window [{start_dt.isoformat()} , {end_dt.isoformat()})"
    )
    return start_dt, end_dt


# ------------------------------------------------------------------
# MAIN ACTIONS
# ------------------------------------------------------------------
def submit_tasks_for_date_full_lst(date_str: str) -> int:
    """
    Queue Celery tasks for all freqs for date_str, but only MS whose filename-UTC time
    falls inside the computed full-LST UTC window.

    Returns number of submitted tasks.
    """
    window = _compute_full_lst_utc_window(date_str)
    if window is None:
        logging.warning(f"[{date_str}] Could not compute LST window; submitting BASE UTC hours only as fallback.")
        # Fallback: old behavior (base hours only)
        hours = range(BASE_UTC_START_HOUR, BASE_UTC_END_HOUR + 1)
        sigs = []
        for freq in FREQ_DIRS:
            recs = _list_ms_records(freq, date_str, hours)
            for _, ms in recs:
                # Explicitly pass start/end so hour filtering stays consistent
                sigs.append(run_pipeline_slow_on_one_cpu_nvme.s(ms, start=BASE_UTC_START_HOUR, end=BASE_UTC_END_HOUR))
        if sigs:
            random.shuffle(sigs)
            group(sigs).apply_async()
        return len(sigs)

    start_dt, end_dt = window

    ext_hours = range(EXT_UTC_START_HOUR, EXT_UTC_END_HOUR + 1)
    sigs = []

    for freq in FREQ_DIRS:
        recs = _list_ms_records(freq, date_str, ext_hours)
        n_before = len(sigs)
        for dt, ms in recs:
            # Guard against weird spillover filenames (paranoia)
            if dt.date().isoformat() != date_str:
                continue
            if start_dt <= dt < end_dt:
                # Pass explicit hour filter to the task; include the extra edge hours
                sigs.append(
                    run_pipeline_slow_on_one_cpu_nvme.s(
                        ms,
                        start=EXT_UTC_START_HOUR,
                        end=EXT_UTC_END_HOUR,
                    )
                )
        logging.info(f"[{date_str}] {freq}: selected {len(sigs) - n_before} MS in full-LST window")

    if not sigs:
        logging.warning(f"[{date_str}] No MS selected for any freq")
        return 0

    random.shuffle(sigs)
    logging.info(f"[{date_str}] Submitting {len(sigs)} Celery tasks (full-LST window)")
    try:
        group(sigs).apply_async()  # fire-and-forget
        logging.info(f"[{date_str}] ✔ Celery submission completed")
    except Exception as e:
        err_logger.error(f"[{date_str}] Celery submit error: {e}")
        logging.error(f"[{date_str}] ✖ Celery submission errored")

    return len(sigs)


def move_averaged_for_date(date_str: str):
    """Move slow-averaged/<freq>/<date_str> → night-time/averaged/<freq>/<date_str>."""
    for freq in FREQ_DIRS:
        src = os.path.join(SLOW_AVG_BASE, freq, date_str)
        dst = os.path.join(NIGHT_AVG_BASE, freq, date_str)
        if os.path.isdir(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                logging.info(f"Moved {src} → {dst}")
            except Exception as e:
                err_logger.error(f"Move failed for {src}: {e}")
                logging.error(f"✖ Move failed for {date_str}/{freq}")
        else:
            logging.info(f"No dir to move at {src} (maybe not finished yet)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 flagging_averaging_all_nvme_auto_24h.py YYYY-MM-DD")
        sys.exit(1)

    try:
        current_date = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    except ValueError:
        print("Error: date must be in YYYY-MM-DD format")
        sys.exit(1)

    logging.info("=== AUTO PIPELINE STARTUP (FULL-LST SUBMISSION) ===")
    logging.info(
        f"BASE UTC window: {BASE_UTC_START_HOUR:02d}..{BASE_UTC_END_HOUR:02d} "
        f"(ext search {EXT_UTC_START_HOUR:02d}..{EXT_UTC_END_HOUR:02d}), ref={REFERENCE_FREQ}"
    )

    last_submitted_date = None

    while True:
        tick_start = time.monotonic()

        if last_submitted_date is not None:
            logging.info(f"--- START moving for {last_submitted_date} ---")
            move_averaged_for_date(last_submitted_date)
            logging.info(f"--- DONE moving for {last_submitted_date} ---")

        date_str = current_date.isoformat()
        logging.info(f"--- START submission for {date_str} ---")
        n_tasks = submit_tasks_for_date_full_lst(date_str)
        logging.info(f"--- DONE submission for {date_str} (tasks={n_tasks}) ---\n")

        last_submitted_date = date_str
        current_date += datetime.timedelta(days=1)

        elapsed = time.monotonic() - tick_start
        sleep_s = max(0.0, 86400.0 - elapsed)
        logging.info(f"Sleeping {sleep_s:.1f}s until next run (elapsed this tick {elapsed:.1f}s)")
        time.sleep(sleep_s)

