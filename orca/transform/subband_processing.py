"""Pure functions for subband processing steps.

These functions are called by the Celery tasks in subband_tasks.py but contain
NO Celery decorators — they are plain Python so they can be tested locally.

Adapted to use orca wrappers where available and NVMe paths from
subband_config.
"""
import os
import re
import sys
import glob
import shutil
import logging
import subprocess
import traceback
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import casacore.tables as pt
from casatasks import casalog, concat, flagdata
from astropy.time import Time
from astropy.io import fits
from astropy.stats import mad_std

try:
    import bdsf
    BDSF_AVAILABLE = True
except ImportError:
    BDSF_AVAILABLE = False

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  CASA log redirect
# ---------------------------------------------------------------------------
def redirect_casa_log(work_dir: str) -> None:
    """Point the casalog to a file inside the work directory."""
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'casa_pipeline.log')
    try:
        casalog.setlogfile(log_path)
        # Also set env var to catch subprocess CASA instances
        os.environ['CASALOGFILE'] = log_path
    except Exception as e:
        logger.warning(f"Failed to redirect CASA log: {e}")


# ---------------------------------------------------------------------------
#  File discovery
# ---------------------------------------------------------------------------
def find_archive_files_for_subband(
    start_dt: datetime,
    end_dt: datetime,
    subband: str,
    input_dir: Optional[str] = None,
) -> List[str]:
    """Locate averaged MS files for a subband within a time range.

    Searches either a custom *input_dir* or the default Lustre nighttime
    directory tree ``/lustre/pipeline/night-time/averaged/<subband>/<date>/<hour>/``.

    Args:
        start_dt: Start of the observation window (UTC).
        end_dt:   End of the observation window (UTC).
        subband:  Frequency label, e.g. ``'73MHz'``.
        input_dir: Optional override directory to search.

    Returns:
        Sorted list of absolute paths to matching ``.ms`` directories.
    """
    filename_pattern = re.compile(
        r'(\d{8})_(\d{6})_' + re.escape(subband) + r'(?:|_averaged)\.ms'
    )
    file_list: List[str] = []

    if input_dir:
        search_pattern = os.path.join(input_dir, f'*{subband}*.ms')
        for f_path in glob.glob(search_pattern):
            filename = os.path.basename(f_path)
            match = filename_pattern.search(filename)
            if match:
                date_str_file, time_str_file = match.groups()
                try:
                    file_start_dt = datetime.strptime(
                        date_str_file + time_str_file, '%Y%m%d%H%M%S'
                    )
                    if start_dt <= (file_start_dt + timedelta(seconds=5)) < end_dt:
                        file_list.append(f_path)
                except ValueError:
                    pass
    else:
        base_dir = '/lustre/pipeline/night-time/averaged/'
        current_hour = start_dt.replace(minute=0, second=0, microsecond=0)
        end_buffer = end_dt + timedelta(hours=1)
        while current_hour <= end_buffer:
            date_str = current_hour.strftime('%Y-%m-%d')
            hour_str = current_hour.strftime('%H')
            target_dir = os.path.join(base_dir, subband, date_str, hour_str)
            if os.path.isdir(target_dir):
                for f in os.listdir(target_dir):
                    match = filename_pattern.search(f)
                    if match:
                        date_str_file, time_str_file = match.groups()
                        try:
                            file_start_dt = datetime.strptime(
                                date_str_file + time_str_file, '%Y%m%d%H%M%S'
                            )
                            if start_dt <= (file_start_dt + timedelta(seconds=5)) < end_dt:
                                file_list.append(os.path.join(target_dir, f))
                        except ValueError:
                            pass
            current_hour += timedelta(hours=1)

    return sorted(file_list)


# ---------------------------------------------------------------------------
#  Copy MS files to NVMe
# ---------------------------------------------------------------------------
def copy_ms_to_nvme(src_ms: str, nvme_work_dir: str) -> str:
    """Copy a single MS directory to the NVMe work directory.

    Args:
        src_ms: Source path on Lustre.
        nvme_work_dir: Target directory on local NVMe.

    Returns:
        Path to the copied MS on NVMe.
    """
    dest = os.path.join(nvme_work_dir, os.path.basename(src_ms))
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src_ms, dest)
    logger.info(f"Copied {src_ms} → {dest}")
    return dest


# ---------------------------------------------------------------------------
#  SPW Mapping (from pipeline_utils.py)
# ---------------------------------------------------------------------------
def calculate_spwmap(
    ms_path: str, caltable_path: str
) -> Optional[List[int]]:
    """Compute the spectral-window mapping between an MS and a calibration table.

    For each SPW in the MS, finds the best-overlapping SPW in the cal table.
    Falls back to nearest-frequency match when there is no overlap.

    Args:
        ms_path: Path to the measurement set.
        caltable_path: Path to the calibration table.

    Returns:
        List of cal-table SPW indices (one per MS SPW), or None on failure.
    """
    try:
        with pt.table(os.path.join(ms_path, 'SPECTRAL_WINDOW'), ack=False) as t:
            ms_freqs = [t.getcell('CHAN_FREQ', i) for i in range(t.nrows())]
        with pt.table(os.path.join(caltable_path, 'SPECTRAL_WINDOW'), ack=False) as t:
            cal_freqs = [t.getcell('CHAN_FREQ', i) for i in range(t.nrows())]

        spwmap = []
        for ms_f in ms_freqs:
            ms_min, ms_max = np.min(ms_f), np.max(ms_f)
            best_match, best_overlap = -1, 0.0
            for cal_idx, cal_f in enumerate(cal_freqs):
                cal_min, cal_max = np.min(cal_f), np.max(cal_f)
                overlap = min(ms_max, cal_max) - max(ms_min, cal_min)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = cal_idx
            if best_match == -1:
                ms_center = np.mean(ms_f)
                diffs = [np.abs(np.mean(cf) - ms_center) for cf in cal_freqs]
                best_match = int(np.argmin(diffs))
            spwmap.append(best_match)
        return spwmap
    except Exception as e:
        logger.error(f"Error calculating SPW map: {e}")
        return None


# ---------------------------------------------------------------------------
#  Calibration application  (mirrors pipeline_utils.apply_calibration)
# ---------------------------------------------------------------------------
def apply_calibration(
    ms_path: str, bp_table: str, xy_table: str,
) -> bool:
    """Apply bandpass + XY-phase calibration tables to an MS.

    Also flags channels above 85 MHz if present.

    Args:
        ms_path: Path to the measurement set (modified in-place).
        bp_table: Path to bandpass calibration table.
        xy_table: Path to XY-phase calibration table.

    Returns:
        True on success, False on failure.
    """
    # Flag >85 MHz if applicable
    try:
        with pt.table(os.path.join(ms_path, "SPECTRAL_WINDOW"), ack=False) as t:
            freqs = t.getcol("CHAN_FREQ").ravel()
            if np.max(freqs) > 85e6:
                logger.info(f"Flagging >85 MHz in {os.path.basename(ms_path)}")
                flagdata(vis=ms_path, mode='manual', spw='*:85.0~100.0MHz', flagbackup=False)
    except Exception as e:
        logger.warning(f"Could not check >85 MHz flagging: {e}")

    bp_map = calculate_spwmap(ms_path, bp_table)
    xy_map = calculate_spwmap(ms_path, xy_table)
    if bp_map is None or xy_map is None:
        logger.error("Failed to map SPWs. Skipping calibration.")
        return False

    # Run calibration in a subprocess to avoid CASA lock issues
    python_code = f"""
import sys
from casatasks import clearcal, applycal
try:
    clearcal(vis='{ms_path}', addmodel=False)
    applycal(vis='{ms_path}', gaintable=['{bp_table}', '{xy_table}'],
             spwmap=[{bp_map}, {xy_map}], flagbackup=False, calwt=False)
except Exception as e:
    print(f"CASA Error: {{e}}")
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", python_code], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"Calibration subprocess failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Subprocess launch failed: {e}")
        return False


# ---------------------------------------------------------------------------
#  Antenna flagging via mnc_python  (mirrors pipeline_utils.run_antenna_flagging)
# ---------------------------------------------------------------------------
def get_bad_antenna_numbers(ms_path: str) -> List[int]:
    """Query mnc_python for bad antennas at the observation time.

    Requires the ``development`` conda environment with mnc_python installed.

    Args:
        ms_path: Path to MS; the observation time is read from the OBSERVATION table.

    Returns:
        List of bad correlator numbers (may be empty).
    """
    import astropy.time
    try:
        with pt.table(os.path.join(ms_path, 'OBSERVATION'), ack=False) as t:
            time_range = t.getcol('TIME_RANGE')[0]
            obs_mjd = astropy.time.Time(time_range[0], format='mjd', scale='utc').mjd
    except Exception:
        return []

    # The helper script lives in orca/utils/mnc_antennas.py; it needs the
    # 'development' conda env which has mnc and lwa_antpos installed.
    helper_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'utils', 'mnc_antennas.py'
    )
    if not os.path.exists(helper_script):
        logger.warning(f"MNC helper not found at {helper_script}; skipping auto-flagging")
        return []

    cmd = ['conda', 'run', '-n', 'development', 'python', helper_script, str(obs_mjd)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        lines = res.stdout.strip().split('\n')
        data = None
        for line in reversed(lines):
            try:
                data = json.loads(line)
                break
            except Exception:
                continue
        if data and data.get('bad_correlator_numbers'):
            return data['bad_correlator_numbers']
    except Exception as e:
        logger.error(f"MNC flagging helper failed: {e}")
    return []


def flag_bad_antennas(ms_path: str) -> str:
    """Flag bad antennas in an MS using mnc_python lookup.

    Args:
        ms_path: Path to the measurement set.

    Returns:
        The ms_path (for chaining).
    """
    bad_ants = get_bad_antenna_numbers(ms_path)
    if bad_ants:
        bad_ant_str = ",".join(map(str, bad_ants))
        logger.warning(f"Flagging bad antennas: {bad_ant_str}")
        flagdata(vis=ms_path, mode='manual', antenna=bad_ant_str, flagbackup=False)
    return ms_path


# ---------------------------------------------------------------------------
#  Auto-heal concatenation  (mirrors pipeline_utils.concatenate_with_auto_heal)
# ---------------------------------------------------------------------------
def _find_culprit(log_path: str) -> Optional[str]:
    """Parse a CASA concat log to find the MS that caused a failure."""
    if not os.path.exists(log_path):
        return None
    ms_pattern = re.compile(r"concatenating (.*\.ms) into")
    error_patterns = ["FilebufIO::readBlock", "BucketCache", "RuntimeError", "SEVERE"]
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if any(p in lines[i] for p in error_patterns):
                for j in range(i, -1, -1):
                    match = ms_pattern.search(lines[j])
                    if match:
                        return match.group(1).strip()
    except Exception as e:
        logger.error(f"Log parsing failed: {e}")
    return None


def concatenate_ms(
    ms_files: List[str], work_dir: str, subband: str, max_retries: int = 10,
) -> Optional[str]:
    """Concatenate MS files with automatic pruning of corrupt files.

    If concat fails, the CASA log is parsed to identify the culprit MS,
    which is removed from the list and concat is retried.

    Args:
        ms_files: List of MS paths to concatenate.
        work_dir: Working directory for the output.
        subband: Subband label (used in output filename).
        max_retries: Maximum number of retry attempts.

    Returns:
        Path to the concatenated MS, or None on total failure.
    """
    output_ms = os.path.join(work_dir, f"{subband}_concat.ms")
    os.environ['CASA_USE_NO_LOCKING'] = '1'
    current_list = sorted(list(ms_files))

    for attempt in range(max_retries):
        attempt_log = os.path.join(work_dir, f"concat_attempt_{attempt}.log")
        casalog.setlogfile(attempt_log)
        logger.info(f"Concat attempt {attempt}: merging {len(current_list)} files")
        try:
            if os.path.exists(output_ms):
                shutil.rmtree(output_ms)
            concat(vis=current_list, concatvis=output_ms, timesort=True)
            logger.info(f"Concat succeeded → {os.path.basename(output_ms)}")
            return output_ms
        except Exception:
            bad_ms = _find_culprit(attempt_log)
            if bad_ms and bad_ms in current_list:
                logger.error(f"AUTO-HEAL: pruning corrupt file {os.path.basename(bad_ms)}")
                current_list.remove(bad_ms)
                # Clean up scratch column dirs (restored from stable pipeline)
                for suffix in ["", "_with_scrcols"]:
                    path = bad_ms + suffix
                    if os.path.exists(path) and suffix != "":
                        shutil.rmtree(path)
            else:
                logger.error("Culprit identification failed, retrying...")
    return None


# ---------------------------------------------------------------------------
#  Field ID fix  (mirrors pipeline_utils.fix_field_id)
# ---------------------------------------------------------------------------
def fix_field_id(ms_path: str) -> bool:
    """Set all FIELD_ID values to 0 in the MS.

    Args:
        ms_path: Path to the measurement set.

    Returns:
        True on success.
    """
    try:
        with pt.table(ms_path, readonly=False, ack=False) as t:
            if 'FIELD_ID' in t.colnames():
                t.putcol('FIELD_ID', np.zeros(t.nrows(), dtype=np.int32))
        logger.info(f"Fixed FIELD_ID in {os.path.basename(ms_path)}")
        return True
    except Exception as e:
        logger.error(f"Failed to fix FIELD_ID: {e}")
        return False


# ---------------------------------------------------------------------------
#  Snapshot QA  (mirrors pipeline_utils.analyze_snapshot_quality)
# ---------------------------------------------------------------------------
def analyze_snapshot_quality(
    image_list: List[str],
) -> Tuple[List[int], List[Dict]]:
    """Analyse pilot snapshot images and flag bad integrations.

    Computes RMS in a central 1024×1024 box for each Stokes-V snapshot.
    Outliers (>3σ above median or <0.5× median) are flagged.

    Args:
        image_list: Sorted list of FITS snapshot paths.

    Returns:
        Tuple of (bad_indices, stats_list).
    """
    stats: List[Dict] = []
    for idx, img_path in enumerate(image_list):
        try:
            with fits.open(img_path) as hdul:
                data = hdul[0].data.squeeze()
                h, w = data.shape
                cw, ch = w // 2, h // 2
                box_r = 512
                center = data[ch - box_r:ch + box_r, cw - box_r:cw + box_r]
                rms = float(np.nanstd(center))
                peak = float(np.nanmax(np.abs(data)))
                stats.append({'idx': idx, 'rms': rms, 'peak': peak,
                              'file': os.path.basename(img_path)})
        except Exception:
            continue

    if not stats:
        return [], []

    rmses = np.array([s['rms'] for s in stats])
    med_rms = np.nanmedian(rmses)
    std_rms = mad_std(rmses, ignore_nan=True)
    high_thresh = med_rms + 3.0 * std_rms
    low_thresh = 0.5 * med_rms

    bad_indices = [
        s['idx'] for s in stats
        if s['rms'] > high_thresh or s['rms'] < low_thresh
    ]
    logger.info(
        f"QA: flagged {len(bad_indices)}/{len(stats)} integrations. "
        f"Median RMS={med_rms:.4f}"
    )
    return bad_indices, stats


def flag_bad_integrations(
    ms_path: str, bad_indices: List[int], n_total: int,
) -> None:
    """Flag bad scans using SCAN_NUMBER.

    Each snapshot index maps 1:1 to a scan in the MS.  We read the unique
    scan numbers, map bad_indices to scan numbers, and flag by scan.
    This is cleaner and more reliable than time-range flagging.

    Args:
        ms_path: Concatenated MS.
        bad_indices: Snapshot indices to flag.
        n_total: Total number of integrations.
    """
    if not bad_indices:
        return
    try:
        with pt.table(ms_path, ack=False) as t:
            scans = t.getcol("SCAN_NUMBER")
            unique_scans = sorted(set(scans))

        # Map snapshot indices to scan numbers
        bad_scans = []
        for idx in sorted(bad_indices):
            if idx < len(unique_scans):
                bad_scans.append(unique_scans[idx])
            else:
                logger.warning(
                    f"Snapshot index {idx} exceeds scan count ({len(unique_scans)})"
                )

        if not bad_scans:
            logger.warning("No valid scan numbers found for flagging.")
            return

        scan_str = ",".join(str(s) for s in bad_scans)
        logger.info(f"Applying QA flags on {len(bad_scans)} scans: {scan_str}")
        flagdata(vis=ms_path, mode='manual', scan=scan_str, flagbackup=False)
    except Exception as e:
        logger.error(f"Failed to apply QA flags: {e}")


# ---------------------------------------------------------------------------
#  Timestamp renaming  (mirrors pipeline_utils.add_timestamps_to_images)
# ---------------------------------------------------------------------------
def add_timestamps_to_images(
    target_dir: str, prefix: str, ms_path: str, n_intervals: int,
) -> bool:
    """Rename WSClean output images to include UTC timestamps.

    Args:
        target_dir: Directory containing the FITS images.
        prefix: WSClean filename prefix.
        ms_path: Concatenated MS (for time column).
        n_intervals: Number of time intervals produced.

    Returns:
        True on success.
    """
    try:
        with pt.table(ms_path, ack=False) as t:
            times = t.getcol("TIME")
            t_min, t_max = float(np.min(times)), float(np.max(times))
            duration = t_max - t_min

        files = glob.glob(os.path.join(target_dir, f"{prefix}*-*.fits"))
        if not files:
            return False

        for i in range(n_intervals):
            chunk_len = duration / n_intervals if n_intervals > 1 else duration
            mid_mjd_sec = t_min + (i * chunk_len) + (chunk_len / 2.0)
            ts_str = Time(mid_mjd_sec / 86400.0, format='mjd', scale='utc').datetime.strftime(
                "%Y%m%d_%H%M%S"
            )
            old_suffix = f"-t{i:04d}" if n_intervals > 1 else ""
            new_suffix = f"-{ts_str}"
            for f_path in files:
                f_name = os.path.basename(f_path)
                if n_intervals > 1:
                    if old_suffix in f_name:
                        new_name = f_name.replace(old_suffix, new_suffix)
                        shutil.move(f_path, os.path.join(target_dir, new_name))
                else:
                    if new_suffix not in f_name:
                        base, ext = os.path.splitext(f_name)
                        new_name = f"{base}{new_suffix}{ext}"
                        shutil.move(f_path, os.path.join(target_dir, new_name))
        return True
    except Exception as e:
        logger.error(f"Error renaming images: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
#  QA plotting  (mirrors pipeline_utils.plot_snapshot_diagnostics)
# ---------------------------------------------------------------------------
def plot_snapshot_diagnostics(
    stats: List[Dict], bad_indices: List[int], work_dir: str, subband: str,
) -> None:
    """Save a snapshot-RMS-vs-time diagnostic plot.

    Args:
        stats: List of per-integration stats dicts.
        bad_indices: Indices flagged as bad.
        work_dir: Working directory (QA subdir is used).
        subband: Subband label for title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not stats:
        return
    rmses = [s['rms'] for s in stats]
    indices = [s['idx'] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.plot(indices, rmses, 'b.-', label='RMS (Stokes V)')
    if bad_indices:
        bad_rmses = [
            stats[i]['rms'] for i in range(len(stats))
            if stats[i]['idx'] in bad_indices
        ]
        bad_idxs = [
            stats[i]['idx'] for i in range(len(stats))
            if stats[i]['idx'] in bad_indices
        ]
        plt.plot(bad_idxs, bad_rmses, 'rx', markersize=10, label='Flagged')
    plt.xlabel("Integration Index")
    plt.ylabel("Image RMS (Jy/beam)")
    plt.title(f"Snapshot Quality: {subband}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    qa_dir = os.path.join(work_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    plt.savefig(os.path.join(qa_dir, f"snapshot_rms_vs_time_{subband}.png"))
    plt.close()


# ---------------------------------------------------------------------------
#  CASA subprocess wrapper  (crash isolation — from new pipeline_utils.py)
# ---------------------------------------------------------------------------
def run_casa_task(work_dir: str, task_code: str) -> bool:
    """Execute a CASA task in a subprocess for crash isolation.

    Writes a temporary Python script and executes it, so that any CASA crash
    or table-locking issue does not bring down the Celery worker.

    Args:
        work_dir: Working directory (logs stored here).
        task_code: Python code string (indented for embedding in the wrapper).

    Returns:
        True on success, False on failure.
    """
    import time as _time
    timestamp = int(_time.time())
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    task_script = os.path.join(work_dir, f"casa_task_{timestamp}.py")
    log_path = os.path.join(log_dir, 'casa_pipeline.log')

    full_code = f"""
__casatask__ = True
import sys
import os
from casatasks import casalog
casalog.setlogfile('{log_path}')
try:
{task_code}
except Exception as e:
    print(f"INTERNAL_CASA_ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
    try:
        with open(task_script, "w") as f:
            f.write(full_code)
        result = subprocess.run(
            [sys.executable, task_script], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"CASA Subprocess Failure (Exit {result.returncode})")
            logger.error(f"Captured Traceback:\n{result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to launch CASA subprocess: {e}")
        return False
    finally:
        if os.path.exists(task_script):
            os.remove(task_script)


# ---------------------------------------------------------------------------
#  Primary beam correction
# ---------------------------------------------------------------------------
def apply_pb_correction_to_images(
    target_dir: str, base_prefix: str,
) -> int:
    """Apply primary beam correction to all image FITS files in a directory.

    Tries to import ``pb_correct.apply_pb_correction`` from the preliminary
    pipeline.  If not available, logs a warning and skips.

    Args:
        target_dir: Directory containing FITS images.
        base_prefix: WSClean filename prefix to match.

    Returns:
        Number of images successfully corrected.
    """
    try:
        from orca.transform.pb_correction import apply_pb_correction
    except ImportError:
        logger.warning(
            "orca.transform.pb_correction not importable — skipping PB correction.  "
            "Ensure extractor_pb_75 is installed on the cluster."
        )
        return 0

    count = 0
    for img in glob.glob(os.path.join(target_dir, f"{base_prefix}*image*.fits")):
        if "pbcorr" in img:
            continue
        try:
            apply_pb_correction(img)
            count += 1
        except Exception as e:
            logger.error(f"PB correction failed for {os.path.basename(img)}: {e}")
            traceback.print_exc()
    return count


# ---------------------------------------------------------------------------
#  Run subprocess wrapper
# ---------------------------------------------------------------------------
def run_subprocess(cmd: List[str], description: str) -> None:
    """Run a shell command and log start/finish.

    Args:
        cmd: Command and arguments.
        description: Human-readable label.

    Raises:
        subprocess.CalledProcessError on non-zero exit.
    """
    logger.info(f"START: {description}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"DONE: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description} (exit code {e.returncode})")
        raise


# ---------------------------------------------------------------------------
#  Deep image finder + BDSF source extraction (from ExoPipe pipeline_utils.py)
# ---------------------------------------------------------------------------
def find_deep_image(
    run_dir: str, freq_mhz: float, pol: str = 'I',
) -> Optional[str]:
    """Find the best deep tapered image for a subband.

    Searches ``<run_dir>/<freq>MHz/<pol>/deep/`` and falls back to
    ``<run_dir>/<pol>/deep/`` for PB-corrected tapered images.

    Args:
        run_dir: Working directory (NVMe or archive).
        freq_mhz: Subband centre frequency in MHz.
        pol: Stokes parameter ('I' or 'V').

    Returns:
        Path to the best deep image, or None.
    """
    pat_root = os.path.join(run_dir, f"{int(freq_mhz)}MHz", pol, "deep",
                            "*Taper*pbcorr.fits")
    pat_local = os.path.join(run_dir, pol, "deep", "*Taper*pbcorr.fits")
    candidates = glob.glob(pat_root) + glob.glob(pat_local)
    if not candidates:
        pat_root_raw = os.path.join(run_dir, f"{int(freq_mhz)}MHz", pol, "deep",
                                    "*Taper*image.fits")
        pat_local_raw = os.path.join(run_dir, pol, "deep", "*Taper*image.fits")
        candidates = glob.glob(pat_root_raw) + glob.glob(pat_local_raw)
    candidates = [c for c in candidates if "NoTaper" not in c]
    if candidates:
        return sorted(candidates)[0]
    return None


def extract_sources_to_df(
    filename: str, thresh_pix: float = 10.0,
) -> 'pd.DataFrame':
    """Extract sources from a FITS image using PyBDSF.

    Args:
        filename: Path to FITS image.
        thresh_pix: Detection threshold in pixels.

    Returns:
        DataFrame with columns ra, dec, flux_peak_I_app, maj, min.
        Empty DataFrame on failure.
    """
    if not BDSF_AVAILABLE:
        logger.warning("bdsf not available — cannot extract sources")
        return pd.DataFrame()
    try:
        logger.info(
            f"Extracting sources from {os.path.basename(filename)} "
            f"(thresh_pix={thresh_pix:.0f})..."
        )
        img = bdsf.process_image(
            filename, thresh_pix=thresh_pix, thresh_isl=5.0,
            adaptive_rms_box=True, quiet=True,
        )
        sources_raw = []
        for s in img.sources:
            if not np.isnan(s.posn_sky_max[0]):
                sources_raw.append({
                    'ra': s.posn_sky_max[0],
                    'dec': s.posn_sky_max[1],
                    'flux_peak_I_app': s.peak_flux_max,
                    'maj': getattr(s, 'maj_axis', 0.0),
                    'min': getattr(s, 'min_axis', 0.0),
                })
        if not sources_raw:
            logger.warning(f"No sources found in {filename}")
            return pd.DataFrame()
        return pd.DataFrame(sources_raw)
    except Exception as e:
        logger.error(f"BDSF extraction failed for {filename}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
#  Archive results to Lustre
# ---------------------------------------------------------------------------
def archive_results(
    work_dir: str,
    archive_base: str,
    subband: str = '',
    cleanup_concat: bool = True,
    cleanup_workdir: bool = False,
) -> str:
    """Copy pipeline products from NVMe work_dir to Lustre archive.

    Copies subdirectories I/, V/, snapshots/, QA/, samples/, detections/,
    Movies/, Dewarp_Diagnostics/ and loose files.
    Also writes to the centralised ``samples/`` and ``detections/`` trees
    under ``LUSTRE_ARCHIVE_DIR`` so that products from many runs are
    aggregated in one place.

    Args:
        work_dir: NVMe working directory.
        archive_base: Lustre destination directory.
        subband: Subband label (e.g. '73MHz') — used for centralized archive paths.
        cleanup_concat: Whether to remove concat MS on NVMe.
        cleanup_workdir: Whether to remove the entire work_dir after archiving.
            Supersedes cleanup_concat when True.

    Returns:
        The archive_base path.
    """
    os.makedirs(archive_base, exist_ok=True)
    logger.info(f"Archiving results → {archive_base}")

    for top_level in ['I', 'V', 'snapshots', 'QA', 'samples', 'detections',
                      'Movies', 'Dewarp_Diagnostics']:
        src = os.path.join(work_dir, top_level)
        dest = os.path.join(archive_base, top_level)
        if os.path.exists(src):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

    # Copy loose files (logs, debug FITS, etc.)
    for f in glob.glob(os.path.join(work_dir, "*")):
        if os.path.isfile(f):
            shutil.copy(f, archive_base)

    # --- Centralized samples archive ---
    # /lustre/pipeline/images/samples/{sample_name}/{target_name}/{subband}/
    samples_src = os.path.join(work_dir, "samples")
    if os.path.exists(samples_src) and subband:
        lustre_archive_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(archive_base)))
        lustre_samples_root = os.path.join(lustre_archive_dir, "samples")
        for sample_dir in glob.glob(os.path.join(samples_src, "*")):
            if not os.path.isdir(sample_dir):
                continue
            sample_name = os.path.basename(sample_dir)
            for target_dir in glob.glob(os.path.join(sample_dir, "*")):
                if not os.path.isdir(target_dir):
                    # Loose file (e.g. photometry CSV) at sample level
                    dest_dir = os.path.join(
                        lustre_samples_root, sample_name, subband)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(target_dir, dest_dir)
                    continue
                target_name = os.path.basename(target_dir)
                dest_dir = os.path.join(
                    lustre_samples_root, sample_name, target_name, subband)
                os.makedirs(dest_dir, exist_ok=True)
                for f in glob.glob(os.path.join(target_dir, "*")):
                    if os.path.isfile(f):
                        shutil.copy(f, dest_dir)
                    elif os.path.isdir(f):
                        dest_sub = os.path.join(dest_dir, os.path.basename(f))
                        if os.path.exists(dest_sub):
                            shutil.rmtree(dest_sub)
                        shutil.copytree(f, dest_sub)
        logger.info(f"Samples archived to {lustre_samples_root}")

    # --- Centralized detections archive ---
    # Transients:    .../detections/transients/{I,V}/{J-name}/{subband}/
    # SolarSystem:   .../detections/SolarSystem/{Body}/{subband}/
    # Target dets:   .../detections/{sample_name}/{target_name}/{subband}/
    detections_src = os.path.join(work_dir, "detections")
    if os.path.exists(detections_src) and subband:
        lustre_archive_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(archive_base)))
        lustre_det_root = os.path.join(lustre_archive_dir, "detections")

        # Transients
        for stokes in ['I', 'V']:
            trans_src = os.path.join(detections_src, "transients", stokes)
            if not os.path.exists(trans_src):
                continue
            for jdir in glob.glob(os.path.join(trans_src, "*")):
                if not os.path.isdir(jdir):
                    continue
                jname = os.path.basename(jdir)
                dest_dir = os.path.join(
                    lustre_det_root, "transients", stokes, jname, subband)
                os.makedirs(dest_dir, exist_ok=True)
                for f in glob.glob(os.path.join(jdir, "*")):
                    if os.path.isfile(f):
                        shutil.copy(f, dest_dir)

        # Transient debug files
        debug_src = os.path.join(detections_src, "transients", "debug")
        if os.path.exists(debug_src):
            dest_dir = os.path.join(
                lustre_det_root, "transients", "debug", subband)
            os.makedirs(dest_dir, exist_ok=True)
            for f in glob.glob(os.path.join(debug_src, "*")):
                if os.path.isfile(f):
                    shutil.copy(f, dest_dir)

        # Solar System
        ss_src = os.path.join(detections_src, "SolarSystem")
        if os.path.exists(ss_src):
            for body_dir in glob.glob(os.path.join(ss_src, "*")):
                if not os.path.isdir(body_dir):
                    continue
                body = os.path.basename(body_dir)
                dest_dir = os.path.join(
                    lustre_det_root, "SolarSystem", body, subband)
                os.makedirs(dest_dir, exist_ok=True)
                for f in glob.glob(os.path.join(body_dir, "*")):
                    if os.path.isfile(f):
                        shutil.copy(f, dest_dir)

        # Target-based detections (sample_name/target_name subdirs)
        for item in glob.glob(os.path.join(detections_src, "*")):
            bn = os.path.basename(item)
            if bn in ('transients', 'SolarSystem', 'debug'):
                continue
            if os.path.isdir(item):
                for target_dir_path in glob.glob(os.path.join(item, "*")):
                    if os.path.isdir(target_dir_path):
                        target_name = os.path.basename(target_dir_path)
                        dest_dir = os.path.join(
                            lustre_det_root, bn, target_name, subband)
                        os.makedirs(dest_dir, exist_ok=True)
                        for f in glob.glob(os.path.join(target_dir_path, "*")):
                            if os.path.isfile(f):
                                shutil.copy(f, dest_dir)

        logger.info(f"Detections archived to {lustre_det_root}")

    if cleanup_workdir:
        shutil.rmtree(work_dir)
        logger.info(f"Cleaned up entire work_dir: {work_dir}")
    elif cleanup_concat:
        for ms in glob.glob(os.path.join(work_dir, "*_concat.ms")):
            if os.path.exists(ms):
                shutil.rmtree(ms)
                logger.info(f"Cleaned up {ms}")

    return archive_base
