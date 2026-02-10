#!/usr/bin/env python3
"""Submit subband processing pipelines to Celery.

Replaces the previous Slurm-based pipeline controller.
Discovers MS files, computes LST segments, and submits one chord per
(subband, LST-hour) combination to the correct calim node queue.

Usage examples
--------------
Process one subband on one calim server (testing)::

    python subband_celery.py \\
        --range 14-15 --date 2025-06-15 \\
        --bp_table /lustre/gh/calibration/pipeline/bandpass/73MHz/latest.bandpass \\
        --xy_table /lustre/gh/calibration/pipeline/xy/73MHz/latest.X \\
        --subbands 73MHz \\
        --peel_sky --peel_rfi

Process all subbands for a full observation::

    python subband_celery.py \\
        --range 13-19 --date 2025-06-15 \\
        --bp_table /lustre/gh/calibration/pipeline/bandpass/latest.bandpass \\
        --xy_table /lustre/gh/calibration/pipeline/xy/latest.X \\
        --peel_sky --peel_rfi --hot_baselines
"""
import argparse
import logging
import sys
from datetime import datetime

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from orca.tasks.subband_tasks import submit_subband_pipeline
from orca.transform.subband_processing import find_archive_files_for_subband
from orca.resources.subband_config import (
    NODE_SUBBAND_MAP,
    get_queue_for_subband,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CONTROLLER] - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

OVRO_LOC = EarthLocation(
    lat=37.239780 * u.deg, lon=-118.276250 * u.deg, height=1222 * u.m,
)


# ---------------------------------------------------------------------------
#  LST scheduling  (ported from pipeline_controller.py)
# ---------------------------------------------------------------------------

def parse_time_range(range_str: str, date_str: str):
    """Convert an LST range like '14-15' + a date to a UTC (start, end) pair.

    Also accepts explicit UTC:  '2025-06-15:12:00:00,2025-06-15:18:00:00'

    Args:
        range_str: Either 'LSTstart-LSTend' (hours) or 'UTC_start,UTC_end'.
        date_str: Reference date 'YYYY-MM-DD' (required for LST mode).

    Returns:
        Tuple of (astropy.time.Time start, astropy.time.Time end).
    """
    # Explicit UTC pair
    if ',' in range_str and ':' in range_str:
        s, e = range_str.split(',')
        t_start = Time(datetime.strptime(s, '%Y-%m-%d:%H:%M:%S'),
                       scale='utc', location=OVRO_LOC)
        t_end = Time(datetime.strptime(e, '%Y-%m-%d:%H:%M:%S'),
                     scale='utc', location=OVRO_LOC)
        return t_start, t_end

    # LST range
    clean = range_str.lower().replace('h', '')
    lst_start, lst_end = map(float, clean.split('-'))
    ref_date = datetime.strptime(date_str, '%Y-%m-%d')
    t_ref = Time(ref_date, scale='utc', location=OVRO_LOC)
    lst_ref = t_ref.sidereal_time('mean').hour

    diff_start = (lst_start - lst_ref) % 24
    diff_end = (lst_end - lst_ref) % 24
    if diff_end < diff_start:
        diff_end += 24

    sidereal_factor = 0.99726958
    t_start = t_ref + (diff_start * u.hour * sidereal_factor)
    t_end = t_ref + (diff_end * u.hour * sidereal_factor)
    logger.info(f"UTC range: {t_start.isot} → {t_end.isot}")
    return t_start, t_end


def generate_lst_segments(t_start, t_end, override=False):
    """Split a time range into 1-LST-hour segments.

    Args:
        t_start: Start time.
        t_end: End time.
        override: If True, return a single segment (no splitting).

    Returns:
        List of dicts with 'start', 'end', 'lst_label' keys.
    """
    if override:
        return [{'start': t_start, 'end': t_end, 'lst_label': 'custom'}]

    jobs = []
    current_t = t_start
    while (t_end - current_t).sec > 10.0:
        current_lst = current_t.sidereal_time('mean').hour
        next_lst_hour = np.floor(current_lst) + 1.0
        dt_solar = (next_lst_hour - current_lst) * 0.99726958 * u.hour
        segment_end = min(current_t + dt_solar, t_end)

        if (segment_end - current_t).sec < 10.0:
            current_t = segment_end
            continue

        midpoint = current_t + (segment_end - current_t) / 2
        mid_lst = int(np.floor(midpoint.sidereal_time('mean').hour)) % 24
        jobs.append({
            'start': current_t,
            'end': segment_end,
            'lst_label': f"{mid_lst:02d}h",
        })
        current_t = segment_end

    return jobs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Submit subband processing to Celery workers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--range', required=True,
                        help="LST range e.g. '14-15' or explicit UTC pair")
    parser.add_argument('--date',
                        help="Reference date YYYY-MM-DD (required for LST range)")
    parser.add_argument('--bp_table', required=True,
                        help='Path to bandpass calibration table')
    parser.add_argument('--xy_table', required=True,
                        help='Path to XY-phase calibration table')
    parser.add_argument('--subbands', nargs='+', default=None,
                        help='Subbands to process (default: all from NODE_SUBBAND_MAP)')
    parser.add_argument('--input_dir', default=None,
                        help='Override input directory for MS files')
    parser.add_argument('--run_label', default=None,
                        help='Run label (default: auto-generated)')
    parser.add_argument('--peel_sky', action='store_true')
    parser.add_argument('--peel_rfi', action='store_true')
    parser.add_argument('--hot_baselines', action='store_true')
    parser.add_argument('--override_range', action='store_true',
                        help='Do not split into LST-hour segments')
    parser.add_argument('--skip_cleanup', action='store_true',
                        help='Keep intermediate files on NVMe')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be submitted without actually submitting')
    args = parser.parse_args()

    run_label = args.run_label or f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== Pipeline Start === Run label: {run_label}")

    # Parse time range
    t_start, t_end = parse_time_range(args.range, args.date)
    segments = generate_lst_segments(t_start, t_end, args.override_range)

    # Determine subbands
    if args.subbands:
        subbands = args.subbands
    else:
        # Priority order like the original controller
        priority = ['73MHz', '78MHz', '69MHz', '82MHz']
        all_subs = list(NODE_SUBBAND_MAP.keys())
        subbands = priority + [s for s in all_subs if s not in priority]

    results = []

    for seg in segments:
        obs_date = seg['start'].datetime.strftime('%Y-%m-%d')
        lst_label = seg['lst_label']
        start_dt = seg['start'].datetime
        end_dt = seg['end'].datetime

        for subband in subbands:
            queue = get_queue_for_subband(subband)
            node = NODE_SUBBAND_MAP[subband]

            # Discover MS files
            ms_files = find_archive_files_for_subband(
                start_dt, end_dt, subband, input_dir=args.input_dir,
            )

            if not ms_files:
                logger.warning(
                    f"No files for {subband} in {lst_label} "
                    f"({start_dt} → {end_dt})"
                )
                continue

            logger.info(
                f"Submitting {subband} | {lst_label} | {len(ms_files)} files "
                f"→ {node} (queue={queue})"
            )

            if args.dry_run:
                for ms in ms_files:
                    logger.info(f"  [DRY RUN] {ms}")
                continue

            result = submit_subband_pipeline(
                ms_files=ms_files,
                subband=subband,
                bp_table=args.bp_table,
                xy_table=args.xy_table,
                lst_label=lst_label,
                obs_date=obs_date,
                run_label=run_label,
                peel_sky=args.peel_sky,
                peel_rfi=args.peel_rfi,
                hot_baselines=args.hot_baselines,
                skip_cleanup=args.skip_cleanup,
            )
            results.append({
                'subband': subband,
                'lst_label': lst_label,
                'node': node,
                'n_files': len(ms_files),
                'result': result,
            })

    # Summary
    logger.info(f"=== Submitted {len(results)} subband jobs ===")
    for r in results:
        logger.info(
            f"  {r['subband']:>6s} | {r['lst_label']} | "
            f"{r['n_files']:3d} files → {r['node']}"
        )

    if results and not args.dry_run:
        logger.info(
            "Monitor progress with:\n"
            "  celery -A orca.celery flower --port=5555\n"
            "  ssh -L 5555:localhost:5555 <user>@lwacalim10\n"
            "  → http://localhost:5555"
        )


if __name__ == '__main__':
    main()
