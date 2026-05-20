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
import os
import sys
from datetime import datetime

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from orca.tasks.subband_tasks import (
    submit_subband_pipeline,
    submit_subband_pipeline_chained,
    _push_work_units,
    _pop_and_submit,
    _dynamic_queue_length,
)
from orca.transform.subband_processing import find_archive_files_for_subband
from orca.resources.subband_config import (
    NODE_SUBBAND_MAP,
    DYNAMIC_NODE_POOL,
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
    parser.add_argument('--cleanup_nvme', action='store_true',
                        help='Remove entire NVMe work_dir after archiving to Lustre')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Target list files for photometry (one or more paths)')
    parser.add_argument('--catalog', default=None,
                        help='BDSF catalog file for transient search masking')
    parser.add_argument('--clean_snapshots', action='store_true',
                        help='Produce CLEANed Stokes-I snapshots (in addition to '
                             'dirty pilots) in snapshots_clean/. Uses optimised '
                             'wsclean params (auto-mask=5, mgain=0.9999). '
                             'Compressed with fpack automatically.')
    parser.add_argument('--clean_reduced_pixels', action='store_true',
                        help='(Deprecated, no-op) Per-subband pixel scaling is '
                             'now always applied to all imaging.')
    parser.add_argument('--skip_science', action='store_true',
                        help='Stop after imaging + PB correction; skip dewarping, '
                             'photometry, transient search, flux check. '
                             'Still archives products to Lustre.')
    parser.add_argument('--reduced_pixels', action='store_true',
                        help='(Deprecated, no-op) Per-subband pixel scaling is '
                             'now always applied to all imaging.')
    parser.add_argument('--compress_snapshots', action='store_true',
                        help='Compress snapshot FITS with fpack (.fits → .fits.fz). '
                             'Originals are deleted. Deep images are NOT compressed.')
    parser.add_argument('--snapshot_only', action='store_true',
                        help='Lightweight mode: skip pilot V, deep imaging, V movies, '
                             'QA, and science. Only produce clean Stokes-I snapshots '
                             'and I movies (Raw + Filtered). For reprocessing old dates.')
    parser.add_argument('--remap', nargs='+', default=None, metavar='SUBBAND=NODE',
                        help='Override node routing, e.g. --remap 18MHz=calim08 23MHz=calim08')
    parser.add_argument('--dynamic', action='store_true',
                        help='Dynamic scheduling: work units are dispatched to '
                             'whichever node finishes first. Incompatible with --remap.')
    parser.add_argument('--nodes', nargs='+', default=None, metavar='NODE',
                        help='Node pool for --dynamic (default: DYNAMIC_NODE_POOL from config). '
                             'e.g. --nodes calim01 calim03 calim08')
    parser.add_argument('--exclude_nodes', nargs='+', default=None, metavar='NODE',
                        help='Exclude nodes from --dynamic pool, e.g. --exclude_nodes calim10')
    parser.add_argument('--dynamic_queue_label', default=None,
                        help='Shared dynamic Redis queue label. Use this to append '
                            'multiple submissions to the same dynamic queue. '
                            'Default: current run label.')
    parser.add_argument('--dynamic_append_only', action='store_true',
                        help='Dynamic mode only: append work units to Redis queue '
                            'without seeding any new node tasks.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be submitted without actually submitting')
    args = parser.parse_args()

    if args.dynamic and args.remap:
        logger.error('--dynamic and --remap are mutually exclusive')
        sys.exit(1)

    # Resolve target/catalog paths to absolute so they work on remote workers.
    # Paths under orca/resources/ are resolved relative to the orca package
    # install location (which differs between submission host and workers).
    # Use Lustre paths or other shared-filesystem paths for portability.
    if args.targets:
        args.targets = [os.path.abspath(t) for t in args.targets]
        for t in args.targets:
            if not os.path.exists(t):
                print(f"WARNING: target file not found locally: {t}")
    if args.catalog:
        args.catalog = os.path.abspath(args.catalog)
        if not os.path.exists(args.catalog):
            print(f"WARNING: catalog file not found locally: {args.catalog}")

    # Parse node overrides: {subband: queue_name}
    remap = {}
    if args.remap:
        for entry in args.remap:
            if '=' not in entry:
                logger.error(f"Invalid --remap format: {entry} (expected SUBBAND=NODE)")
                sys.exit(1)
            sb, node = entry.split('=', 1)
            remap[sb] = node
            logger.info(f"Remap: {sb} → queue {node}")

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

    # =======================================================================
    #  DYNAMIC MODE
    # =======================================================================
    if args.dynamic:
        node_pool = args.nodes or list(DYNAMIC_NODE_POOL)
        if args.exclude_nodes:
            node_pool = [n for n in node_pool if n not in args.exclude_nodes]
        if not node_pool:
            logger.error('Node pool is empty after exclusions')
            sys.exit(1)

        dynamic_queue_label = args.dynamic_queue_label or run_label
        if dynamic_queue_label != run_label:
            logger.info(
                f"Dynamic queue label: {dynamic_queue_label} "
                f"(output run label: {run_label})"
            )

        # Build all work units
        all_work_units = []
        for subband in subbands:
            for seg in segments:
                obs_date = seg['start'].datetime.strftime('%Y-%m-%d')
                lst_label = seg['lst_label']
                start_dt = seg['start'].datetime
                end_dt = seg['end'].datetime
                ms_files = find_archive_files_for_subband(
                    start_dt, end_dt, subband, input_dir=args.input_dir,
                )
                if not ms_files:
                    logger.warning(
                        f"No files for {subband} in {lst_label} "
                        f"({start_dt} → {end_dt})"
                    )
                    continue

                all_work_units.append({
                    'ms_files': ms_files,
                    'subband': subband,
                    'bp_table': args.bp_table,
                    'xy_table': args.xy_table,
                    'lst_label': lst_label,
                    'obs_date': obs_date,
                    'run_label': run_label,
                    'peel_sky': args.peel_sky,
                    'peel_rfi': args.peel_rfi,
                    'hot_baselines': args.hot_baselines,
                    'skip_cleanup': args.skip_cleanup,
                    'cleanup_nvme': args.cleanup_nvme,
                    'targets': args.targets,
                    'catalog': args.catalog,
                    'clean_snapshots': args.clean_snapshots,
                    'clean_reduced_pixels': args.clean_reduced_pixels,
                    'reduced_pixels': args.reduced_pixels,
                    'skip_science': args.skip_science,
                    'compress_snapshots': args.compress_snapshots,
                    'snapshot_only': args.snapshot_only,
                })

        if not all_work_units:
            logger.error('No work units found')
            sys.exit(1)

        logger.info(
            f"=== Dynamic mode: {len(all_work_units)} work units "
            f"across {len(node_pool)} nodes ==="
        )
        for wu in all_work_units:
            logger.info(
                f"  {wu['subband']:>6s} | {wu['lst_label']} | "
                f"{len(wu['ms_files']):3d} files"
            )

        if args.dry_run:
            queued_before = _dynamic_queue_length(dynamic_queue_label)
            logger.info(
                f"[DRY RUN] Would push {len(all_work_units)} work units "
                f"to Redis queue {dynamic_queue_label} "
                f"({queued_before} currently queued)."
            )
            if args.dynamic_append_only:
                logger.info("[DRY RUN] Append-only mode: no node seeding")
            elif queued_before > 0:
                logger.info(
                    "[DRY RUN] Queue already has pending work; would skip seeding "
                    "and append only"
                )
            else:
                logger.info(
                    f"[DRY RUN] Would seed up to {len(node_pool)} nodes: "
                    f"{', '.join(node_pool)}"
                )
            sys.exit(0)

        queued_before = _dynamic_queue_length(dynamic_queue_label)

        # Push all work units to Redis
        _push_work_units(dynamic_queue_label, all_work_units)

        # Seed strategy:
        # - append-only mode: never seed
        # - if queue already had pending work: append only (avoid interference)
        # - otherwise: seed one per node to kick off processing
        seeded = 0
        if args.dynamic_append_only:
            logger.info("Dynamic append-only mode: skipped node seeding")
        elif queued_before > 0:
            logger.info(
                f"Dynamic queue {dynamic_queue_label} already had {queued_before} "
                "queued work units — appended only, no seeding"
            )
        else:
            for node_queue in node_pool:
                result = _pop_and_submit(dynamic_queue_label, node_queue)
                if result:
                    seeded += 1
                    logger.info(f"  Seeded {node_queue}")
                else:
                    break  # queue exhausted

        logger.info(
            f"=== Dynamic dispatch started: {seeded} nodes seeded, "
            f"{len(all_work_units) - seeded} queued ==="
        )
        logger.info(
            "Nodes will self-schedule from Redis as they finish.\n"
            "Monitor progress with:\n"
            "  celery -A orca.celery flower --port=5555"
        )
        sys.exit(0)

    # =======================================================================
    #  STATIC MODE  (original behavior)
    # =======================================================================
    results = []

    for subband in subbands:
        queue_override = remap.get(subband)
        queue = queue_override or get_queue_for_subband(subband)
        node = NODE_SUBBAND_MAP[subband]
        if queue_override:
            node = f"lwa{queue_override}"

        hour_specs = []
        for seg in segments:
            obs_date = seg['start'].datetime.strftime('%Y-%m-%d')
            lst_label = seg['lst_label']
            start_dt = seg['start'].datetime
            end_dt = seg['end'].datetime

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
                f"  {subband} | {lst_label} | {len(ms_files)} files "
                f"→ {node} (queue={queue})"
            )

            if args.dry_run:
                for ms in ms_files:
                    logger.info(f"    [DRY RUN] {ms}")

            hour_specs.append({
                'ms_files': ms_files,
                'lst_label': lst_label,
                'obs_date': obs_date,
            })

        if not hour_specs:
            continue

        if args.dry_run:
            labels = [h['lst_label'] for h in hour_specs]
            logger.info(
                f"[DRY RUN] {subband}: {len(hour_specs)} hours chained "
                f"sequentially → {' → '.join(labels)}"
            )
            continue

        result = submit_subband_pipeline_chained(
            hour_specs=hour_specs,
            subband=subband,
            bp_table=args.bp_table,
            xy_table=args.xy_table,
            run_label=run_label,
            peel_sky=args.peel_sky,
            peel_rfi=args.peel_rfi,
            hot_baselines=args.hot_baselines,
            skip_cleanup=args.skip_cleanup,
            cleanup_nvme=args.cleanup_nvme,
            queue_override=queue_override,
            targets=args.targets,
            catalog=args.catalog,
            clean_snapshots=args.clean_snapshots,
            clean_reduced_pixels=args.clean_reduced_pixels,
            reduced_pixels=args.reduced_pixels,
            skip_science=args.skip_science,
            compress_snapshots=args.compress_snapshots,
            snapshot_only=args.snapshot_only,
        )
        results.append({
            'subband': subband,
            'node': node,
            'n_hours': len(hour_specs),
            'total_files': sum(len(h['ms_files']) for h in hour_specs),
            'result': result,
        })

    # Summary
    logger.info(f"=== Submitted {len(results)} subband chains ===")
    for r in results:
        logger.info(
            f"  {r['subband']:>6s} | {r['n_hours']} hours chained | "
            f"{r['total_files']:3d} total files → {r['node']}"
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
