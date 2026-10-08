#!/usr/bin/env python3
"""Flag + frequency-average slow-buffer data for a fixed LST window on one node.

For each date, the LST window (e.g. 03-06) is converted to UTC exactly as
``subband_celery.py --range`` does, so the averaged data covers whole LST
hours for the main pipeline. Matching ``.ms.tar`` / ``.ms`` files under
``/lustre/pipeline/slow/<subband>/<date>/<hour>/`` are submitted as
``run_pipeline_slow_on_one_cpu_nvme`` tasks to a single calimNN queue, and
the averaged MS land in ``<output_base>/<subband>/<date>/<hour>/`` (default:
the night-time tree that ``subband_celery.py`` reads).

Safe to re-run: files whose averaged MS already exists are skipped, and files
already submitted (recorded in --ledger) are skipped unless --resubmit.

Example::

    python pipeline/average_lst_window.py \\
        --lst 03-06 --start_date 2026-10-07 --end_date 2026-10-12 \\
        --subbands 41MHz 55MHz --node calim00 --dry_run
"""
import argparse
import datetime
import glob
import os
import re
import sys

from celery import group

from orca.tasks.pipeline_tasks import run_pipeline_slow_on_one_cpu_nvme
from subband_celery import parse_time_range

ROOT_SLOW = '/lustre/pipeline/slow'


def select_files(subband, t_start, t_end, margin_s):
    """Return sorted raw files of *subband* whose start time is inside the window.

    Uses the same test as find_archive_files_for_subband (file start + 5 s),
    widened by *margin_s* on both sides so boundary integrations are kept.
    """
    lo = t_start.to_datetime() - datetime.timedelta(seconds=margin_s)
    hi = t_end.to_datetime() + datetime.timedelta(seconds=margin_s)
    pattern = re.compile(r'(\d{8}_\d{6})_' + re.escape(subband) + r'\.ms(?:\.tar)?$')

    files = {}
    hour = lo.replace(minute=0, second=0, microsecond=0)
    while hour <= hi:
        hour_dir = os.path.join(ROOT_SLOW, subband, hour.strftime('%Y-%m-%d'), hour.strftime('%H'))
        # Reverse sort so .ms.tar wins over .ms for the same timestamp
        for path in sorted(glob.glob(os.path.join(hour_dir, '*.ms*')), reverse=True):
            m = pattern.search(os.path.basename(path))
            if not m:
                continue
            t_file = datetime.datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
            if lo <= t_file + datetime.timedelta(seconds=5) < hi:
                files.setdefault(m.group(1), path)
        hour += datetime.timedelta(hours=1)
    return [files[k] for k in sorted(files)]


def averaged_path(vis, output_base):
    """Averaged MS path the task writes for *vis* (mirrors run_pipeline_slow_on_one_cpu_nvme)."""
    rel_dir = os.path.dirname(vis.split('/slow/', 1)[1])
    ms_name = os.path.basename(vis)
    if ms_name.endswith('.tar'):
        ms_name = ms_name[:-4]
    return os.path.join(output_base, rel_dir, f"{os.path.splitext(ms_name)[0]}_averaged.ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lst', required=True, help="LST hour range, e.g. '03-06'")
    parser.add_argument('--start_date', required=True, help='First date YYYY-MM-DD')
    parser.add_argument('--end_date', help='Last date YYYY-MM-DD (inclusive, default: start_date)')
    parser.add_argument('--subbands', nargs='+', required=True)
    parser.add_argument('--node', required=True, help='calimNN queue to run on, e.g. calim00')
    parser.add_argument('--output_base', default='/lustre/pipeline/night-time/averaged/')
    parser.add_argument('--chanbin', type=int, default=4)
    parser.add_argument('--margin_s', type=float, default=60.0,
                        help='Extra seconds kept on each side of the LST window')
    parser.add_argument('--ledger', default=os.path.expanduser('~/average_lst_window_submitted.txt'),
                        help='File listing already-submitted inputs')
    parser.add_argument('--resubmit', action='store_true',
                        help='Ignore the ledger (still skips finished outputs)')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    if not re.fullmatch(r'calim\d\d', args.node):
        sys.exit(f"--node must look like calim00, got {args.node!r}")

    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date or args.start_date)

    submitted = set()
    if os.path.exists(args.ledger) and not args.resubmit:
        with open(args.ledger) as f:
            submitted = {line.strip() for line in f if line.strip()}

    now = datetime.datetime.utcnow()
    to_submit = []
    date = start
    while date <= end:
        t_start, t_end = parse_time_range(args.lst, date.isoformat())
        note = '  (window not finished yet)' if t_end.to_datetime() > now else ''
        print(f"{date}  LST {args.lst}  UTC {t_start.isot[:19]} -> {t_end.isot[:19]}{note}")
        for sb in args.subbands:
            files = select_files(sb, t_start, t_end, args.margin_s)
            done = [f for f in files if os.path.isdir(averaged_path(f, args.output_base))]
            pending = [f for f in files if f not in done and f not in submitted]
            print(f"    {sb}: {len(files)} raw, {len(done)} averaged, "
                  f"{len(files) - len(done) - len(pending)} in ledger, {len(pending)} to submit")
            to_submit += pending
        date += datetime.timedelta(days=1)

    print(f"Total to submit: {len(to_submit)} -> queue {args.node}")
    if args.dry_run or not to_submit:
        return

    group(
        run_pipeline_slow_on_one_cpu_nvme.s(
            f, start=0, end=23, chanbin=args.chanbin, output_base=args.output_base,
        ).set(queue=args.node)
        for f in to_submit
    ).apply_async()
    with open(args.ledger, 'a') as f:
        f.writelines(p + '\n' for p in to_submit)
    print(f"Submitted; recorded in {args.ledger}")


if __name__ == '__main__':
    main()
