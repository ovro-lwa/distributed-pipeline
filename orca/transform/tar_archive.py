"""Tar / untar averaged MS directories to save inode count on Lustre.

Library functions importable by the pipeline, plus a CLI entry-point::

    # Tar .ms → .ms.tar
    python -m orca.transform.tar_archive tar --date 2024-12-27
    python -m orca.transform.tar_archive tar --date 2024-12-27 --subband 73MHz --hour 05
    python -m orca.transform.tar_archive tar --older-than 7
    python -m orca.transform.tar_archive tar --date 2024-12-27 --dry-run

    # Untar .ms.tar → .ms
    python -m orca.transform.tar_archive untar --date 2024-12-27
    python -m orca.transform.tar_archive untar --date 2024-12-27 --subband 73MHz --hour 05
    python -m orca.transform.tar_archive untar --date 2024-12-27 --dry-run

The pipeline (``find_archive_files_for_subband`` + ``copy_ms_to_nvme``)
handles both ``.ms`` and ``.ms.tar`` transparently, so you can tar old data
at any time without breaking anything.
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

BASE_DIR = '/lustre/pipeline/night-time/averaged'


# ---------------------------------------------------------------------------
#  Core functions (importable)
# ---------------------------------------------------------------------------
def tar_ms_directory(ms_path: str, dry_run: bool = False) -> bool:
    """Create ``<name>.ms.tar`` and remove the original ``.ms`` directory.

    Uses GNU tar (no compression — MS data doesn't compress well).

    Returns True on success or skip, False on failure.
    """
    tar_path = ms_path + '.tar'
    parent = os.path.dirname(ms_path)
    ms_name = os.path.basename(ms_path)

    if os.path.exists(tar_path):
        logger.info(f"SKIP (tar exists): {tar_path}")
        return True

    if dry_run:
        logger.info(f"[DRY RUN] would tar: {ms_path}")
        return True

    try:
        subprocess.run(
            ['tar', 'cf', tar_path, ms_name],
            cwd=parent,
            check=True,
            capture_output=True,
        )
        if os.path.getsize(tar_path) == 0:
            logger.error(f"Tar file is empty: {tar_path}")
            os.remove(tar_path)
            return False

        shutil.rmtree(ms_path)
        logger.info(f"Tarred {ms_path} → {tar_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"tar failed for {ms_path}: {e.stderr.decode()}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return False


def untar_ms_archive(tar_path: str, dry_run: bool = False) -> bool:
    """Extract ``<name>.ms.tar`` back to a ``.ms`` directory and remove the tar.

    Returns True on success or skip, False on failure.
    """
    ms_path = tar_path.removesuffix('.tar')
    parent = os.path.dirname(tar_path)

    if os.path.isdir(ms_path):
        logger.info(f"SKIP (ms dir exists): {ms_path}")
        return True

    if dry_run:
        logger.info(f"[DRY RUN] would untar: {tar_path}")
        return True

    try:
        with tarfile.open(tar_path, 'r') as tf:
            tf.extractall(path=parent)
        if not os.path.isdir(ms_path):
            logger.error(f"Expected {ms_path} after extraction but not found")
            return False

        os.remove(tar_path)
        logger.info(f"Untarred {tar_path} → {ms_path}")
        return True
    except Exception as e:
        logger.error(f"untar failed for {tar_path}: {e}")
        return False


def find_ms_entries(
    date: Optional[str] = None,
    subband: Optional[str] = None,
    hour: Optional[str] = None,
    older_than: Optional[int] = None,
    ext: str = '.ms',
    base_dir: str = BASE_DIR,
) -> Iterator[str]:
    """Yield .ms or .ms.tar paths matching the given filters.

    Args:
        date: Date string ``YYYY-MM-DD``.
        subband: Subband label (e.g. ``'73MHz'``).
        hour: Hour string (e.g. ``'05'``).
        older_than: Only include dates older than this many days.
        ext: ``'.ms'`` to find dirs (for tarring) or ``'.ms.tar'`` (for untarring).
        base_dir: Root of the averaged directory tree.
    """
    cutoff = None
    if older_than is not None:
        cutoff = datetime.utcnow() - timedelta(days=older_than)

    if subband:
        subband_dirs = [os.path.join(base_dir, subband)]
    else:
        try:
            subband_dirs = sorted(
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            )
        except FileNotFoundError:
            return

    for sb_dir in subband_dirs:
        if not os.path.isdir(sb_dir):
            continue

        if date:
            date_dirs = [os.path.join(sb_dir, date)]
        else:
            date_dirs = sorted(
                os.path.join(sb_dir, d)
                for d in os.listdir(sb_dir)
                if os.path.isdir(os.path.join(sb_dir, d))
            )

        for d_dir in date_dirs:
            if not os.path.isdir(d_dir):
                continue
            if cutoff:
                try:
                    dir_date = datetime.strptime(os.path.basename(d_dir), '%Y-%m-%d')
                    if dir_date >= cutoff:
                        continue
                except ValueError:
                    continue

            if hour:
                hour_dirs = [os.path.join(d_dir, hour)]
            else:
                hour_dirs = sorted(
                    os.path.join(d_dir, h)
                    for h in os.listdir(d_dir)
                    if os.path.isdir(os.path.join(d_dir, h))
                )

            for h_dir in hour_dirs:
                if not os.path.isdir(h_dir):
                    continue
                for entry in sorted(os.listdir(h_dir)):
                    full = os.path.join(h_dir, entry)
                    if ext == '.ms' and entry.endswith('.ms') and not entry.endswith('.ms.tar') and os.path.isdir(full):
                        yield full
                    elif ext == '.ms.tar' and entry.endswith('.ms.tar') and os.path.isfile(full):
                        yield full


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _cli():
    top = argparse.ArgumentParser(
        description='Tar/untar averaged MS directories on Lustre.',
    )
    sub = top.add_subparsers(dest='action', required=True)

    # -- tar subcommand --
    p_tar = sub.add_parser('tar', help='Tar .ms directories into .ms.tar')
    p_tar.add_argument('--date', help='Date (YYYY-MM-DD)')
    p_tar.add_argument('--subband', help='Subband (e.g. 73MHz)')
    p_tar.add_argument('--hour', help='Hour (e.g. 05)')
    p_tar.add_argument('--older-than', type=int, metavar='DAYS')
    p_tar.add_argument('--dry-run', action='store_true')
    p_tar.add_argument('--base-dir', default=BASE_DIR)

    # -- untar subcommand --
    p_untar = sub.add_parser('untar', help='Untar .ms.tar back to .ms directories')
    p_untar.add_argument('--date', help='Date (YYYY-MM-DD)')
    p_untar.add_argument('--subband', help='Subband (e.g. 73MHz)')
    p_untar.add_argument('--hour', help='Hour (e.g. 05)')
    p_untar.add_argument('--dry-run', action='store_true')
    p_untar.add_argument('--base-dir', default=BASE_DIR)

    args = top.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.action == 'tar':
        if not args.date and args.older_than is None:
            top.error('tar requires --date or --older-than')
        entries = list(find_ms_entries(
            date=args.date, subband=args.subband, hour=args.hour,
            older_than=args.older_than, ext='.ms', base_dir=args.base_dir,
        ))
        print(f"Found {len(entries)} .ms directories")
        ok = sum(1 for p in entries if tar_ms_directory(p, dry_run=args.dry_run))
        fail = len(entries) - ok
    else:
        if not args.date:
            top.error('untar requires --date')
        entries = list(find_ms_entries(
            date=args.date, subband=args.subband, hour=args.hour,
            ext='.ms.tar', base_dir=args.base_dir,
        ))
        print(f"Found {len(entries)} .ms.tar archives")
        ok = sum(1 for p in entries if untar_ms_archive(p, dry_run=args.dry_run))
        fail = len(entries) - ok

    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    _cli()
