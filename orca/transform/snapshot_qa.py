"""Persistence helpers for snapshot image quality measurements."""

import csv
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Sequence


logger = logging.getLogger(__name__)

CSV_FIELDS = [
    'snapshot_index',
    'scan_number',
    'timestamp_utc',
    'rms_jy_per_beam',
    'peak_jy_per_beam',
    'flagged',
    'filename',
]


def _timestamp_from_filename(filename: str) -> str:
    match = re.search(r'(\d{8}_\d{6})', filename)
    if not match:
        return ''
    try:
        timestamp = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    except ValueError:
        return ''
    return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')


def write_snapshot_qa_csv(
    stats: List[Dict],
    bad_indices: List[int],
    scan_numbers: Sequence[int],
    work_dir: str,
    subband: str,
) -> str:
    """Write the machine-readable data behind the snapshot RMS plot."""
    qa_dir = os.path.join(work_dir, 'QA')
    os.makedirs(qa_dir, exist_ok=True)
    output_path = os.path.join(qa_dir, f'snapshot_qa_{subband}.csv')
    flagged_indices = set(bad_indices)
    unmapped_count = 0

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for stat in sorted(stats, key=lambda item: item['idx']):
            idx = int(stat['idx'])
            scan_number = (
                scan_numbers[idx] if 0 <= idx < len(scan_numbers) else ''
            )
            if scan_number == '':
                unmapped_count += 1
            writer.writerow({
                'snapshot_index': idx,
                'scan_number': scan_number,
                'timestamp_utc': _timestamp_from_filename(stat['file']),
                'rms_jy_per_beam': stat['rms'],
                'peak_jy_per_beam': stat['peak'],
                'flagged': 'true' if idx in flagged_indices else 'false',
                'filename': stat['file'],
            })

    if unmapped_count:
        logger.warning(
            'No scan number available for %d snapshot QA rows', unmapped_count,
        )

    return output_path
