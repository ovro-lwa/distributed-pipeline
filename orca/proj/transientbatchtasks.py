"""An experimental imaging module that uses the scratch disk.

This might be retired later, but is useful in the meantime for benchmarking purposes. In general, tasks should be as
small as possible, but if using local scratch speeds things up, then it might be necessary to have bigger tasks.
"""
from .celery import app
from datetime import datetime
from typing import List, Optional, Tuple
from functools import partial
from itertools import chain

import os
import shutil
from glob import glob
import logging
import uuid
from billiard.pool import Pool

from orca.transform import imaging, gainscaling
from orca.wrapper import change_phase_centre
from orca.utils import image_sub
from orca.flagging.flagoperations import merge_group_flags

log = logging.getLogger(__name__)


@app.task
def make_image_products(ms_parent_list: List[str], ms_parent_day2_list: List[str],
                        ms_parent_after_end: str, ms_parent_after_end_day2: str,
                        snapshot_image_dir: str, snapshot_narrow_dir: str, snapshot_diff_outdir: str,
                        scratch_dir: str, narrow_chan_width: int = 30,
                        spw_list: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Make snapshot images for two chunks of data separated by 1 sidereal day that are sidereal subtraction ready.
    Also makes subsequent frame subtraction images.

    1. Copy day1 and day2 measurement sets to scratch
    2. chgcentre everything to a common phase center
    3. merge flags in everything
    4. gain corr from two sidereal days.
    6. Make each snapshot with source removed.
    7. Repeat but only for narrow-band

    Args:
        ms_parent_list: list of parent directories containing all the measurement sets (by spw) for single integrations
            for day 1 (not modified).
        ms_parent_day2_list: list of parent directories containing all the measurement sets (by spw) for single
            for single integrations for day 2 (not modified).
        ms_parent_after_end: parent directory containing all the measurement sets (by spw) for the single integration
            after the end of the ms_parent_list for day 1. For image subtraction purpose.
        ms_parent_after_end_day2: parent directory containing all the measurement sets (by spw) for the single integration
            after the end of the ms_parent_list for day 2.
        snapshot_image_dir: Output base directory for fullband snapshot image
        snapshot_narrow_dir: Output base directory for narrow-band snapshot image
        snapshot_diff_outdir: Output base directory for fullband subsequent diff images.
        scratch_dir: Scratch directory to put the copied ms and other intermediate data products.
        narrow_chan_width: frequency width in number of channels for the narrow-band image around 60 MHz.
        spw_list: list of spectral windows.

    Returns:

    """
    assert len(ms_parent_list) == len(ms_parent_day2_list)
    if not spw_list:
        spw_list = [f'{i:02d}' for i in range(22)]

    # TODO try-catch-finally, or tempfile
    temp = f'{scratch_dir}/{uuid.uuid4()}'
    os.makedirs(temp)
    small_pool = Pool(4)
    large_pool = Pool(11)
    try:
        log.info(f'Start copying files for a size {len(ms_parent_list)}.')
        copied_ms_parent_list = _parallel_copy(large_pool, ms_parent_list, temp)
        copied_ms_parent_day2_list = _parallel_copy(large_pool, ms_parent_day2_list, temp)
        copied_after_end = f'{temp}/{os.path.basename(ms_parent_after_end)}'
        copied_after_end_day2 = f'{temp}/{os.path.basename(ms_parent_after_end_day2)}'
        shutil.copytree(ms_parent_after_end, copied_after_end)
        shutil.copytree(ms_parent_after_end_day2, copied_after_end_day2)

        # Just to be safe
        ms_parent_list, ms_parent_day2_list, ms_parent_after_end, ms_parent_after_end_day2 = [], [], '', ''

        log.info('Merging flags.')
        _parallel_merge_flags(large_pool, copied_ms_parent_list + copied_ms_parent_day2_list, spw_list)
        middle_ms = glob(f'{copied_ms_parent_list[len(copied_ms_parent_list) // 2]}/??_*.ms')[0]
        phase_center = change_phase_centre.get_phase_center(middle_ms)

        log.info(f'Change chunk phase center to {phase_center}.')
        _parallel_chgcentre(large_pool, copied_ms_parent_list + copied_ms_parent_day2_list, phase_center)

        # gain correction
        log.info('Start pair-wise gain correction.')
        _parallel_gain_correction(large_pool, copied_ms_parent_day2_list, ms_parent_list, spw_list)

        snapshots1, timestamps1 = _parallel_wsclean_snapshot_sources_removed(small_pool,
                                                                             copied_ms_parent_list + [copied_after_end],
                                                                             temp, snapshot_image_dir, spw_list)
        snapshots2, timestamps2 = _parallel_wsclean_snapshot_sources_removed(small_pool,
                                                                             copied_ms_parent_day2_list +
                                                                             [copied_after_end_day2],
                                                                             temp, snapshot_image_dir, spw_list)

        for i in range(len(snapshots1[:-1])):
            outdir1 = f'{snapshot_diff_outdir}/{timestamps1[-1].date()}/hh={timestamps1[-1].hour:02d}'
            outdir2 = f'{snapshot_diff_outdir}/{timestamps2[-1].date()}/hh={timestamps2[-1].hour:02d}'
            os.makedirs(outdir1, exist_ok=True)
            os.makedirs(outdir2, exist_ok=True)
            image_sub.image_sub(snapshots1[i], snapshots1[i], outdir1)
            image_sub.image_sub(snapshots2[i], snapshots2[i], outdir2)
        os.remove(snapshots1[-1])
        os.remove(snapshots2[-1])
        snapshots1 = snapshots1[:-1]
        snapshots2 = snapshots2[:-1]

        # Narrow band imaging
        tmp1 = f'{temp}/tmp1'
        os.makedirs(tmp1)
        start_chan = str(51 - narrow_chan_width // 2)
        end_chan = str(51 + narrow_chan_width // 2 + 1)
        narrow_snapshots1 = _parallel_wsclean_snapshot_sources_removed(small_pool,
                                                                       copied_ms_parent_list, tmp1,
                                                                       snapshot_narrow_dir, ['06'],
                                                                       more_args=['-channelrange', start_chan, end_chan])
        narrow_snapshots2 = _parallel_wsclean_snapshot_sources_removed(small_pool,
                                                                       copied_ms_parent_day2_list, tmp1,
                                                                       snapshot_narrow_dir, ['06'],
                                                                       more_args=['-channelrange', start_chan, end_chan])
    finally:
        small_pool.close()
        large_pool.close()
        small_pool.join()
        large_pool.join()
        shutil.rmtree(temp)

    return snapshots1, snapshots2, narrow_snapshots1, narrow_snapshots2


def _parallel_chgcentre(pool: Pool, ms_parent_list: List[str], phase_center: str):
    """
    Change phase center for all the measurement sets under ms_parent in parallel

    Args:
        ms_parent:
        phase_center:

    Returns:

    """
    ms_list = chain.from_iterable((glob(f'{ms_parent}/??_*ms') for ms_parent in ms_parent_list))
    pool.starmap(change_phase_centre.change_phase_center, ((ms, phase_center) for ms in ms_list))


def _parallel_copy(pool, directories: List[str], dest_directory: str) -> List[str]:
    """
    Copy each element of directories to under dest_directory with n_thread parallel cp at a time.

    Args:
        directories:
        dest_directory:
        n_procs:

    Returns:

    """
    dests = [f'{dest_directory}/{os.path.basename(d)}' for d in directories]
    pool.starmap(shutil.copytree, ((src, dest) for src, dest in zip(directories, dests)))
    return dests


def _parallel_gain_correction(pool, baseline_parents, target_parents, spw_list):
    pool.starmap(gainscaling.correct_scaling,
                 ((f'{baseline_parent}/{spw}_{os.path.basename(baseline_parent)}.ms',
                   f'{target_parent}/{spw}_{os.path.basename(target_parent)}.ms') for spw in spw_list
                  for baseline_parent, target_parent in zip(baseline_parents, target_parents)))

"""
def _parallel_wsclean_subsequent_diff(ms_parent_list, temp, snap_outdir, diff_outdir, spw_list, n_procs=4) -> None:
    pool = Pool(n_procs)
    go = partial(imaging.make_dirty_image, inner_tukey=20)
    # need to index snap_outdir
    args = [([f'{ms_parent_path}/{_fn(ms_parent_path, spw)}' for spw in spw_list], snap_outdir,
             os.path.basename(ms_parent_path))
            for ms_parent_path in ms_parent_list]
    images = pool.starmap(go, args)

    for i, fits in enumerate(images[:-1]):
        timestamp = datetime.strptime(os.path.basename(ms_parent_list[i]), "%Y-%m-%dT%H:%M:%S")
        indexd_out_dir = f'{out_dir}/{timestamp.date()}/hh={timestamp.hour:02d}'
        image_sub.image_sub(images[i], images[i+1], indexd_out_dir)
"""


def _parallel_wsclean_snapshot_sources_removed(pool, ms_parent_list, temp, out_dir, spw_list,
                                               more_args=None) -> Tuple[List[str], List[datetime]]:
    mki = partial(imaging.make_residual_image_with_source_removed, inner_tukey=20, more_args=more_args)

    args_list = []
    timestamps = []
    for ms_parent_path in ms_parent_list:
        timestamp_str = os.path.basename(ms_parent_path)
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        timestamps.append(timestamp)
        indexed_outdir = f'{out_dir}/{timestamp.date()}/hh={timestamp.hour:02d}'
        os.makedirs(indexed_outdir, exist_ok=True)
        args_list.append(
            ([f'{ms_parent_path}/{_fn(ms_parent_path, spw)}' for spw in spw_list],
             timestamp, indexed_outdir, timestamp_str, temp)
        )
    images = pool.starmap(mki, args_list)

    return images, timestamps


def _parallel_merge_flags(pool, ms_parent_list, spw_list):
    pool.map(merge_group_flags, [[f'{ms_parent}/{_fn(ms_parent, spw)}' for ms_parent in ms_parent_list]
                                 for spw in spw_list])


def _fn(ms_parent_path: str, spw: str) -> str:
    return f'{spw}_{os.path.basename(ms_parent_path)}.ms'
