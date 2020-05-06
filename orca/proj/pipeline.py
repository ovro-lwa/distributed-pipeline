from orca.transform import imaging
from orca.proj.boilerplate import run_dada2ms, peel, apply_a_priori_flags, flag_chans, make_first_image
from .celery import app
from celery import group
from ..transform import siderealsubtraction, gainscaling
from ..wrapper import change_phase_centre, wsclean
from ..flagging import merge_flags
from ..metadata.pathsmanagers import OfflinePathsManager
from ..utils import image_sub
from datetime import datetime, timedelta
import os
import sys
import logging
import glob
import shutil
from typing import List, Tuple
import uuid

pm = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                         dadafile_dir='/lustre/data/2018-03-20_100hr_run',
                         msfile_dir='/lustre/yuping/0-100-hr-reduction/salf/msfiles',
                         bcal_dir='/lustre/yuping/2019-10-100-hr-take-two/bandpass/2018-03-22',
                         flag_npy_path='/home/yuping/100-hr-a-priori-flags/20191125-consolidated-flags/20191125-consolidated-flags.npy')


@app.task
def prep_image_for_sidereal_subtraction(dir1, dir2, datetime_1, datetime_2, out_dir1, out_dir2):
    # Copy over to local disk, chgcentre, merge flag, and then subtract, write back.
    temp_tree = f'/pipedata/workdir/yuping/{uuid.uuid4()}'
    # I'd rather have an error thrown here if there's a UUID clash which shouldn't ever happen.
    os.makedirs(temp_tree, exist_ok=True)
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    try:
        tree1 = f'{temp_tree}/{datetime_1}'
        tree2 = f'{temp_tree}/{datetime_2}'
        shutil.copytree(f'{dir1}/{datetime_1}', tree1)
        shutil.copytree(f'{dir2}/{datetime_2}', tree2)
        new_phase_center = change_phase_centre.get_phase_center(f'{tree1}/00_{datetime_1}.ms')
        spws = [f'{i:02d}' for i in range(22)]
        for s in spws:
            gainscaling.correct_scaling(f'{tree1}/{s}_{datetime_1}.ms', f'{tree2}/{s}_{datetime_2}.ms')
            merge_flags.merge_flags(f'{tree1}/{s}_{datetime_1}.ms', f'{tree2}/{s}_{datetime_2}.ms')
            change_phase_centre.change_phase_center(f'{tree2}/{s}_{datetime_2}.ms', new_phase_center)

        im1 = imaging.make_residual_image_with_source_removed(sorted(glob.glob(f'{tree1}/??_{datetime_1}.ms')),
                                                              temp_tree,
                                                              datetime_1, imaging.SUN, temp_tree, inner_tukey='20')
        im2 = imaging.make_residual_image_with_source_removed(sorted(glob.glob(f'{tree2}/??_{datetime_2}.ms')),
                                                              temp_tree,
                                                              datetime_2, imaging.SUN, temp_tree, inner_tukey='20')
        shutil.copy(im1, out_dir1)
        shutil.copy(im2, out_dir2)
    finally:
        shutil.rmtree(temp_tree)


def small_imaging_test():
    out_dir = '/lustre/yuping/0-100-hr-reduction/qual/sidereal-subtraction-3/2018-03-22/hh=15'
    sday = timedelta(days=0, hours=23, minutes=56, seconds=4)
    pairs = [(datetime(2018, 3, 22, 15, 36, 4), datetime(2018, 3, 22, 15, 36, 4) + sday)]
    logging.info(f'There are {len(pairs)} pairs of sidereally separated images to process.')
    group(prep_image_for_sidereal_subtraction.s(
        f'/lustre/yuping/0-100-hr-reduction/salf/msfiles/2018-03-22/hh=15',
        f'/lustre/yuping/0-100-hr-reduction/salf/msfiles/2018-03-23/hh=15',
        p[0].isoformat(),
        p[1].isoformat(),
        out_dir1='/lustre/yuping/0-100-hr-reduction/salf/prep-sidereal-images3/2018-03-22/hh=15',
        out_dir2='/lustre/yuping/0-100-hr-reduction/salf/prep-sidereal-images3/2018-03-23/hh=15')
          for p in pairs)()