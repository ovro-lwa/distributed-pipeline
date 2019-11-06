from orca.proj.boilerplate import run_dada2ms, run_chgcentre, peel, apply_a_priori_flags, flag_chans
from .celery import app
from celery import group
from ..transform import dada2ms, change_phase_centre, imaging, sidereal_subtraction_kit
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

pm = OfflinePathsManager(utc_times_txt_path='/home/yuping/utc_times.txt',
                         dadafile_dir='/lustre/data/2018-03-20_100hr_run',
                         msfile_dir='/lustre/yuping/0-100-hr-reduction/blflag/msfile',
                         bcal_dir='/lustre/yuping/2019-10-100-hr-take-two/2018-03-22T17:34:35')


def dispatch_dada2ms(start_time, end_time):
    spws = [f'{i:02d}' for i in range(22)]
    params = [{'dada_file': pm.get_dada_path(s, time), 'out_ms': pm.get_ms_path(time, s), 'gaintable': pm.get_gaintable_path(s)}
              for time in pm.utc_times_mapping if start_time <= time < end_time
              for s in spws]
    logging.info(f'Making {len(params)} dada2ms calls.')
    group(run_dada2ms.s(**p) for p in params)()


"""
A bunch of light wrapper that converts existing functions into celery tasks
"""


@app.task
def make_first_image(prefix, datetime_string, out_dir):
    logging.info(f'Glob statement is {prefix}/{datetime_string}/??_{datetime_string}.ms')
    ms_list = sorted(glob.glob(f'{prefix}/{datetime_string}/??_{datetime_string}.ms'))
    assert len(ms_list) == 22
    imaging.make_image(ms_list, datetime_string, out_dir)
    pass

@app.task
def subsequent_frame_subtraction(dir1, dir2, datetime_1, datetime_2, out_dir):
    # Copy over to local disk, chgcentre, merge flag, and then subtract, write back.
    temp_tree = f'/pipedata/workdir/yuping/{uuid.uuid4()}'
    # I'd rather have an error thrown here if there's a UUID clash which shouldn't ever happen.
    os.mkdir(temp_tree)
    try:
        tree1 = f'{temp_tree}/{datetime_1}'
        tree2 = f'{temp_tree}/{datetime_2}'
        shutil.copytree(dir1, tree1)
        shutil.copytree(dir2, tree2)
        new_phase_center = change_phase_centre.get_phase_center(f'{tree1}/00_{datetime_1}.ms')
        spws = [f'{i:02d}' for i in range(22)]
        for s in spws:
            merge_flags.merge_flags(f'{tree1}/{s}_{datetime_1}.ms', f'{tree2}/{s}_{datetime_2}.ms')
            change_phase_centre.change_phase_center(f'{tree2}/{s}_{datetime_2}.ms', new_phase_center)
        im1 = imaging.make_image(sorted(glob.glob(f'{tree2}/??_{datetime_2}.ms')), datetime_2, temp_tree)
        im2 = imaging.make_image(sorted(glob.glob(f'{tree1}/??_{datetime_1}.ms')), datetime_1, temp_tree)
        image_sub.image_sub(im1, im2, out_dir)
    finally:
        shutil.rmtree(temp_tree)


# OH MAN I NEED TO REFACTOR OUT THIS ABOMNIATION OF DUPLICATE CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.task
def prep_image_for_sidereal_subtraction(dir1, dir2, datetime_1, datetime_2, out_dir1, out_dir2):
    # Copy over to local disk, chgcentre, merge flag, and then subtract, write back.
    temp_tree = f'/pipedata/workdir/yuping/{uuid.uuid4()}'
    # I'd rather have an error thrown here if there's a UUID clash which shouldn't ever happen.
    os.mkdir(temp_tree)
    try:
        tree1 = f'{temp_tree}/{datetime_1}'
        tree2 = f'{temp_tree}/{datetime_2}'
        shutil.copytree(f'{dir1}/{datetime_1}', tree1)
        shutil.copytree(f'{dir2}/{datetime_2}', tree2)
        new_phase_center = change_phase_centre.get_phase_center(f'{tree1}/00_{datetime_1}.ms')
        spws = [f'{i:02d}' for i in range(22)]
        for s in spws:
            merge_flags.merge_flags(f'{tree1}/{s}_{datetime_1}.ms', f'{tree2}/{s}_{datetime_2}.ms')
            change_phase_centre.change_phase_center(f'{tree2}/{s}_{datetime_2}.ms', new_phase_center)
        im1, psf = imaging.make_image(sorted(glob.glob(f'{tree1}/??_{datetime_1}.ms')), datetime_1,
                                      temp_tree, make_psf=True)
        im2 = imaging.make_image(sorted(glob.glob(f'{tree2}/??_{datetime_2}.ms')), datetime_2, temp_tree)
        shutil.copy(im1, out_dir1)
        shutil.copy(psf, out_dir1)
        shutil.copy(im2, out_dir2)
    finally:
        shutil.rmtree(temp_tree)


@app.task
def sidereal_subtract_image(im1_path, im2_path, out_dir):
    sidereal_subtraction_kit.subtract_images(im1_path, im2_path, out_dir)


@app.task
def sidereal_subtract_image2(im1_path, im2_path, psf_path, out_dir):
    # subtract the crab and do a flux scale.
    sidereal_subtraction_kit.subtract_images(im1_path, im2_path, out_dir, psf_path, subtract_crab=True, scale=True)


"""
Combinations of stuff to run to do actually dispatch the tasks via celery canvas.
"""
def get_data():
        s = datetime(2018, 3, 22, 1, 0, 0)
        e = datetime(2018, 3, 22, 6, 0, 0)
        dispatch_dada2ms(s, e)


def do_flag():
    ms_list = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02/*/??_*ms'))
    logging.info(f'Making {len(ms_list)} apply_flag calls.')
    group(apply_a_priori_flags.s(ms, '/home/yuping/100-hr-a-priori-flags/201906_consolidated.npy', True)
          for ms in ms_list)()


def do_peel():
    ms_list = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02/*/??_*ms'))
    logging.info(f'Making {len(ms_list)} peeling calls.')
    group(peel.s(ms, '/home/yuping/casA_resolved_rfi.json') for ms in ms_list)()


def do_flag_chans():
    spws = [f'{i:02d}' for i in range(22)]
    for s in spws:
        # TODO generate this without having to stat. This doesn't scale well on lustre
        ms_list = sorted(glob.glob(f'/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02/*/{s}_*.ms'))
        logging.info(f'Making {len(ms_list)} flag_chans calls.')
        group(flag_chans.s(ms, s) for ms in ms_list)()


def do_first_batch_image():
    prefix = '/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02'
    ms_list = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02/*'))
    group(make_first_image.s(prefix, os.path.basename(ms),
                             '/lustre/yuping/0-100-hr-reduction/qual/images/2018-03-22/hh=02') for ms in ms_list)()


def do_subsequent_frame_subtraction():
    ms_list = sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-23/hh=02/*'))
    group(subsequent_frame_subtraction.s(ms_list[i], ms_list[i+1],
                                         os.path.basename(ms_list[i]), os.path.basename(ms_list[i+1]),
                                 '/lustre/yuping/0-100-hr-reduction/qual/subsequent-subtraction/2018-03-23/hh=02') for
          i, dir1 in enumerate(ms_list[:-1]))()


def generate_datetime_pairs() -> List[Tuple[datetime, datetime]]:
    sday = timedelta(days=0, hours=23, minutes=56, seconds=4)
    day_1_times = [ datetime.fromisoformat(os.path.basename(p)) for p in
                    sorted(glob.glob('/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02/*'))]
    return [(one, one + sday) for one in day_1_times if one > datetime(2018, 3, 22, 2, 3, 56)]


def prep_sidereal_subtraction():
    """TODO for sidereal subtraction involving more than one day. This directory structure won't really work since
    the flags will be different.
    """
    ms_pairs = generate_datetime_pairs()
    logging.info(f'There are {len(ms_pairs)} pairs of sidereally separated measurement sets to process.')
    group(prep_image_for_sidereal_subtraction.s(
        '/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-22/hh=02',
        '/lustre/yuping/0-100-hr-reduction/qual/msfiles/2018-03-23/hh=02',
        p[0].isoformat(), p[1].isoformat(),
        '/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-22/hh=02',
        '/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-23/hh=02')
          for p in ms_pairs)()


def do_sidereal_subtraction():
    # call the kit
    # do the subtraction
    out_dir = '/lustre/yuping/0-100-hr-reduction/qual/sidereal-subtraction-1/2018-03-22/hh=02'
    pairs = generate_datetime_pairs()
    logging.info(f'There are {len(pairs)} pairs of sidereally separated images to process.')
    group(sidereal_subtract_image.s(
        f'/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-22/hh=02/{p[0].isoformat()}-image.fits',
        f'/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-23/hh=02/{p[1].isoformat()}-image.fits',
        out_dir)
          for p in pairs)()


def do_sidereal_subtraction2():
    # call the kit
    # do the subtraction
    out_dir = '/lustre/yuping/0-100-hr-reduction/qual/sidereal-subtraction-2/2018-03-22/hh=02'
    pairs = generate_datetime_pairs()
    logging.info(f'There are {len(pairs)} pairs of sidereally separated images to process.')
    group(sidereal_subtract_image2.s(
        f'/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-22/hh=02/{p[0].isoformat()}-image.fits',
        f'/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-23/hh=02/{p[1].isoformat()}-image.fits',
        f'/lustre/yuping/0-100-hr-reduction/qual/prep-sidereal-images/2018-03-22/hh=02/{p[0].isoformat()}-psf.fits',
        out_dir)
          for p in pairs)()
