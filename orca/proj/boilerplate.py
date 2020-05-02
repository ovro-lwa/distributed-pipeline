import glob
import os
import logging

import orca.transform.imaging
from orca.flagging import merge_flags, flag_bad_chans
from orca.proj.celery import app
from orca.wrapper import dada2ms, change_phase_centre, wsclean
from orca.transform import peeling


"""
Celery adapter on top of transforms.
"""
@app.task
def run_dada2ms(dada_file, out_ms, gaintable=None):
    dada2ms.dada2ms(dada_file, out_ms, gaintable)


@app.task
def run_chgcentre(ms_file, direction):
    change_phase_centre.change_phase_center(ms_file, direction)


@app.task
def peel(ms_file, sources):
    peeling.peel_with_ttcal(ms_file, sources)


@app.task
def apply_a_priori_flags(ms_file, flag_npy_path):
    merge_flags.write_to_flag_column(ms_file, flag_npy_path)


@app.task
def apply_ant_flag(ms_file, ants):
    from casacore.tables import table, taql
    t = table(ms_file)
    taql(f"update $t set FLAG=True where any(ANTENNA1==$ants || ANTENNA2==$ants)")


@app.task
def flag_chans(ms, spw):
    flag_bad_chans.flag_bad_chans(ms, spw, apply_flag=True)
    return ms


@app.task
def make_first_image(prefix, datetime_string, out_dir):
    logging.info(f'Glob statement is {prefix}/{datetime_string}/??_{datetime_string}.ms')
    os.makedirs(out_dir,exist_ok=True)
    ms_list = sorted(glob.glob(f'{prefix}/{datetime_string}/??_{datetime_string}.ms'))
    assert len(ms_list) == 22
    orca.transform.imaging.make_dirty_image(ms_list, out_dir, datetime_string)
