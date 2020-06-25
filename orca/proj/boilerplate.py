import glob
import os
import logging
import sys

import orca.transform.imaging
from orca.flagging import flagoperations, flag_bad_chans
from orca.proj.celery import app
from orca.wrapper import dada2ms, change_phase_centre, wsclean
from orca.transform import peeling, integrate, gainscaling, spectrum

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""
Celery adapter on top of transforms.
"""
@app.task
def run_dada2ms(dada_file, out_ms, gaintable=None):
    dada2ms.dada2ms(dada_file, out_ms, gaintable)
    return out_ms


@app.task
def run_chgcentre(ms_file, direction):
    return change_phase_centre.change_phase_center(ms_file, direction)

@app.task
def peel(ms_file, sources):
    return peeling.peel_with_ttcal(ms_file, sources)

@app.task
def zest(ms_file):
    return peeling.zest_with_casa(ms_file)

@app.task
def get_spectrum(ms_file, source, data_column='CORRECTED_DATA', timeavg=False):
    return spectrum.gen_spectrum(ms_file, source, data_column, timeavg)


@app.task
def apply_a_priori_flags(ms_file, flag_npy_path):
    return flagoperations.write_to_flag_column(ms_file, flag_npy_path)


@app.task
def apply_ant_flag(ms_file, ants):
    from casacore.tables import table, taql
    t = table(ms_file)
    taql(f"update $t set FLAG=True where any(ANTENNA1==$ants || ANTENNA2==$ants)")
    return ms_file


@app.task
def apply_bl_flag(ms_file, bl_file):
    return flagoperations.flag_bls(ms_file, bl_file)


@app.task
def flag_chans(ms, spw):
    return flag_bad_chans.flag_bad_chans(ms, spw, apply_flag=True)


@app.task
def make_first_image(prefix, datetime_string, out_dir):
    logging.info(f'Glob statement is {prefix}/{datetime_string}/??_{datetime_string}.ms')
    os.makedirs(out_dir,exist_ok=True)
    ms_list = sorted(glob.glob(f'{prefix}/{datetime_string}/??_{datetime_string}.ms'))
    assert len(ms_list) == 22
    return orca.transform.imaging.make_dirty_image(ms_list, out_dir, datetime_string)


@app.task
def run_integrate_with_concat(ms_list, out_ms, phase_center=None):
    return integrate.integrate(ms_list, out_ms, phase_center)


@app.task
def run_correct_scaling(baseline_ms, target_ms, data_column='CORRECTED_DATA'):
    return gainscaling.correct_scaling(baseline_ms, target_ms, data_column=data_column)


@app.task
def run_merge_flags(ms1, ms2):
    flagoperations.merge_flags(ms1, ms2)


@app.task
def add(x, y):
    return x+y

