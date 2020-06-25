import glob
import os
import logging
from datetime import datetime
import sys
from typing import Optional

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
def run_dada2ms(dada_file: str, out_ms: str, gaintable: Optional[str] = None) -> str:
    dada2ms.dada2ms(dada_file, out_ms, gaintable)
    return out_ms


@app.task
def run_chgcentre(ms_file: str, direction: str) -> str:
    return change_phase_centre.change_phase_center(ms_file, direction)


@app.task
def peel(ms_file: str, utc_datetime: str) -> str:
    return peeling.ttcal_peel_from_data_to_corrected_data(ms_file,
                                                          datetime.strptime(utc_datetime, "%Y-%m-%dT%H:%M:%S"))

@app.task
def zest(ms_file):
    return peeling.zest_with_casa(ms_file)

@app.task
def get_spectrum(ms_file: str, source: str, data_column: str = 'CORRECTED_DATA', timeavg: bool = False) -> str:
    return spectrum.gen_spectrum(ms_file, source, data_column, timeavg)


@app.task
def apply_a_priori_flags(ms_file: str, flag_npy_path: str) -> str:
    return flagoperations.write_to_flag_column(ms_file, flag_npy_path)


@app.task
def apply_bl_flag(ms_file: str, bl_file: str) -> str:
    return flagoperations.flag_bls(ms_file, bl_file)


@app.task
def flag_chans(ms: str, spw: str) -> str:
    return flag_bad_chans.flag_bad_chans(ms, spw, apply_flag=True)


@app.task
def make_first_image(prefix: str, datetime_string: str, out_dir: str) -> str:
    logging.info(f'Glob statement is {prefix}/{datetime_string}/??_{datetime_string}.ms')
    os.makedirs(out_dir,exist_ok=True)
    ms_list = sorted(glob.glob(f'{prefix}/{datetime_string}/??_{datetime_string}.ms'))
    return orca.transform.imaging.make_dirty_image(ms_list, out_dir, datetime_string)


@app.task
def run_integrate_with_concat(ms_list: str, out_ms: str, phase_center: Optional[str] = None) -> str:
    return integrate.integrate(ms_list, out_ms, phase_center)


@app.task
def run_correct_scaling(baseline_ms: str, target_ms: str, data_column='CORRECTED_DATA') -> str:
    return gainscaling.correct_scaling(baseline_ms, target_ms, data_column=data_column)


@app.task
def run_merge_flags(ms1: str, ms2: str) -> None:
    flagoperations.merge_flags(ms1, ms2)


@app.task
def add(x: int, y: int) -> int:
    return x+y


@app.task
def str_concat(first, second, third=''):
    return f'{first}{second}{third}'
