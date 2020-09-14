import logging
from datetime import datetime
import sys
from typing import Optional, List

from orca.flagging import flagoperations, flag_bad_chans
from orca.proj.celery import app
from orca.wrapper import dada2ms, change_phase_centre
from orca.transform import peeling, integrate, gainscaling, spectrum, calibration, image_sub
from orca.utils import fitsutils

import casatasks

import billiard

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


"""
Celery adapter on top of transforms.
"""
@app.task
def run_dada2ms(dada_file: str, out_ms: str, gaintable: Optional[str] = None, addspw: bool = False) -> str:
    dada2ms.dada2ms(dada_file, out_ms, gaintable, addspw)
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
def do_calibration(ms_file):
    return calibration.calibration_steps(ms_file)
    
@app.task
def do_bandpass_correction(spectrum_file, bcal_file=None, plot=False):
    return calibration.bandpass_correction(spectrum_file, bcal_file, plot)

@app.task
def do_applycal(ms_file: str, cal_tables: List[str]) -> str:
    casatasks.applycal(ms_file, gaintable=cal_tables, flagbackup=False)
    return ms_file

@app.task
def apply_a_priori_flags(ms_file: str, flag_npy_path: str) -> str:
    return flagoperations.write_to_flag_column(ms_file, flag_npy_path)

@app.task
def apply_ant_flag(ms_file: str, ants: list) -> str:
    from casacore.tables import table, taql
    with table(ms_file, ack=False) as t:
        taql(f"update $t set FLAG=True where any(ANTENNA1==$ants || ANTENNA2==$ants)")
    return ms_file


@app.task
def apply_bl_flag(ms_file: str, bl_file: str) -> str:
    return flagoperations.flag_bls(ms_file, bl_file)


@app.task
def flag_chans(ms: str, spw: str, crosshand: bool = False, uvcut_m: float = None) -> str:
    return flag_bad_chans.flag_bad_chans(ms, spw, apply_flag=True, crosshand=crosshand, uvcut_m=uvcut_m)


@app.task
def run_integrate_with_concat(ms_list: List[str], out_ms: str, phase_center: Optional[str] = None) -> str:
    return integrate.integrate(ms_list, out_ms, phase_center)


@app.task
def run_correct_gain_scaling(baseline_ms: str, target_ms: str, data_column='CORRECTED_DATA') -> str:
    return gainscaling.correct_scaling(baseline_ms, target_ms, data_column=data_column)


@app.task
def run_merge_flags(ms1: str, ms2: str) -> None:
    flagoperations.merge_flags(ms1, ms2)


@app.task
def run_image_sub(file1: str, file2: str, out_dir: str, out_prefix) -> str:
    return image_sub.image_sub(file1, file2, out_dir, out_prefix)


@app.task
def run_image_sub2(file1_file2: List[str], out_dir: str, out_prefix) -> str:
    assert len(file1_file2) == 2
    return image_sub.image_sub(file1_file2[0], file1_file2[1], out_dir, out_prefix)


@app.task
def run_co_add(fits_list: List[str], output_fits_path: str, header_index: Optional[int] = None) -> str:
    return fitsutils.co_add(fits_list, output_fits_path, header_index)


@app.task
def add(x: int, y: int) -> int:
    return x+y


@app.task
def str_concat(first, second, third=''):
    return f'{first}{second}{third}'


@app.task
def test_multiprocessing():
    def worker(num):
        return num

    for i in range(5):
        p = billiard.Process(target=worker, args=(i,))
        p.start()
