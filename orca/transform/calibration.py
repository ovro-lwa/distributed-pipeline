from os import path
import logging
import shutil
import os
import uuid
import subprocess

import numpy as np
from casatasks import ft, bandpass, applycal
from casacore.tables import table

from orca.celery import app

from orca.utils.calibrationutils import gen_model_ms_stokes

from orca.transform.integrate import integrate
from orca.transform.precalflag import find_dead_ants
from orca.wrapper import change_phase_centre
from orca.flagging import flagoperations

logger = logging.getLogger(__name__)

@app.task
def di_cal(ms, out=None, do_polcal=False, refant='199') -> str:
    """ Perform DI calibration and solve for cal table.

    Args:
        ms: Measurement set to solve with
        out: Output path for the derived cal table (incl the table name). Default is None.
        do_polcal: Do polarization calibration. Default is False.

    Returns: Path to the derived cal table.

    """
    clfile = gen_model_ms_stokes(ms)
    ft(ms, complist = clfile)
    bcalfile = path.splitext(ms)[0]+'.bcal' if out is None else out
    Xcalfile = path.splitext(ms)[0]+'.X'
    dcalfile = path.splitext(ms)[0]+'.dcal'
    if do_polcal:
        raise NotImplementedError('Polarization calibration not yet implemented.')
    bandpass(ms, bcalfile, refant=refant, uvrange='>100m', combine='scan,field,obs',
        fillgaps=5)
    flag_bad_sol(bcalfile)
    return bcalfile


@app.task
def di_cal_multi(ms_list, scrach_dir, out, do_polcal=False, refant='199', flag_ant=True) -> str:
    """ Perform DI calibration on multiple integrations. Copy, concat, then solve.

    Args:
        ms_list: List of measurement sets to solve with
        scrach_dir: Directory to store temporary files
        out: Output path for the derived cal table.
        do_polcal: Do polarization calibration. Default is False.

    Returns: List of paths to the derived cal tables.
    """
    tmpdir = f'{scrach_dir}/tmp-{str(uuid.uuid4())}'
    os.mkdir(tmpdir)

    subprocess.check_call(['/usr/bin/cp', '-r'] + ms_list + [tmpdir])
    msl = []
    for m in ms_list:
        target = f'{tmpdir}/{path.basename(m)}'
        # shutil.copytree(m, target, copy_function=shutil.copyfile)
        msl.append(target)
        if flag_ant:
            dead_ants = find_dead_ants(target)
            flagoperations.flag_ants(target, dead_ants)

    concat = integrate(msl, f'{tmpdir}/CONCAT.ms', phase_center=change_phase_centre.get_phase_center(msl[len(msl)//2]))
    res = di_cal(concat, out, do_polcal=do_polcal, refant=refant)
    if path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    return res


def flag_bad_sol(bcal:str) -> str:
    """ Flag bad solutions in bandpass calibration. Modify the bandpass table.
    """
    with table(bcal, ack=False, readonly=False) as t:
        gain_amps = np.abs(t.getcol('CPARAM'))
        flag = t.getcol('FLAG')
        bad = (gain_amps < 0.01 * np.median(gain_amps))
        flag = flag | bad
        t.putcol('FLAG', flag)
    n_bad = np.sum(bad)
    if n_bad > 0:
        logger.info(f'Flagged {n_bad} sols that will blow up amplitude in {bcal}.')
    return bcal

@app.task
def applycal_data_col(ms: str, gaintable: str, out_ms: str) -> str:
    """ Apply calibration to the measurement set. Write to a new measurement set.

    Args:
        ms: Measurement set to apply calibration to
        caltable: Calibration table to apply
        out: Output path for the calibrated measurement set

    Returns: Path to the calibrated measurement set.
    """
    shutil.copytree(ms, out_ms)
    applycal(out_ms, gaintable=gaintable, flagbackup=False, applymode='calflag')
    with table(out_ms, ack=False, readonly=False) as t:
        d = t.getcol('CORRECTED_DATA')
        t.removecols('CORRECTED_DATA')
        t.putcol('DATA', d)
    return out_ms

def apply_bcal_in_mem(ms: str, bcal: str) -> np.ndarray:
    raise NotImplementedError