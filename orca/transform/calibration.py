from os import path
import logging
import tempfile
import shutil

import numpy as np
from casatasks import ft, bandpass
from casacore.tables import table

from orca.celery import app

from orca.utils.calibrationutils import gen_model_ms_stokes

from orca.transform.integrate import integrate

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
def di_cal_multi(ms_list, scrach_dir, out, do_polcal=False, refant='199') -> str:
    """ Perform DI calibration on multiple integrations. Copy, concat, then solve.

    Args:
        ms_list: List of measurement sets to solve with
        scrach_dir: Directory to store temporary files
        out: Output path for the derived cal table.
        do_polcal: Do polarization calibration. Default is False.

    Returns: List of paths to the derived cal tables.
    """
    with tempfile.TemporaryDirectory(dir=scrach_dir) as tmpdir:
        msl = []
        for m in ms_list:
            target = f'{tmpdir}/{m.name}'
            shutil.copytree(m.absolute(), target)
            msl.append(target)
        concat = integrate(msl, f'{tmpdir}/CONCAT.ms')
        return di_cal(concat, do_polcal, refant, out)


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