from orca.celery import app
from os import path
import logging
import numpy as np

from casatasks import ft, bandpass
from casacore.tables import table

from orca.utils.calibrationutils import gen_model_ms_stokes

logger = logging.getLogger(__name__)

@app.task
def di_cal(ms, do_polcal=False, refant='199') -> str:
    """ Perform DI calibration and solve for cal table.

    Args:
        ms_list: Measurement set to solve with
        do_polcal: Do polarization calibration. Default is False.

    Returns: Path to the derived cal table.

    """
    clfile = gen_model_ms_stokes(ms)
    ft(ms, complist = clfile)
    bcalfile = path.splitext(ms)[0]+'.bcal'
    Xcalfile = path.splitext(ms)[0]+'.X'
    dcalfile = path.splitext(ms)[0]+'.dcal'
    if do_polcal:
        raise NotImplementedError('Polarization calibration not yet implemented.')
    bandpass(ms, bcalfile, refant=refant, uvrange='>100m', combine='scan,field,obs',
        fillgaps=5)
    flag_bad_sol(bcalfile)
    return bcalfile


def flag_bad_sol(bcal:str) -> str:
    """ Flag bad solutions in bandpass calibration. Modify the bandpass table.
    """
    with table(bcal, ack=False, readonly=False) as t:
        gain_amps = np.abs(t.getcol('CPARAM'))
        flag = t.getcol('FLAG')
        bad = (gain_amps < 0.1 * np.median(gain_amps))
        flag = flag | bad
        t.putcol('FLAG', flag)
    logger.info(f'Flag {len(bad)} dead antennas in {bcal}.')
    return bcal