from datetime import datetime
from os import path
import shutil
import os
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from casacore.tables import table
import shutil
from casatasks import applycal

from orca.metadata.stageiii import StageIIIPathsManager, spws
from orca.transform.integrate import integrate
from orca.transform.flagging import flag_with_aoflagger, flag_ants
from orca.wrapper.wsclean import wsclean


SCRATCH_DIR = '/fast/yuping/'
NIGHTTIME_DIR = '/lustre/pipeline/night-time/'
WORK_DIR = '/lustre/celery/'

logger = logging.getLogger(__name__)

def applycal_data_col(ms: str, gaintable: str) -> str:
    applycal(ms, gaintable=gaintable, flagbackup=False, applymode='calflag')
    with table(ms, ack=False, readonly=False) as t:
        d = t.getcol('CORRECTED_DATA')
        t.removecols('CORRECTED_DATA')
        t.putcol('DATA', d)
    return ms


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s = datetime(2023, 12, 22, 13, 30, 0)
    e = datetime(2023, 12, 22, 14, 30, 0)
    tmpdir = f'{SCRATCH_DIR}/tmp-{str(uuid.uuid4())}'
    os.mkdir(tmpdir)
    spws_todo = spws[-10:]

    integrated_msl = []
    for spw in spws_todo:
        logger.info('applycal SPW %s', spw)
        pm = StageIIIPathsManager(NIGHTTIME_DIR, WORK_DIR, spw, s, e)
        msl = []
        with ThreadPoolExecutor(10) as pool:
            futures = [ pool.submit(shutil.copytree, ms, f'{tmpdir}/{path.basename(ms)}') for _, ms in pm.ms_list ]
            """
            for _, ms in pm.ms_list:
                # applycal
                msl.append(applycal_data_col(ms, pm.get_bcal_path(s.date()),
                                f'{tmpdir}/{path.basename(ms)}'))
            """
            for r in as_completed(futures):
                m = r.result()
                applycal_data_col(m, pm.get_bcal_path(s.date()))
                msl.append(m)

        msl.sort()
        logger.info('integrate SPW %s', spw)
        integrated = integrate(msl, f'{tmpdir}/{spw}.ms')
        for ms in msl:
           shutil.rmtree(ms)
       # flag
        logger.info('flag SPW %s', spw)
        flag_ants(integrated, [70,79,80,117,137,193,150,178,201,208,224,261,215,236,246,294,298,301,307,289,33,3,41,42,44,92,12,14,17,21,154,29,28,127,126])
        flag_with_aoflagger(integrated)
        integrated_msl.append(integrated)

    logger.info('Done with all spws.')
    # image
    wsclean(integrated_msl, tmpdir, '1hour',
            extra_arg_list=['-weight', 'briggs', '0', '-niter', '0', '-size', '4096', '4096', '-scale', '0.03125'],
            num_threads=10, mem_gb=200)
