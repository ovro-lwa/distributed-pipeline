import argparse
import glob
from orca.utils import calibrationutils
import numpy as np
import shutil
import logging, sys
import os
import datetime

from casacore.tables import table
from casatasks import ft, bandpass, gaincal, applycal
from scipy.stats import median_abs_deviation
from orca.flagging import flag_bad_chans
from orca.transform import imaging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

BAD_ANT_FILE = '/home/pipeline/bad_ants_latest.txt'

# main function that takes prefix with argparse
def make_cal_tables(prefix: str, refant: str, image: bool):
    # TODO copy the ms to /fast``
    ms_l = sorted(glob.glob(f"/data??/slow/{prefix}*.ms"))
    assert len(ms_l) > 0, 'No MS files found'
    assert len(ms_l) <= 2, 'More than two MS files found. Abort.'
    bad_ants = np.loadtxt(BAD_ANT_FILE, dtype=np.int32, delimiter=',')
    workdir = f'/fast/pipeline/tempcal/{prefix}/{datetime.datetime.now().isoformat()}'
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    for i, ms_orig in enumerate(ms_l):
        basename = os.path.basename(ms_orig)
        ms = f'/{workdir}/{basename}'
        shutil.copytree(ms_orig, ms)
        fn = ms.rstrip('.ms')
        with table(ms, ack=False, readonly=False) as t:
            ant1 = t.getcol('ANTENNA1')
            ant2 = t.getcol('ANTENNA2')
            flag = t.getcol('FLAG') | \
                np.isin(ant1, bad_ants)[:, None, None] | \
                np.isin(ant2, bad_ants)[:, None, None]
            t.putcol('FLAG', flag)
        gaincal(vis=ms, caltable=fn+'.K0', refant=refant, gaintype='K')
        with table(fn+'.K0', ack=False) as t:
            delays_mad = median_abs_deviation(t.getcol('FPARAM').flatten(), scale='normal')
            logger.info(f"Estimated scatter of delay sols: {delays_mad:.1f} ns.")
        clpath = calibrationutils.gen_model_ms_stokes(ms)
        if not clpath:
            raise ValueError(f'No calibrator sources found for {ms}.')
        ft(vis=ms, complist=clpath)
        caltable_path = fn + '.bcal'
        bandpass(vis=ms, caltable=caltable_path, refant=refant)
        shutil.copytree(caltable_path, f'/home/pipeline/caltables/staging/{os.path.basename(caltable_path)}')
        # TODO QA, plot, etc
        if image:
            applycal(vis=ms, gaintable=fn + '.bcal')
            flag_bad_chans.flag_bad_chans(ms, str(i), apply_flag=True)
            imaging.make_dirty_image([ms], workdir, basename ,briggs=0.5)
            
        logger.info('Working directory was %s.', workdir)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bandpass calibration')
    parser.add_argument('prefix', type=str, help='Timestamp prefix for the measurement set')
    parser.add_argument('refant', type=str, help='Refant')
    parser.add_argument('--image', action='store_true', default=False, help='Make images')
    args = parser.parse_args()
    try:
        make_cal_tables(args.prefix, args.refant, args.image)
    except Exception as e:
        logger.error(e)
        raise e
