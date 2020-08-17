import subprocess
from casacore.tables import table
from astropy.coordinates import SkyCoord
import os
from typing import Tuple, List, Dict

CHGCENTRE_PATH = '/opt/astro/wsclean/bin/chgcentre'
NEW_ENV = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/:/opt/astro/casacore-1.7.0/lib',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')


def change_phase_center(ms: str, center_dir: str) -> str:
    args = _get_subprocess_args(ms, center_dir)
    try:
        p = subprocess.Popen(args, env=NEW_ENV)
        p.communicate()
    except subprocess.CalledProcessError as e:
        raise e
    finally:
        p.terminate()
    return ms


def get_phase_center(ms: str) -> str:
    with table(f'{ms}/FIELD') as t:
        ra, dec = t.getcol('PHASE_DIR')[0][0]
    return SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian').to_string('hmsdms')


def _get_subprocess_args(ms: str, center_dir: str) -> List[str]:
    ra, dec = center_dir.split(' ')
    return [CHGCENTRE_PATH, ms, ra, dec]
