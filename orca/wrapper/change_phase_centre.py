import subprocess
from casacore.tables import table
from astropy.coordinates import SkyCoord
import os
from typing import Tuple, List, Dict

CHGCENTRE_PATH = '/opt/bin/chgcentre'


def change_phase_center(ms: str, center_dir: str) -> str:
    args = _get_subprocess_args(ms, center_dir)
    try:
        p = subprocess.Popen(args)
        p.communicate()
    except subprocess.CalledProcessError as e:
        raise e
    finally:
        p.terminate()
    return ms


def get_phase_center(ms: str) -> SkyCoord:
    with table(f'{ms}/FIELD', ack=False) as t:
        ra, dec = t.getcol('PHASE_DIR')[0][0]
    return SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian')


def _get_subprocess_args(ms: str, center_dir: str) -> List[str]:
    ra, dec = center_dir.split(' ')
    return [CHGCENTRE_PATH, ms, ra, dec]
