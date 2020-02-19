import subprocess
from casacore.tables import table
from astropy.coordinates import SkyCoord
import os


CHGCENTRE_PATH = '/opt/astro/wsclean/bin/chgcentre'


def change_phase_center(ms: str, center_dir: str) -> None:
    new_env = dict(os.environ, LD_LIBRARY_PATH='/opt/astro/mwe/usr/lib64:/opt/astro/lib/:/opt/astro/casacore-1.7.0/lib',
                   AIPSPATH='/opt/astro/casa-data dummy dummy')
    ra, dec = center_dir.split(' ')
    try:
        p = subprocess.Popen([CHGCENTRE_PATH, ms, ra, dec], env=new_env)
        p.communicate()
    except subprocess.CalledProcessError as e:
        raise e


def get_phase_center(ms: str) -> str:
    with table(f'{ms}/FIELD') as t:
        ra, dec = t.getcol('PHASE_DIR')[0][0]
    return SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian').to_string('hmsdms')

