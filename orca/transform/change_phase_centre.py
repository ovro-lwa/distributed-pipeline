import subprocess
from casacore.tables import table
from astropy.coordinates import SkyCoord

CHGCENTRE_PATH = '/opt/astro/wsclean/bin/chgcentre'


def change_phase_center(ms: str, center_dir: str) -> None:
    ra, dec = center_dir.split(' ')
    try:
        subprocess.check_output([CHGCENTRE_PATH, ms, ra, dec], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise e


def get_phase_center(ms: str) -> str:
    with table(f'{ms}/FIELD') as t:
        ra, dec = t.getcol('PHASE_DIR')[0][0]
    return SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian').to_string('hmsdms')

