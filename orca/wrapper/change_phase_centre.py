"""Phase center manipulation wrapper.

Provides Python interface to the chgcentre tool for re-phasing
visibility data to a new phase center.
"""
import subprocess
from casacore.tables import table
from astropy.coordinates import SkyCoord
from typing import List

CHGCENTRE_PATH = '/opt/bin/chgcentre'


def change_phase_center(ms: str, center_dir: str) -> str:
    """Change the phase center of a measurement set.

    Args:
        ms: Path to the measurement set (modified in-place).
        center_dir: New phase center as 'RA DEC' string in HMS/DMS format.

    Returns:
        Path to the modified measurement set.
    """
    args = _get_subprocess_args(ms, center_dir)
    try:
        p = subprocess.Popen(args)
        stdout, stderr = p.communicate()
    except subprocess.CalledProcessError as e:
        raise e
    finally:
        p.terminate()
    return ms


def get_phase_center(ms: str) -> SkyCoord:
    """Get the current phase center of a measurement set.

    Args:
        ms: Path to the measurement set.

    Returns:
        Phase center as an astropy SkyCoord object.
    """
    with table(f'{ms}/FIELD', ack=False) as t:
        ra, dec = t.getcol('PHASE_DIR')[0][0]
    return SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian')


def _get_subprocess_args(ms: str, center_dir: str) -> List[str]:
    """Build command-line arguments for chgcentre.

    Args:
        ms: Path to measurement set.
        center_dir: New phase center as 'RA DEC' string.

    Returns:
        List of command-line arguments.
    """
    ra, dec = center_dir.split(' ')
    return [CHGCENTRE_PATH, ms, ra, dec]
