"""Source catalog manipulation utilities.

Provides functions for creating, reading, and manipulating source catalogs
in astropy Table format. Uses column naming conventions compatible with
Aegean source finder.

Internal representation uses astropy.table.Table with columns derived from
Aegean's SimpleSource class (local_rms, ra, dec, peak_flux, peak_pixel,
x, y, a, b, pa, id).

FITS is the recommended storage format. Use astropy.Table.write/read for I/O.
"""
from typing import Dict, Optional
from astropy.table import Table
import numpy as np


def to_table(ra_abs, dec_abs, pkflux_abs, bmaj_abs, bmin_abs, bpa_abs, dateobs=None, jdobs=None,
             rmscell_abs=None, xpos_abs=None, ypos_abs=None, extra_meta: Optional[Dict] = None) -> Table:
    """
    Convert npz from Marin's source_find.py as
    Aegean SimpleSource-like catalog as astropy Table with metadata. Most of the arguments are familiar to
    source_find.py. These map to column names that are consistent with Aegean's.

    Args:
        ra_abs: Renamed ra
        dec_abs: Renamed dec
        pkflux_abs: Renamed peak_flux
        bmaj_abs: Array for source maj, renamed a in the Table representation.
        bmin_abs: Array for source min, renamed b in Table representation.
        bpa_abs:Renamed pa
        dateobs: date of the observation and goes to header
        jdobs: JD of observation and goes to header.
        rmscell_abs: renamed local_rms
        xpos_abs: Renamed x, optional
        ypos_abs: Renamed y, optional
        extra_meta: Extra metadata. Make sure that these are ok FITS header values (str, float, int, etc). Things like
            beam shape can go here.

    Returns:

    """
    if xpos_abs is None:
        data = {'ra': ra_abs, 'dec': dec_abs,
                'peak_flux': pkflux_abs, 'a': bmaj_abs, 'b': bmin_abs, 'pa': bpa_abs}
    else:
        data = {'x': xpos_abs, 'y': ypos_abs, 'ra': ra_abs, 'dec': dec_abs,
                'peak_flux': pkflux_abs, 'a': bmaj_abs, 'b': bmin_abs, 'pa': bpa_abs, 'local_rms': rmscell_abs}
    if dateobs is not None:
        if isinstance(dateobs, np.ndarray) and dateobs.dtype == '|S21':
            # When read from npz file that has dtype |21 as opposed to <U21
            # need the follows to extract the datetime as a string.
            default_meta = {'DATE': dateobs.tobytes().decode('utf8'), 'JDOBS': float(jdobs)}
        else:
            default_meta = {'DATE': str(dateobs), 'JDOBS': float(jdobs)}
    else:
        default_meta = {}

    if extra_meta:
        default_meta.update(extra_meta)
    return Table(data, meta=default_meta)


def add_id_column(t: Table) -> None:
    t.add_column(np.arange(len(t)), name='id')


def read_npz(npz_file: str):
    cat = np.load(npz_file)
    if 'xpos_abs' in cat:
        return to_table(ra_abs=cat['ra_abs'],
                        dec_abs=cat['dec_abs'], pkflux_abs=cat['pkflux_abs'], bmaj_abs=cat['bmaj_abs'],
                        bmin_abs=cat['bmin_abs'], bpa_abs=cat['bpa_abs'], dateobs=cat['dateobs'],
                        jdobs=cat['jdobs'], rmscell_abs=cat['rmscell_abs'],
                        xpos_abs=cat['xpos_abs'], ypos_abs=cat['ypos_abs'])
    else:
        return to_table(ra_abs=cat['ra_abs'],
                        dec_abs=cat['dec_abs'], pkflux_abs=cat['pkflux_abs'], bmaj_abs=cat['bmaj_abs'],
                        bmin_abs=cat['bmin_abs'], bpa_abs=cat['bpa_abs'])
