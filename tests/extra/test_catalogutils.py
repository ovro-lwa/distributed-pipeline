import pytest

from os import path
from datetime import datetime
import numpy as np

from orca.extra import catalogutils
from astropy.table import Table

TEST_NPZ = f'{path.dirname(__file__)}/../resources/diff_2018-03-22T15:57:44-image_sfind_marin.npz'


def test_npz_to_table():
    t = catalogutils.read_npz(TEST_NPZ)
    assert t.dtype == \
           np.dtype([('x', '<i8'), ('y', '<i8'), ('ra', '<f8'), ('dec', '<f8'), ('peak_flux', '<f8'),
                     ('a', '<f8'), ('b', '<f8'), ('pa', '<f8'), ('local_rms', '<f8')])
    assert 'JDOBS' in t.meta
    assert 'DATE' in t.meta
    # test casting to datetime
    datetime.strptime(str(t.meta['DATE']).split('.')[0], "%Y-%m-%dT%H:%M:%S")


def get_test_table():
    cat =np.load(TEST_NPZ)
    return catalogutils.to_table(xpos_abs=cat['xpos_abs'], ypos_abs=cat['ypos_abs'], ra_abs=cat['ra_abs'],
                                 dec_abs=cat['dec_abs'], pkflux_abs=cat['pkflux_abs'], bmaj_abs=cat['bmaj_abs'],
                                 bmin_abs=cat['bmin_abs'], bpa_abs=cat['bpa_abs'], dateobs=cat['dateobs'],
                                 jdobs=cat['jdobs'], rmscell_abs=cat['rmscell_abs'])


def test_create_table():
    t = get_test_table()
    for c in ('x', 'y', 'ra', 'dec'):
        assert c in t.columns


def test_write_catalog_table_to_fits(tmp_path):
    t = get_test_table()
    fits = tmp_path / 'cat.fits'
    t.write(fits.as_posix())
    t2 = Table.read(fits.as_posix())
    for col in t.columns:
        assert col in t2.columns

    for col in t2.columns:
        assert col in t.columns
        assert np.all(t[col] == t2[col])

    for k in t.meta:
        assert k in t2.meta


def test_add_id_column():
    t = get_test_table()
    catalogutils.add_id_column(t)
    assert t['id'][0] == 0
    assert t['id'][-1] == t['x'].shape[0] - 1