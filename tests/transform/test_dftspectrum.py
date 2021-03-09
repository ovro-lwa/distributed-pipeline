import pytest

import numpy as np
from astropy.coordinates import SkyCoord
from casacore.tables import table

from orca.transform import dftspectrum
from ..common import TEST_MS

from os import path


def test_dimensions():
    n_bl = 1000
    n_freqs = 109
    n_pols = 4

    data = np.zeros((n_bl, n_freqs, n_pols), dtype='complex64')
    uvw = np.zeros((n_bl, 3))
    freqs = np.zeros((1, n_freqs))
    dftspectrum.phase_data_to_pos_inplace(data, uvw, freqs, 0., 0., 0., 0.)


def test_against_chgcentre_spectrum():
    pos = SkyCoord('06h30m0s 45d05m45s')
    with table(f'{TEST_MS}/SPECTRAL_WINDOW') as tspw:
        freqs = tspw.getcol('CHAN_FREQ')
    with table(f'{TEST_MS}/FIELD') as tfield:
        ra, dec = tfield.getcol('PHASE_DIR')[0][0]
        phase_center = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='radian')
    with table(TEST_MS) as t:
        ans = np.mean(np.real(dftspectrum.phase_shift_vis(t, freqs, phase_center, pos, 'DATA')), axis=0)
    # from summing real parts of the chgcentre'd data column
    expected = np.load(f'{path.dirname(__file__)}/../resources/phased_spec.npy')
    corr_ratio = np.abs((ans - expected)/expected)
    assert max(corr_ratio[:, 0].max(), corr_ratio[:, 3].max()) < 1e-4
    assert max(corr_ratio[:, 1].max(), corr_ratio[:, 2].max()) < 1e-1
