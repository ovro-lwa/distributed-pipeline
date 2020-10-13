from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord

from casacore.tables import table

INVERSE_C_MS = 3.3356409519815204e-09


def phase_shift_vis(t: table, freqs: np.array, phase_center: SkyCoord, pos: SkyCoord, columnname: str,
                    startrow: int=0, nrow: int=-1, rowincr: int = 1) -> np.array:
    """
    Phase shift visibility data column and return the shifted data. It does not write back to the table, nor does it
    shift the UVW coordinates. Only works with data with one spectral window.

    Args:
        t: table containing the data
        freqs: (1, N_chan) channel array. Can get from tspw.getcol('CHAN_FREQ').
        phase_center: current phase center.
        pos: position to shift to.
        columnname: column name for the data.
        startrow: see casacore getcol documentation.
        nrow: see casacore getcol documentation.
        rowincr: see casacore getcol documentation.

    Returns: the phased shifted data

    """
    data = t.getcol(columnname, startrow, nrow, rowincr)
    phase_data_to_pos_inplace(data,
                              t.getcol('UVW', startrow, nrow, rowincr),
                              freqs, phase_center.ra.rad, phase_center.dec.rad,
                              pos.ra.rad, pos.dec.rad)
    return data


def phase_data_to_pos_inplace(data: np.array, uvw: np.array, freqs: np.array,
                              ra0_rad: float, dec0_rad: float, ra_rad: float, dec_rad: float):
    n_freqs = freqs.shape[-1]
    n_uvws = uvw.shape[0]
    assert uvw.shape[1] == 3
    assert len(uvw.shape) == 2, 'uvw is assumed to be a 2-D array (n_uvw, 3)'
    assert len(freqs.shape) <= 2, 'freqs should have leq 2 dimensions (single spw probably).'
    assert data.shape[0] == n_uvws, 'Data is assumed to be ordered by N_UVWs and then N_FREQS'
    assert data.shape[1] == n_freqs, 'Data is assumed to be ordered by N_UVWs and then N_FREQS'

    freqs = freqs.astype(np.float32)
    uvw = uvw.astype(np.float32)
    lmn = np.empty(shape=3, dtype=np.float32)

    dra = ra_rad - ra0_rad
    dec = dec_rad
    dec0 = dec0_rad
    lmn[0] = np.sin(dra) * np.cos(dec)
    lmn[1] = np.sin(dec) * np.cos(dec0) - np.cos(dra) * np.cos(dec) * np.sin(dec0)
    lmn[2] = np.sin(dec) * np.sin(dec0) + np.cos(dra) * np.cos(dec) * np.cos(dec0) - 1 # makes the work below easier
    theta = -2 * np.pi * np.outer(uvw.dot(lmn), freqs) * INVERSE_C_MS
    ph = np.cos(theta) + 1j * np.sin(theta)
    data *= ph[:, :, np.newaxis]
