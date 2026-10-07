"""Tests for the Stokes-I spectral cube helpers (no casacore / Celery needed)."""
import numpy as np
import pytest
from astropy.io import fits

from orca.transform import cube_imaging as ci


BASE = '50MHz-I-Deep-Taper-Robust-0-cube'
TS = '20250421_122430'
F0, DF = 50.184e6, 95.703e3


def _write_wsclean_image(path, value, freq, bmaj=0.1, shape=(8, 8)):
    """Minimal 4-D WSClean-style image: numpy (stokes, freq, y, x)."""
    data = np.full((1, 1) + shape, value, dtype=np.float32)
    hdr = fits.Header()
    for i, (ctype, crval) in enumerate(
            [('RA---SIN', 270.0), ('DEC--SIN', 37.2), ('FREQ', freq), ('STOKES', 1.0)], 1):
        hdr[f'CTYPE{i}'] = ctype
        hdr[f'CRVAL{i}'] = crval
        hdr[f'CRPIX{i}'] = 1.0
        hdr[f'CDELT{i}'] = 1.0
    hdr['BMAJ'] = bmaj
    hdr['BMIN'] = bmaj / 2
    hdr['BPA'] = 10.0
    fits.writeto(path, data, hdr)


def test_patch_cube_args_replaces_channels_and_niter():
    args = ['-channels-out', '192', '-pol', 'I', '-niter', '500000']
    out = ci.patch_cube_args(args, 48, 72169)
    assert out == ['-channels-out', '48', '-pol', 'I', '-niter', '72169']
    assert args[1] == '192'  # config not mutated


def test_patch_cube_args_adds_missing_flags():
    out = ci.patch_cube_args(['-pol', 'I'], 48, 10)
    assert out[out.index('-channels-out') + 1] == '48'
    assert out[out.index('-niter') + 1] == '10'


def test_cube_niter_scales_with_sqrt_nchan():
    assert ci.cube_niter(500000, 48) == 72169
    assert ci.cube_niter(500000, 96) == 51031


def test_stack_cube_products_builds_cubes_and_keeps_mfs(tmp_path):
    nchan = 4
    for c in range(nchan):
        _write_wsclean_image(tmp_path / f'{BASE}-{c:04d}-image-{TS}.fits',
                             c, F0 + c * DF, bmaj=0.1 + c * 0.01)
        _write_wsclean_image(tmp_path / f'{BASE}-{c:04d}-image-{TS}.pbcorr.fits',
                             10 + c, F0 + c * DF)
    mfs = tmp_path / f'{BASE}-MFS-image-{TS}.fits'
    _write_wsclean_image(mfs, 99, F0 + 1.5 * DF)

    written = ci.stack_cube_products(str(tmp_path), BASE)

    assert sorted(p.split('/')[-1] for p in written) == [
        f'{BASE}-image-{TS}.fits', f'{BASE}-image-{TS}.pbcorr.fits']
    assert mfs.exists()
    assert not list(tmp_path.glob(f'{BASE}-0*'))  # per-channel files removed

    with fits.open(tmp_path / f'{BASE}-image-{TS}.fits') as h:
        assert h[0].data.shape == (1, nchan, 8, 8)
        np.testing.assert_array_equal(h[0].data[0, :, 0, 0], np.arange(nchan))
        assert h[0].header['CTYPE3'] == 'FREQ'
        assert h[0].header['CRVAL3'] == pytest.approx(F0)
        assert h[0].header['CDELT3'] == pytest.approx(DF)
        chans = h['CHANNELS'].data
        np.testing.assert_allclose(chans['FREQ'], F0 + np.arange(nchan) * DF)
        np.testing.assert_allclose(chans['BMAJ'], 0.1 + 0.01 * np.arange(nchan))


def test_channel_grouping_sorts_numerically_and_ignores_other_bases(tmp_path):
    for c in (10, 2, 0):
        _write_wsclean_image(tmp_path / f'{BASE}-{c:04d}-psf.fits', c, F0 + c * DF)
    _write_wsclean_image(tmp_path / f'{BASE}-other-0001-psf.fits', 0, F0)
    groups = ci.group_channel_files(str(tmp_path), BASE)
    assert list(groups) == ['psf']
    assert [c for c, _ in groups['psf']] == [0, 2, 10]


def test_dewarp_scales_screen_by_inverse_frequency_squared(tmp_path):
    shape = (16, 16)
    img = np.zeros(shape, dtype=np.float32)
    img[8, 8] = 1.0
    ref = F0
    files = []
    for name, freq in (('lo', ref), ('hi', 2 * ref)):
        p = tmp_path / f'{name}.fits'
        _write_wsclean_image(p, 0, freq, shape=shape)
        with fits.open(p, mode='update') as h:
            h[0].data[0, 0] = img
        files.append(str(p))

    # Screen of +4 px in x at ref freq -> +1 px at 2x the frequency.
    sx = np.full(shape, 4.0)
    sy = np.zeros(shape)
    assert ci.dewarp_channel_images(files, sx, sy, ref) == 2

    lo = fits.getdata(tmp_path / 'lo_dewarped.fits').squeeze()
    hi = fits.getdata(tmp_path / 'hi_dewarped.fits').squeeze()
    assert np.unravel_index(lo.argmax(), shape) == (8, 4)
    assert np.unravel_index(hi.argmax(), shape) == (8, 7)
    assert fits.getdata(tmp_path / 'lo_dewarped.fits').shape == (1, 1) + shape
