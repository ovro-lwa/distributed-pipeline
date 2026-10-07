"""Spectral-line (RRL) Stokes-I cube helpers for the subband pipeline.

WSClean with ``-channels-out N`` writes one FITS file per channel per
product (``<base>-0000-image.fits`` ... plus ``<base>-MFS-*.fits``).  These
helpers build the cube-specific WSClean arguments, dewarp each channel with a
frequency-scaled ionospheric screen, and stack the per-channel planes into
single FITS cubes so the archived product is one file per product type.
"""
import os
import re
import glob
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# <base>-0012-<rest>.fits  (rest = e.g. "image-20250421_122430.pbcorr_dewarped")
_CHAN_RE = re.compile(r'^(?P<base>.+)-(?P<chan>\d{4})-(?P<rest>.+)\.fits$')


def get_ms_nchan(ms_path: str) -> int:
    """Return the number of channels in the first spectral window of *ms_path*."""
    import casacore.tables as pt
    with pt.table(os.path.join(ms_path, 'SPECTRAL_WINDOW'), ack=False) as t:
        return int(t.getcell('NUM_CHAN', 0))


def cube_niter(niter_ref: int, nchan: int) -> int:
    """Per-channel CLEAN depth: ``niter_ref / sqrt(nchan)``."""
    return max(1, int(round(niter_ref / np.sqrt(max(nchan, 1)))))


def patch_cube_args(args: List[str], nchan: int, niter: int) -> List[str]:
    """Return a copy of *args* with ``-channels-out`` and ``-niter`` replaced."""
    args = list(args)
    for flag, value in (('-channels-out', nchan), ('-niter', niter)):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args = [flag, str(value)] + args
    return args


def image_freq_hz(header) -> Optional[float]:
    """Frequency of a WSClean image from its FREQ axis (CRVAL3)."""
    for i in range(1, header.get('NAXIS', 0) + 1):
        if str(header.get(f'CTYPE{i}', '')).upper().startswith('FREQ'):
            return float(header[f'CRVAL{i}'])
    freq = header.get('CRVAL3', header.get('RESTFRQ'))
    return float(freq) if freq is not None else None


def dewarp_channel_images(
    files: List[str], screen_x: np.ndarray, screen_y: np.ndarray,
    ref_freq_hz: float,
) -> int:
    """Apply an ionospheric warp screen to each channel image.

    The screen was measured at *ref_freq_hz*; refractive offsets scale as
    nu^-2, so each channel uses ``screen * (ref_freq / freq)**2``.
    Writes ``<name>_dewarped.fits`` next to each input.

    Returns:
        Number of images dewarped.
    """
    from orca.transform.ionospheric_dewarping import apply_warp

    n_done = 0
    for f in files:
        out = f.replace('.fits', '_dewarped.fits')
        if os.path.exists(out):
            continue
        try:
            with fits.open(f) as hdul:
                header = hdul[0].header
                data = hdul[0].data.squeeze()
                freq = image_freq_hz(header) or ref_freq_hz
                scale = (ref_freq_hz / freq) ** 2
                warped = apply_warp(data, screen_x * scale, screen_y * scale)
                if warped is None:
                    continue
                fits.writeto(out, warped.reshape(hdul[0].data.shape).astype(np.float32),
                             header, overwrite=True)
                n_done += 1
        except Exception as e:
            logger.error(f"Cube dewarp failed for {os.path.basename(f)}: {e}")
    return n_done


def group_channel_files(target_dir: str, base: str) -> Dict[str, List[Tuple[int, str]]]:
    """Group per-channel WSClean outputs of *base* by product suffix.

    Returns:
        ``{rest: [(chan, path), ...]}`` sorted by channel, where *rest* is
        the part of the filename after the channel index.
    """
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(target_dir, f"{base}-*.fits")):
        m = _CHAN_RE.match(os.path.basename(f))
        if m and m.group('base') == base:
            groups[m.group('rest')].append((int(m.group('chan')), f))
    return {k: sorted(v) for k, v in groups.items()}


def stack_channels_to_cube(
    channel_files: List[str], out_path: str,
) -> str:
    """Stack single-channel WSClean FITS images into one (STOKES, FREQ, Y, X) cube.

    The header of the first channel is reused with the FREQ axis rewritten
    to a linear axis (WSClean channels are evenly spaced).  Per-channel
    frequencies and restoring beams are also written to a ``CHANNELS``
    binary table extension.
    """
    planes, freqs, beams = [], [], []
    header0 = None
    for f in channel_files:
        with fits.open(f) as hdul:
            h = hdul[0].header
            if header0 is None:
                header0 = h.copy()
            planes.append(np.asarray(hdul[0].data, dtype=np.float32).squeeze())
            freqs.append(image_freq_hz(h) or np.nan)
            beams.append((h.get('BMAJ', np.nan), h.get('BMIN', np.nan),
                          h.get('BPA', np.nan)))

    cube = np.stack(planes)[np.newaxis]  # (1, nchan, ny, nx)
    freqs = np.asarray(freqs, dtype=np.float64)

    hdr = header0.copy()
    for key in ('NAXIS3', 'NAXIS4'):
        hdr.pop(key, None)
    # FITS axis order (x, y, freq, stokes) == numpy (stokes, freq, y, x)
    hdr['CTYPE3'] = 'FREQ'
    hdr['CUNIT3'] = 'Hz'
    hdr['CRPIX3'] = 1.0
    hdr['CRVAL3'] = float(freqs[0])
    hdr['CDELT3'] = float(np.median(np.diff(freqs))) if len(freqs) > 1 else \
        float(header0.get('CDELT3', 0.0))
    hdr['CTYPE4'] = header0.get('CTYPE4', 'STOKES')
    hdr['CRPIX4'] = header0.get('CRPIX4', 1.0)
    hdr['CRVAL4'] = header0.get('CRVAL4', 1.0)
    hdr['CDELT4'] = header0.get('CDELT4', 1.0)
    hdr['NCHAN'] = (len(channel_files), 'Number of stacked channel images')

    primary = fits.PrimaryHDU(cube, header=hdr)
    beams = np.asarray(beams, dtype=np.float64)
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='CHAN', format='J', array=np.arange(len(freqs))),
        fits.Column(name='FREQ', format='D', unit='Hz', array=freqs),
        fits.Column(name='BMAJ', format='D', unit='deg', array=beams[:, 0]),
        fits.Column(name='BMIN', format='D', unit='deg', array=beams[:, 1]),
        fits.Column(name='BPA', format='D', unit='deg', array=beams[:, 2]),
    ], name='CHANNELS')
    fits.HDUList([primary, table]).writeto(out_path, overwrite=True)
    return out_path


def stack_cube_products(
    target_dir: str, base: str, remove_channels: bool = True,
) -> List[str]:
    """Stack every per-channel product of *base* in *target_dir* into cubes.

    ``<base>-0000-image-TS.pbcorr.fits`` ... becomes
    ``<base>-image-TS.pbcorr.fits`` (MFS files are left untouched).

    Args:
        remove_channels: Delete the per-channel files once the cube is written.

    Returns:
        Paths of the written cubes.
    """
    written = []
    for rest, chans in sorted(group_channel_files(target_dir, base).items()):
        files = [f for _, f in chans]
        out = os.path.join(target_dir, f"{base}-{rest}.fits")
        try:
            stack_channels_to_cube(files, out)
        except Exception as e:
            logger.error(f"Stacking {rest} ({len(files)} channels) failed: {e}")
            continue
        written.append(out)
        logger.info(f"Cube written: {os.path.basename(out)} ({len(files)} channels)")
        if remove_channels:
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
    return written
