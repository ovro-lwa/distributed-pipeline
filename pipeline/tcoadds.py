"""Time coadd pipeline for Tau Boötis observations.

This script coadds FITS images in 30-minute windows across time
for improved sensitivity in the Tau Boötis exoplanet search.
Organizes by frequency band and applies RMS thresholding.
"""
from pathlib import Path
from typing import List

import numpy as np

from orca.transform import photometry
from orca.metadata.stageiii import spws

WORK_DIR = Path('/lustre/celery')

BAND1 = spws[1:4]
BAND2 = spws[4:9]
BAND3 = spws[9:]

def coadd_30min(fns, pol):
    # within an hour
    fns.sort()
    fn0 = fns[0]
    out_dir = WORK_DIR / f'{pol}.30minfcoadds.fits' / '/'.join(fn0.parts[4:7])
    if len(fns) < 2:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fn in enumerate(fns[:-1]):
        to_coadd = [fn.as_posix(), fns[i+1].as_posix()]
        out_fn = (out_dir / fn.name).as_posix()
        photometry.average_with_rms_threshold.apply_async((
            to_coadd,
            out_fn,
            None, 20, np.inf),
            queue='qa')

def coadd_1hr(fns):
    # within an hour
    fns.sort()
    fn0 = fns[0]
    out_dir = WORK_DIR / f'{pol}.1hrfcoadds.fits' / '/'.join(fn0.parts[4:7])
    if len(fns) < 3:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    to_coadd = [ p.as_posix() for p in fns ]
    out_fn = (out_dir / fn0.name).as_posix()
    photometry.average_with_rms_threshold.apply_async((
        to_coadd,
        out_fn,
        None, 20, np.inf),
        queue='qa')

if __name__ == '__main__':
    for pol in ('I', 'V'):
        src = WORK_DIR / f'{pol}.fcoadds.fits/'
        for spw in (BAND1[0], BAND2[0], BAND3[0]):
            for band in (BAND1, BAND2, BAND3):
                day_hours = sorted(list(src.glob(f'{spw}/*/*')))
                for hr in day_hours:
                    # files within an hour
                    fns = sorted(list(hr.glob('*.fits')))
                    # coadd_30min(fns, pol)
                    coadd_1hr(fns)