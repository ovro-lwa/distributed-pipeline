from pathlib import Path
from typing import List

import numpy as np

from orca.transform import photometry
from orca.metadata.stageiii import spws
from orca.utils.coordutils import TAU_BOO

WORK_DIR = Path('/lustre/celery')
BAND1 = spws[1:3]
BAND2 = spws[3:7]
BAND3 = spws[7:]

def fits_for_band(band: List[str], parts:str, parent: Path) -> List[str]:
    paths: List[str] = []
    for spw in band:
        p = parent / spw / parts
        if p.exists():
            paths.append(p.as_posix())
    return paths

if __name__ == '__main__':
    for pol in ('I', 'V'):
        parent = WORK_DIR / f'{pol}.image.fits/'
        fits_paths = parent.glob('50MHz/2024-05-??/??/*fits')
        for p in fits_paths:
            for band in (BAND1, BAND2, BAND3):
                to_coadd = fits_for_band(band, '/'.join(p.parts[5:]), parent)
                out_dir = WORK_DIR / f'{pol}.fcoadds.fits' / band[0] / '/'.join(p.parts[5:7])
                out_dir.mkdir(parents=True, exist_ok=True)
                out_fn = (out_dir / p.name).as_posix()
                photometry.average_with_rms_threshold.apply_async((
                    to_coadd,
                    out_fn,
                    None, 20, np.inf),
                    queue='qa')