from .celery import app
from typing import Tuple
from datetime import datetime
import casacore.tables as pt
import numpy as np
from os import path

@app.task
def autocorr_sum(msfile: str) -> Tuple[datetime, str, np.ndarray]:
    # Parse the file name to get timestamp and spw
    split1 = path.basename(msfile).split('_')
    spw = split1[0]
    timestamp = datetime.strptime(split1[1].rstrip('.ms'), '%Y-%m-%dT%H:%M:%S')  # type: ignore
    with pt.table(msfile) as t:
        autocorr = t.query('ANTENNA1==ANTENNA2')
        flagcol = autocorr.getcol('FLAG')
        datacol = autocorr.getcol('CORRECTED_DATA')

        flagarr = flagcol[:, :, 0] | flagcol[:, :, 3]
        datacolxx = np.where(~flagarr, datacol[:, :, 0], 0)
        datacolyy = np.where(~flagarr, datacol[:, :, 3], 0)

        power = np.abs(datacolxx[:, 50])
    return timestamp, spw, power.tolist()
