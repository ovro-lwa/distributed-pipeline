from typing import List
from casacore.tables import table
import clickhouse_connect
import numpy as np
from orca.celery import app
from celery.utils.log import get_task_logger

from datetime import datetime
import os

logger = get_task_logger(__name__)

@app.task
def sanity_check(msl) -> List[str]:
    client = clickhouse_connect.get_client(host='10.41.0.85', username=os.getenv('CH_USER'),
                                  password=os.getenv('CH_PWD'))
    dt_list = []
    mhz_list = []
    zero_percent_list = []
    data = np.zeros((62128, 192, 4), dtype=np.complex128)
    bad_io = []
    for ms in msl:
        try:
            with table(ms, ack=False, readonly=True) as t:
            # get shape of DATA column
                if t.nrows() > data.shape[0]:
                    bad_io.append(ms)
                    continue
                t.getcolnp('DATA', data)
        except Exception as e:
            logger.error(f'Error in sanity check for {ms}: {e}')
            bad_io.append(ms)
            continue

        fn = os.path.basename(ms).rstrip('MHz.ms')
        date, time, mhz = fn.split('_')
        dt = datetime.strptime(f'{date}{time}', '%Y%m%d%H%M%S')
        zero_percent = int(100*(data == 0).sum() / data.size)
        zero_percent_list.append(zero_percent)
        dt_list.append(dt)
        mhz_list.append(int(mhz))
    client.insert('slowviz.zero_percent', [dt_list, mhz_list, zero_percent_list],
                  column_names=['timestamp', 'mhz', 'zero_percent'], column_oriented=True)
    return bad_io