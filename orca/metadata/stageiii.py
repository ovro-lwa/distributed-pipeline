from typing import List
from datetime import datetime, timedelta
from pathlib import Path

from orca.metadata.pathsmanagers import PathsManager

FORMAT = '%Y%m%d_%H%M%S'

class StageIIIPathsManager():
    def __init__(self, root_dir: str, subband: str, start: datetime, end: datetime):
        self._root_dir = Path(root_dir)
        self.subband = subband
        self.start = start
        self.end = end
        self.ms_list = _get_ms_list(self._root_dir / subband, start, end)

def _get_ms_list(prefix: Path, start_time: datetime, end_time: datetime) -> List[Path]:
    assert start_time <= end_time
    cur_time = start_time.replace(minute=0, second=0, microsecond=0)
    msl = []
    while cur_time <= end_time:
        msl += sorted([ p for p in prefix.glob(f'{cur_time.date().isoformat()}/{cur_time.hour:02d}/*ms') ], key=lambda x: x.name)
        cur_time += timedelta(hours=1)

    if not msl:
        return []

    i = 0
    for i, p in enumerate(msl):
        if datetime.strptime(p.name[:-9], FORMAT) >= start_time:
            break

    j = 0
    for j, p in enumerate(reversed(msl)):
        if datetime.strptime(p.name[:-9], FORMAT) <= end_time:
            break
    if j > 0:
            msl = msl[i:-j]
    else:
        msl = msl[i:]

    return msl