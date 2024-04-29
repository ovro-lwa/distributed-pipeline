from typing import List, Tuple, Union, Optional
from datetime import date, datetime, timedelta
from pathlib import Path

from dataclasses import dataclass

from orca.metadata.pathsmanagers import PathsManager

spws  = ['13MHz',  '18MHz',  '23MHz',  '27MHz', '32MHz',  '36MHz',  '41MHz',  '46MHz',  '50MHz',  '55MHz',  '59MHz',  
         '64MHz',  '69MHz',  '73MHz',  '78MHz',  '82MHz']

_DATETIME_FORMAT = '%Y%m%d_%H%M%S'
_DATE_FORMAT = '%Y%m%d'

# Use dataclass so that it's serializable
@dataclass
class StageIIIPathsManager(PathsManager):
    root_dir: str
    work_dir: str
    subband: str
    start: datetime
    end: datetime
    
    def __post_init__(self):
        self._root_dir = Path(self.root_dir)
        self._work_dir = Path(self.work_dir)
        self._ms_list = None

    @property
    def ms_list(self) -> List[Tuple[datetime, Path]]:
        if self._ms_list is None:
            self._ms_list = [(datetime.strptime(ms.name[:-9], _DATETIME_FORMAT), ms.absolute().as_posix()) 
                             for ms in _get_ms_list(self._root_dir / self.subband, self.start, self.end)]
        return self._ms_list

    def get_bcal_path(self, bandpass_date: date, spw: Optional[str]=None) -> str:
        spw = self.subband if spw is None else spw
        return self.get_gaintable_path(bandpass_date, spw, 'bcal')

    def get_gaintable_path(self, timestamp: Union[date, datetime], spw: str, gaintype: str) -> str:
        dir = self._work_dir / gaintype / spw
        fn = timestamp.strftime(_DATE_FORMAT if isinstance(timestamp, date) else _DATETIME_FORMAT) + '.' + gaintype
        return (dir / fn).absolute().as_posix()

    def time_filter(self, start_time: datetime, end_time: datetime) -> 'StageIIIPathsManager':
        return StageIIIPathsManager(self.root_dir, self.work_dir, self.subband, start_time, end_time)

    def data_product_path(self, timestamp: datetime, suffix: str) -> str:
        return (self._work_dir / suffix / self.subband /
                timestamp.date().isoformat() / f'{timestamp.hour:02d}' /
                (timestamp.isoformat() + f'.{suffix}')).absolute().as_posix()

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
        if datetime.strptime(p.name[:-9], _DATETIME_FORMAT) >= start_time:
            break

    j = 0
    for j, p in enumerate(reversed(msl)):
        if datetime.strptime(p.name[:-9], _DATETIME_FORMAT) <= end_time:
            break
    if j > 0:
            msl = msl[i:-j]
    else:
        msl = msl[i:]

    return msl