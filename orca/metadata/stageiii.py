"""Stage III slow transient pipeline path management.

Provides the StageIIIPathsManager class for managing file paths in the
Stage III pipeline, which processes slow (10-second integration) visibility
data for transient and slow cadence science.

Classes
-------
StageIIIPathsManager
    Dataclass-based path manager for Stage III data products.
"""
from typing import List, Tuple, Union, Optional
from datetime import date, datetime, timedelta
from pathlib import Path

from dataclasses import dataclass

from orca.metadata.pathsmanagers import PathsManager

spws  = ['13MHz',  '18MHz',  '23MHz',  '27MHz', '32MHz',  '36MHz',  '41MHz',  '46MHz',  '50MHz',  '55MHz',  '59MHz',  
         '64MHz',  '69MHz',  '73MHz',  '78MHz',  '82MHz']

_DATETIME_FORMAT = '%Y%m%d_%H%M%S'
_DATE_FORMAT = '%Y%m%d'


@dataclass
class StageIIIPathsManager(PathsManager):
    """Path manager for Stage III slow transient pipeline.

    A dataclass-based path manager that handles measurement set discovery
    and data product path generation for the Stage III pipeline.

    Attributes
    ----------
    root_dir : str
        Root directory containing raw measurement sets.
    work_dir : str
        Working directory for derived data products.
    subband : str
        Frequency subband identifier (e.g., '73MHz').
    start : datetime
        Start time of the observation range.
    end : datetime
        End time of the observation range.
    partitioned_by_hour : bool
        Whether data is organized in hourly subdirectories.
    """

    root_dir: str
    work_dir: str
    subband: str
    start: datetime
    end: datetime
    partitioned_by_hour: bool = True
    
    def __post_init__(self):
        """Initialize path objects and reset cached MS list."""
        self._root_dir = Path(self.root_dir)
        self._work_dir = Path(self.work_dir)
        self._ms_list = None

    @property
    def ms_list(self) -> List[Tuple[datetime, Path]]:
        """List of (timestamp, path) tuples for measurement sets in the time range."""
        if self._ms_list is None:
            self._ms_list = [(datetime.strptime(ms.name[:-9], _DATETIME_FORMAT), ms.absolute().as_posix()) 
                             for ms in _get_ms_list(self._root_dir / self.subband, self.start, self.end, self.partitioned_by_hour)]
        return self._ms_list

    def get_bcal_path(self, bandpass_date: date, spw: Optional[str]=None) -> str:
        """Get the path to a bandpass calibration table.

        Args:
            bandpass_date: Date of the bandpass solution.
            spw: Spectral window. Defaults to this manager's subband.

        Returns:
            Absolute path to the bandpass calibration table.
        """
        spw = self.subband if spw is None else spw
        return self.get_gaintable_path(bandpass_date, spw, 'bcal')

    def get_gaintable_path(self, timestamp: Union[date, datetime], spw: str, gaintype: str) -> str:
        """Get the path to a gain calibration table.

        Args:
            timestamp: Date or datetime of the calibration solution.
            spw: Spectral window identifier.
            gaintype: Type of gain table (e.g., 'bcal', 'gcal').

        Returns:
            Absolute path to the gain table.
        """
        dir = self._work_dir / gaintype / spw
        fn = timestamp.strftime(_DATE_FORMAT if isinstance(timestamp, date) else _DATETIME_FORMAT) + '.' + gaintype
        return (dir / fn).absolute().as_posix()

    def time_filter(self, start_time: datetime, end_time: datetime) -> 'StageIIIPathsManager':
        """Create a new manager filtered to a specific time range.

        Args:
            start_time: New start time (inclusive).
            end_time: New end time (exclusive).

        Returns:
            New StageIIIPathsManager instance with filtered time range.
        """
        return StageIIIPathsManager(self.root_dir, self.work_dir, self.subband, start_time, end_time)

    def data_product_path(self, timestamp: datetime, suffix: str) -> str:
        """Generate path for a data product file.

        Args:
            timestamp: Timestamp of the observation.
            suffix: Product type suffix (e.g., 'fits', 'npz').

        Returns:
            Absolute path to the data product.
        """
        return (self._work_dir / suffix / self.subband /
                timestamp.date().isoformat() / f'{timestamp.hour:02d}' /
                (timestamp.isoformat() + f'.{suffix}')).absolute().as_posix()


def _get_ms_list(prefix: Path, start_time: datetime, end_time: datetime, partitioned_by_hour: bool) -> List[Path]:
    """Find measurement sets within a time range.

    Args:
        prefix: Base directory to search.
        start_time: Start of time range (inclusive).
        end_time: End of time range (inclusive).
        partitioned_by_hour: Whether MS files are in hourly subdirectories.

    Returns:
        List of Path objects for matching measurement sets, sorted by timestamp.
    """
    assert start_time <= end_time
    cur_time = start_time.replace(minute=0, second=0, microsecond=0)
    msl = []
    while cur_time <= end_time:
        if not partitioned_by_hour:
            msl += sorted([ p for p in prefix.glob(f'{cur_time.date()}/{cur_time.date().strftime(_DATE_FORMAT)}_{cur_time.hour:02d}*ms') ], key=lambda x: x.name)
        else:
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