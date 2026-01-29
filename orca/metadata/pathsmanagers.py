"""Path management utilities for OVRO-LWA data access.

This module provides PathsManager classes that handle file path resolution
for measurement sets, calibration tables, and output directories. Supports
both offline batch processing and real-time pipeline modes.

Classes:
    PathsManager: Abstract base class defining the path management interface.
    OfflinePathsManager: Path manager for offline transient processing.
"""
from datetime import datetime, date, timedelta
from os import path
from typing import Optional, Union, Dict, List
from collections import OrderedDict
from abc import ABC, abstractmethod

from orca.utils.datetimeutils import find_closest
import copy

SIDEREAL_DAY = timedelta(hours=23, minutes=56, seconds=4)


class PathsManager(ABC):
    """Base PathsManager class.
    It contains functionality to manipulate datetime objects and find dada files. Maybe in the future it will evolve
    into an interface with abstract methods.

    Attributes:
        utc_times_mapping: An ordered dictionary mapping datetime objects to dada files.
    """
    @abstractmethod
    def time_filter(self, start_time: datetime, end_time: datetime) -> 'PathsManager':
        return NotImplemented

    @abstractmethod
    def get_bcal_path(self,  bandpass_date: date, spw: Optional[str]) -> str:
        return NotImplemented

class OfflinePathsManager(PathsManager):
    """PathsManager for offline transient processing.

    This could potentially work for processing the buffer too. A config file reader will probably parse a config
    file into this object.

    Assumes that the bandpass calibration table is named like bcal_dir/00.bcal'

    """
    def __init__(self, utc_times_txt_path: str, dadafile_dir: Optional[str] = None, working_dir: Optional[str] = None,
                 gaintable_dir: str = None, flag_npy_paths: Optional[Union[str, Dict[datetime, str]]] = None):
        for d in (dadafile_dir, working_dir, gaintable_dir):
            if d and not path.exists(d):
                raise FileNotFoundError(f"File not found or path does not exist: {d}.")
        self._dadafile_dir = dadafile_dir
        self.utc_times_txt_path = utc_times_txt_path
        # do the mapping thing
        self.utc_times_mapping = OrderedDict()
        with open(utc_times_txt_path) as f:
            for line in f:
                l = line.split()
                self.utc_times_mapping[datetime.strptime(f'{l[0]}T{l[1]}', "%Y-%m-%dT%H:%M:%S")] = l[2].rstrip('\n')
        self._working_dir: Optional[str] = working_dir
        self._gaintable_dir: Optional[str] = gaintable_dir
        self._flag_npy_paths: Optional[Union[str, Dict[datetime, str]]] = flag_npy_paths

    @property
    def working_dir(self) -> str:
        if self._working_dir:
            return self._working_dir
        else:
            raise ValueError('working_dir not set.')

    @property
    def gaintable_dir(self) -> str:
        if self._gaintable_dir:
            return self._gaintable_dir
        else:
            raise ValueError('gaintable_dir not set.')

    def get_bcal_path(self,  bandpass_date: date, spw: str) -> str:
        """Return bandpass calibration path in /gaintable/path/2018-03-02/00.bcal.

        Args:
            bandpass_date: Date of the bandpass solution requested.
            spw: Spectral window

        Returns:
            Bandpass calibration path.
        """
        return self.get_gaintable_path(bandpass_date, spw, 'bcal')

    def get_gaintable_path(self, gaintable_date: date, spw: str, extension: str) -> str:
        """Get the path to a certain CASA gaintable.

        Args:
            gaintable_date: date of the table requested
            spw: spw of the gaintable requested
            extension: extension of the gaintable (bcal etc)

        Returns:
            The path to the requested gaintable.
        """
        return f'{self.gaintable_dir}/{gaintable_date.isoformat()}/{spw}.{extension}'

    def get_ms_path(self, timestamp: datetime, spw: str) -> str:
        """Generate measurement set paths that look like
        /path/to/working_dir/msfiles/2018-03-02/hh=02/2018-03-02T02:02:02/00_2018-03-02T02:02:02.ms.

        Args:
            timestamp: Timestamp of the ms.
            spw: Spectral window of the ms.

        Returns:
            Path to the measurement set.
        """
        parent = self.get_ms_parent_path(timestamp)
        return f'{parent}/{spw}_{timestamp.isoformat()}.ms'

    def get_ms_parent_path(self, timestamp: datetime) -> str:
        """Generate measurement set parent paths that look like
        /path/to/working_dir/msfiles/2018-03-02/hh=02/2018-03-02T02:02:02/.

        Args:
            timestamp: Timestamp of the ms.

        Returns:
            Path to the measurement set.
        """
        hour = f'{timestamp.hour:02d}'
        return f'{self.working_dir}/msfiles/{timestamp.date().isoformat()}/hh={hour}/{timestamp.isoformat()}'

    def get_data_product_path(self, timestamp: datetime, product: str, file_suffix: str,
                              file_prefix: Optional[str] = None) -> str:
        """Generate path for generic data product.
        Looks like /path/to/working_dir/<product>/2018-03-02/hh=02/<file_prefix>_2018-03-02T02:02:02<file_suffix>.
        The first underscore is not there is file_prefix=None

        Args:
            timestamp: Timestamp of observation.
            product: Name of the data product to be used for top-level directory
            file_suffix: Suffix to data file. For example for fits file it'd be '.fits'.
                You can also have something like '_v2.fits'
            file_prefix: Prefix of the file. Can be spectral window. If none specified then no prefix.

        Returns: Full path to the data product.
        """
        assert product, 'The product variable cannot be None or an empty string'
        hour = f'{timestamp.hour:02d}'
        if file_prefix:
            return f'{self.working_dir}/{product}/{timestamp.date().isoformat()}/hh={hour}/' \
                   f'{file_prefix}_{timestamp.isoformat()}{file_suffix}'
        else:
            return f'{self.working_dir}/{product}/{timestamp.date().isoformat()}/hh={hour}/' \
                   f'{timestamp.isoformat()}{file_suffix}'

    dpp = get_data_product_path

    def get_flag_npy_path(self, timestamp: datetime) -> str:
        """ Return the a priori npy for the flags column for a given time.

        Args:
            timestamp:

        Returns:
            If only one flag_npy was supplied, the flag_npy; if a Dict[datetime, str] is supplied, the closest one in
            time to the supplied timestamp.
        """
        assert self._flag_npy_paths is not None
        if isinstance(self._flag_npy_paths, str):
            return self._flag_npy_paths
        elif isinstance(self._flag_npy_paths, dict):
            return self._flag_npy_paths[find_closest(timestamp, self._flag_npy_paths.keys())]
        else:
            raise ValueError(f'flag_npy_paths can only be str or dict. It is type {type(self._flag_npy_paths)}.')

    def time_filter(self, start_time: datetime, end_time: datetime) -> 'OfflinePathsManager':
        """
        Returns another PathsManager object with only utc_times between start_time (inclusive) and end_time (exclusive).

        Args:
            start_time:
            end_time:

        Returns:
            new_paths_manager: New PathsManager object with time filtered.
        """
        new_paths_manager = copy.deepcopy(self)
        if start_time == end_time:
            new_paths_manager.utc_times_mapping = OrderedDict([(start_time, self.utc_times_mapping[start_time])])
        else:
            new_paths_manager.utc_times_mapping = OrderedDict((k, v) for k, v in self.utc_times_mapping.items()
                                                              if start_time <= k < end_time)
        return new_paths_manager

    def chunks_by_integration(self, chunk_size: int) -> List[List[datetime]]:
        """
        Chunk the datetime array by number of integrations such that each chunk contains data spanning equal or
        less than the chunk size. Note that the last chunk may be smaller, if the total number of
        integrations is not divisible by the chunk size.

        Args:
            chunk_size: number of integrations per chunk

        Returns: A list whose elements are the ordered chunks, which are each a list of ordered timestamps.

        """
        datetimes = list(self.utc_times_mapping.keys())
        return [datetimes[i:i+chunk_size] for i in range(0, len(datetimes), chunk_size)]

    def chunks_by_time(self, chunk_time: timedelta) -> List[List[datetime]]:
        """
        Chunk the datetime array by time such that each chunk contains data spanning equal or less than chunk_time
        amount of time. All of the data will be chunked. Note that

        1) The last chunk may be smaller, if the span of the data is not divisible by the chunk time

        2) Chunking is based on time and not by the number of integrations. Therefore, some chunks might have more or
        fewer integrations than some other chunks, if the chunk time is not divisible by the integration time.

        Args:
            chunk_time: a timedelta object for the chunk time.

        Returns: A list whose elements are the ordered chunks, which are each a list of ordered timestamps.

        """
        raise NotImplementedError
