from datetime import datetime, date
from os import path
from typing import Optional, Union, Dict
from collections import OrderedDict

from orca.utils.datetimeutils import find_closest
import copy


class PathsManager(object):
    def __init__(self, utc_times_txt_path: str, dadafile_dir: Optional[str]):
        self.dadafile_dir = dadafile_dir
        # do the mapping thing
        self.utc_times_mapping = OrderedDict()
        with open(utc_times_txt_path) as f:
            for line in f:
                l = line.split()
                self.utc_times_mapping[datetime.strptime(f'{l[0]}T{l[1]}', "%Y-%m-%dT%H:%M:%S")] = l[2].rstrip('\n')

    def get_dada_path(self, spw: str, timestamp: datetime):
        return f'{self.dadafile_dir}/{spw}/{self.utc_times_mapping[timestamp]}'

    def time_filter(self, start_time: datetime, end_time: datetime) -> 'PathsManager':
        """
        Returns another PathsManager object with only utc_times between start_time (inclusive) and end_time (exclusive).
        Args:
            start_time:
            end_time:

        Returns:
            new_paths_manager: New PathsManager object with time filtered.
        """
        new_paths_manager = copy.deepcopy(self)
        new_paths_manager.utc_times_mapping = OrderedDict((k, v) for k,v in self.utc_times_mapping.items()
                                                          if start_time < k < end_time)
        return new_paths_manager


class OfflinePathsManager(PathsManager):
    """PathsManager for offline transient processing.

    This could potentially work for processing the buffer too. A config file reader will probably parse a config
    file into this object.

    Assumes that the bandpass calibration table is named like bcal_dir/00.bcal'
    """
    def __init__(self, utc_times_txt_path: str, dadafile_dir: Optional[str] = None, msfile_dir: Optional[str] = None,
                 gaintable_dir: str = None, flag_npy_paths: Optional[Union[str, Dict[date, str]]] = None):
        for d in (dadafile_dir, msfile_dir, gaintable_dir):
            if d and not path.exists(d):
                raise FileNotFoundError(f"File not found or path does not exist: {d}.")
        super().__init__(utc_times_txt_path, dadafile_dir)
        self.msfile_dir: Optional[str] = msfile_dir
        self.gaintable_dir: Optional[str] = gaintable_dir

        self.flag_npy_paths: Union[str, Dict[date, str], None] = flag_npy_paths

    def get_bcal_path(self,  bandpass_date: date, spw: str) -> str:
        """
        Return bandpass calibration path in /gaintable/path/2018-03-02/00.bcal
        Args:
            bandpass_date: Date of the bandpass solution.
            spw: Spectral window

        Returns:
            Bandpass calibration path.
        """
        return self.get_gaintable_path(bandpass_date, spw, 'bcal')

    def get_gaintable_path(self, bandpass_date: date, spw: str, extension: str) -> str:
        return f'{self.gaintable_dir}/{bandpass_date.isoformat()}/{spw}.{extension}'

    def get_ms_path(self, timestamp: datetime, spw: str) -> str:
        """
        Generate measurement set paths that look like
        /path/to/msfile/2018-03-02/hh=02/2018-03-02T02:02:02/00_2018-03-02T02:02:02.ms.
        Args:
            timestamp:
            spw:

        Returns:
            Path to the measurement set.
        """
        hour = f'{timestamp.hour:02d}'
        return f'{self.msfile_dir}/{timestamp.date().isoformat()}/hh={hour}/{timestamp.isoformat()}/' \
               f'{spw}_{timestamp.isoformat()}.ms'

    def get_flag_npy_path(self, timestamp: datetime) -> str:
        """
        Return the a priori npy for the flags column for a given time.
        Args:
            timestamp:

        Returns:
            The flags npy, if only one was supplied; or the closest npy, if a Dict[datetime, str] is supplied.
        """
        assert self.flag_npy_paths is not None
        if isinstance(self.flag_npy_paths, str):
            return self.flag_npy_paths
        elif isinstance(self.flag_npy_paths, dict):
            return self.flag_npy_paths[find_closest(timestamp, self.flag_npy_paths.keys())]
        else:
            raise ValueError(f'flag_npy_paths can only be str or dict. It is type {type(self.flag_npy_paths)}.')

