from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from typing import Optional


class PathsManager(ABC):
    def __init__(self, utc_times_txt_path: str, dadafile_dir: Optional[str]):
        self.dadafile_dir = dadafile_dir
        # do the mapping thing
        self.utc_times_mapping = {}
        with open(utc_times_txt_path) as f:
            for line in f:
                l = line.split()
                self.utc_times_mapping[datetime.strptime(f'{l[0]}T{l[1]}', "%Y-%m-%dT%H:%M:%S")] = l[2].rstrip('\n')

    def get_dada_path(self, spw: str, timestamp: datetime):
        return f'{self.dadafile_dir}/{spw}/{self.utc_times_mapping[timestamp]}'

    @abstractmethod
    def get_gaintable_path(self, spw: str) -> str:
        """
        Get path of gaintable closest to the timestamp at spw.
        :param spw:
        :return:
        """
        pass

    @abstractmethod
    def get_ms_path(self, timestamp: datetime, spw: str) -> str:
        """
        Get the path for a measurement set given the timestamp and the spw.
        :param timestamp:
        :param spw:
        :return:
        """
        pass

    @abstractmethod
    def get_flag_npy_path(self, timestamp: datetime) -> str:
        pass


class OfflinePathsManager(PathsManager):
    """PathsManager for offline processing.

    This could potentially work for processing the buffer too. A config file reader will probably parse a config
    file into this object.

    Assumes that the bandpass calibration table is named like bcal_dir/00.bcal'
    """
    def __init__(self, utc_times_txt_path: str, dadafile_dir: Optional[str]=None, msfile_dir: Optional[str]=None,
                 bcal_dir: str=None, flag_npy_path: str=None):
        for d in (dadafile_dir,msfile_dir, bcal_dir, flag_npy_path):
            if d and not path.exists(d):
                raise FileNotFoundError(f"File not found or path does not exist: {d}.")
        super().__init__(utc_times_txt_path, dadafile_dir)
        self.msfile_dir = msfile_dir
        self.bcal_dir = bcal_dir
        self.flag_npy_path = flag_npy_path

    def get_gaintable_path(self,  spw: str):
        return f'{self.bcal_dir}/{spw}_concat.bcal'

    def get_ms_path(self, timestamp: datetime, spw: str):
        """
        ms path should looks like /path/to/msfile/2018-03-02/hh=02/2018-03-02T02:02:02/00_2018-03-02T02:02:02.ms
        :param timestamp:
        :param spw:
        :return:
        """
        date = timestamp.date().isoformat()
        hour = f'{timestamp.hour:02d}'
        return f'{self.msfile_dir}/{date}/hh={hour}/{timestamp.isoformat()}/{spw}_{timestamp.isoformat()}.ms'

    def get_flag_npy_path(self, timestamp):
        """
        Returns the same flag npy file regardless of the timestamp...
        :param timestamp:
        :return:
        """
        return self.flag_npy_path
