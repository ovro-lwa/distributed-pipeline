import pytest
from orca.metadata.pathsmanagers import OfflinePathsManager
from os import path
from datetime import datetime, date


def test_read_utc_times():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', '.')
    assert len(pm.utc_times_mapping) == 10
    assert pm.utc_times_mapping[datetime(2019, 10, 28, 23, 4, 44)] == \
           '2019-10-22-02:35:47_0005246685093888.000000.dada'


def test_with_bad_path():
    with pytest.raises(FileNotFoundError):
        pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                                 '/some/bad/path', '.', '.')


def test_get_bcal_path():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             '.', '.', '.')
    assert pm.get_bcal_path(date(2019, 10, 28), '00') == './2019-10-28/00.bcal'


def test_get_ms_file():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             '.', '.', '.')
    assert pm.get_ms_path(datetime(2019, 10, 28, 23, 4, 44), '00') == \
           './2019-10-28/hh=23/2019-10-28T23:04:44/00_2019-10-28T23:04:44.ms'


def test_get_flag_npy_path_single_npy():
    npy = '/tmp/something.npy'
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             flag_npy_paths=npy)
    assert pm.get_flag_npy_path(date(2018, 3, 2)) == npy


def test_get_flag_npy_flag_multiple_npy():
    npy = {date(2018, 3, 2): '/tmp/something.npy',
           date(2018, 3, 3): '/tmp/something_else.npy'
           }
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             flag_npy_paths=npy)
    assert pm.get_flag_npy_path(date(2018, 3, 2)) == '/tmp/something.npy'
