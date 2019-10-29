import pytest
from orca.metadata.pathsmanagers import OfflinePathsManager
from os import path
from datetime import datetime


def test_read_utc_times():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             '.', '.', '.')
    assert len(pm.utc_times_mapping) == 10
    assert pm.utc_times_mapping[datetime(2019, 10, 28, 23, 4, 44)] == \
           '2019-10-22-02:35:47_0005246685093888.000000.dada'


def test_with_bad_path():
    with pytest.raises(FileNotFoundError):
        pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                                 '/some/bad/path', '.', '.')


def test_get_gaintable_path():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             '.', '.', '.')
    assert pm.get_gaintable_path('00') == './00.bcal'
