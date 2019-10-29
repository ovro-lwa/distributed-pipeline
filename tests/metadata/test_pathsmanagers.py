import pytest
from orca.metadata.pathsmanagers import OfflinePathsManager
from os import path
from datetime import datetime


def test_read_utc_times():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             None, None, None)
    assert len(pm._mapping) == 10
    assert pm._mapping[datetime(2019, 10, 28, 23, 4, 44)] == \
           '2019-10-22-02:35:47_0005246685093888.000000.dada'

def test_get_gaintable_path():
    pass
