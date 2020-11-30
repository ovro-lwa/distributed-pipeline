import pytest
from orca.metadata.pathsmanagers import OfflinePathsManager
from os import path
from datetime import datetime, date

import itertools

def test_read_utc_times():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', '.')
    assert len(pm.utc_times_mapping) == 10
    assert pm.utc_times_mapping[datetime(2019, 10, 28, 23, 4, 44)] == \
           '2019-10-22-02:35:47_0005246685093888.000000.dada'


def test_time_filter():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', '.')
    pm2 = pm.time_filter(datetime(2019, 10, 28, 23, 4, 0), datetime(2019, 10, 28, 23, 5, 0))
    assert len(pm2.utc_times_mapping) == 4
    assert pm2.utc_times_mapping[datetime(2019, 10, 28, 23, 4, 44)] == \
           '2019-10-22-02:35:47_0005246685093888.000000.dada'


def test_time_filter_inclusive_exclusive():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', '.')
    b = datetime(2019, 10, 28, 23, 2, 47)
    e = datetime(2019, 10, 28, 23, 4, 5)
    pm2 = pm.time_filter(b, e)
    assert len(pm2.utc_times_mapping) == 6
    assert b in pm2.utc_times_mapping
    assert e not in pm2.utc_times_mapping

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
           './msfiles/2019-10-28/hh=23/2019-10-28T23:04:44/00_2019-10-28T23:04:44.ms'


def test_get_flag_npy_path_single_npy():
    npy = '/tmp/something.npy'
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             flag_npy_paths=npy)
    assert pm.get_flag_npy_path(date(2018, 3, 2)) == npy


def test_get_data_product_path():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', working_dir='.')
    assert pm.get_data_product_path(datetime(2019, 10, 28, 23, 4, 44), 'images', '_diff.fits', '00') == \
    './images/2019-10-28/hh=23/00_2019-10-28T23:04:44_diff.fits'


def test_get_data_product_path_no_suffix():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt', working_dir='.')
    assert pm.get_data_product_path(datetime(2019, 10, 28, 23, 4, 44), 'images', '', '00') == \
    './images/2019-10-28/hh=23/00_2019-10-28T23:04:44'


def test_get_ms_parent_path():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             working_dir='/tmp')
    assert pm.get_ms_parent_path(datetime(2019, 10, 28, 23, 4, 44)) == \
        '/tmp/msfiles/2019-10-28/hh=23/2019-10-28T23:04:44'


def test_get_flag_npy_flag_multiple_npy():
    npy = {date(2018, 3, 2): '/tmp/something.npy',
           date(2018, 3, 3): '/tmp/something_else.npy'
           }
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt',
                             flag_npy_paths=npy)
    assert pm.get_flag_npy_path(date(2018, 3, 2)) == '/tmp/something.npy'


def test_chunks_by_integration():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt')
    for n in range(1, len(pm.utc_times_mapping.keys())):
        chunks = pm.chunks_by_integration(n)
        for c in chunks[:-1]:
            assert len(c) == n
        assert len(chunks[-1]) <= n
        for ans, expected in zip(itertools.chain.from_iterable(chunks), list(pm.utc_times_mapping.keys())):
            assert ans == expected


def test_chunks_by_integration_multiple():
    pm = OfflinePathsManager(f'{path.dirname(__file__)}/../resources/utc_times_test.txt')
    chunks = pm.chunks_by_integration(2)
    for l in chunks:
        print(l)
