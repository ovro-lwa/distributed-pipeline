from pathlib import Path
import pytest
from mock import patch

from datetime import datetime, date
from os import path

from orca.metadata.stageiii import _get_ms_list
from orca.metadata.stageiii import StageIIIPathsManager

DATA_DIR = '/lustre/pipeline/night-time/'

def test_get_ms_list():
    p = Path('.')
    msl = [p / '2021-04-02/12/20240402_121501_82MHz.ms',
                                p / '2024-04-02/12/20240402_121511_82MHz.ms',
                                p / '2024-04-02/12/20240402_121521_82MHz.ms']

    with patch.object(Path, 'glob', return_value = msl) as path_mock:
        prefix = Path('.')
        assert _get_ms_list(prefix, datetime(2024, 4, 2, 12, 15, 0), datetime(2024, 4, 2, 12, 15, 20)) == msl[:2]

def test_get_ms_list_through_method():
    p = Path('.')
    msl = [p / '2021-04-02/12/20240402_121501_82MHz.ms',
                                p / '2024-04-02/12/20240402_121511_82MHz.ms',
                                p / '2024-04-02/12/20240402_121521_82MHz.ms']

    with patch.object(Path, 'glob', return_value = msl) as path_mock:
        pm = StageIIIPathsManager('.', '.', '82MHz', datetime(2024, 4, 2, 12, 15, 0), datetime(2024, 4, 2, 12, 15, 20))
        ans = pm.ms_list
        assert ans == [(datetime(2024, 4, 2, 12, 15, 1), msl[0].absolute().as_posix()), (datetime(2024, 4, 2, 12, 15, 11), msl[1].absolute().as_posix())]

def test_get_ms_list_multi_hour():
    p = Path('.')
    msl = [[p / '2021-04-02/12/20240402_121501_82MHz.ms',
                                p / '2024-04-02/12/20240402_121511_82MHz.ms',
                                p / '2024-04-02/12/20240402_121521_82MHz.ms',],
                                [p / '2024-04-02/13/20240402_131501_82MHz.ms',
                                p / '2024-04-02/13/20240402_131511_82MHz.ms',
                                p / '2024-04-02/13/20240402_131521_82MHz.ms']]

    with patch.object(Path, 'glob', side_effect = msl) as path_mock:
        prefix = Path('.')
        assert _get_ms_list(prefix, datetime(2024, 4, 2, 12, 15, 0), datetime(2024, 4, 2, 13, 15, 20)) == [m for l in msl for m in l][:5]

@pytest.mark.skipif(not path.isdir(DATA_DIR), reason="need acual data.")
def test_get_ms_list_real():
    prefix = Path(DATA_DIR) / '13MHz'
    res = _get_ms_list(prefix, datetime(2024, 4, 2, 8, 15, 0), datetime(2024, 4, 2, 8, 15, 20))
    assert len(res) == 2
    for r in res:
        assert path.isdir(r.resolve())

@pytest.mark.skipif(not path.isdir(DATA_DIR), reason="need acual data.")
def test_get_ms_list_multi_day_real():
    prefix = Path(DATA_DIR) / '13MHz'
    res = _get_ms_list(prefix, datetime(2024, 4, 2, 3, 10, 0), datetime(2024, 4, 2, 11, 10, 0))
    assert path.isdir(res[0].resolve())
    assert res[0].name == '20240402_031000_13MHz.ms'
    assert res[-1].name == '20240402_110959_13MHz.ms'
    assert path.isdir(res[-1].resolve())
    assert len(res) == 359*8

def test_get_bcal_path():
    pm = StageIIIPathsManager('/lustre/some/data', '/lustre/some/workdir', '13MHz', datetime(2024, 4, 2, 12, 15, 0), datetime(2024, 4, 2, 12, 15, 20))
    assert pm.get_bcal_path(date(2024, 4, 2)) == '/lustre/some/workdir/13MHz/20240402.bcal'
