from pathlib import Path
import pytest
from mock import patch

from datetime import datetime

from orca.metadata.stageiii import _get_ms_list

def test_get_ms_list():
    p = Path('.')
    msl = [p / '2021-04-02/12/20240402_121501_82MHz.ms',
                                p / '2024-04-02/12/20240402_121511_82MHz.ms',
                                p / '2024-04-02/12/20240402_121521_82MHz.ms']

    with patch.object(Path, 'glob', return_value = msl) as path_mock:
        prefix = Path('.')
        assert _get_ms_list(prefix, datetime(2024, 4, 2, 12, 15, 0), datetime(2024, 4, 2, 12, 15, 20)) == msl[:2]