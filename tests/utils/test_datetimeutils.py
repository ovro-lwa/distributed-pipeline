import pytest
from datetime import datetime as dt
from orca.utils import datetimeutils


@pytest.mark.parametrize('pivot, datetime_list, expected_datetime', [
    (dt(2008, 1, 1, 13), [dt(2008, 1, 1, 12), dt(2008, 1, 2, 0), dt(2008, 1, 1, 16)], dt(2008, 1, 1, 12)),
    (dt(2008, 1, 1, 13), [dt(2008, 1, 4, 0), dt(2007, 12, 28, 0)], dt(2008, 1, 4, 0)),
    (dt(2008, 1, 1, 13), {dt(2008, 1, 4, 0): 'a', dt(2007, 12, 28, 0): 'b'}.keys(), dt(2008, 1, 4, 0))
])
def test_find_closest(pivot, datetime_list, expected_datetime):
    assert expected_datetime == datetimeutils.find_closest(pivot, datetime_list)
