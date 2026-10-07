"""Sun-altitude frame cut (no CASA / Celery needed)."""
import warnings

import pytest

from orca.utils.sun_cut import filter_sun, frame_time


def _ms(hhmmss, date='20250421'):
    return f'/lustre/pipeline/night-time/averaged/50MHz/2025-04-21/{hhmmss[:2]}/{date}_{hhmmss}_50MHz_averaged.ms'


@pytest.fixture(autouse=True)
def _quiet_iers():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def test_frame_time_from_archive_name():
    assert frame_time(_ms('120004')).isoformat() == '2025-04-21T12:00:04'
    assert frame_time(_ms('120004') + '.tar').isoformat() == '2025-04-21T12:00:04'
    with pytest.raises(ValueError):
        frame_time('/no/timestamp.ms')


def test_2025_04_21_evening_and_morning_limits():
    # OVRO 2025-04-21 (UTC): Sun sets through -18 deg at ~04:07:20 and rises
    # through -12 deg at ~12:10:10.
    files = [_ms(t) for t in ('035900', '040500', '041000', '120500', '120900',
                               '121100', '130000')]
    kept, refused = filter_sun(files)
    assert [frame_time(f).strftime('%H%M%S') for f in kept] == ['041000', '120500', '120900']
    assert [frame_time(f).strftime('%H%M%S') for f in refused] == ['035900', '040500', '121100', '130000']


def test_evening_limit_is_stricter_than_morning():
    # 04:00 UTC: Sun ~-16.8 deg while setting -> refused at -18, kept at -12.
    kept, _ = filter_sun([_ms('040000')])
    assert kept == []
    kept, _ = filter_sun([_ms('040000')], evening_max_deg=-12)
    assert len(kept) == 1


def test_empty_input():
    assert filter_sun([]) == ([], [])
