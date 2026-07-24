"""Date and time utilities for OVRO-LWA data processing.

Provides timestamp manipulation functions and defines the standard
integration time for Stage III data products.
"""
from datetime import datetime, timedelta
from typing import Iterable

#: Standard integration time for Stage III measurement sets.
STAGE_III_INTEGRATION_TIME = timedelta(seconds=10.031)


def measurement_set_seconds_to_mjd(time_seconds: float) -> float:
    """Convert a Measurement Set time value to MJD days.

    Measurement Set ``TIME`` and ``TIME_RANGE`` columns store seconds since
    MJD 0, while Astropy's ``mjd`` format expects a value in days.
    """
    return float(time_seconds) / 86400.0


def find_closest(pivot: datetime, date_time_list: Iterable) -> datetime:
    """Find closest datetime to pivot in the list of date time.

    Args:
        pivot: The datetime object to find the closest one to.
        date_time_list: The list of datetime objects.

    Returns: The datetime in the list that's closest to pivot.

    """
    return min(date_time_list, key=lambda x: abs(x - pivot))
