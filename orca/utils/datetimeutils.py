"""Utilities for timestamps.
"""
from datetime import datetime
from typing import Iterable


def find_closest(pivot: datetime, date_time_list: Iterable) -> datetime:
    """Find closest datetime to pivot in the list of date time.

    Args:
        pivot: The datetime object to find the closest one to.
        date_time_list: The list of datetime objects.

    Returns: The datetime in the list that's closest to pivot.

    """
    return min(date_time_list, key=lambda x: abs(x - pivot))
