from datetime import datetime
from typing import Iterable


def find_closest(pivot: datetime, date_time_list: Iterable) -> datetime:
    return min(date_time_list, key=lambda x: abs(x - pivot))
