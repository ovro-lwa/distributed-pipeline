"""Validation utilities.
"""
from typing import Collection


def check_collection_not_empty(l: Collection):
    if len(l) == 0:
        raise ValueError('Collection cannot be empty.')