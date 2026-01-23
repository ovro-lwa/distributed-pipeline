"""Input validation utilities.

Provides common validation functions for checking inputs to
pipeline functions.
"""
from typing import Collection


def check_collection_not_empty(l: Collection):
    """Validate that a collection is not empty.

    Args:
        l: Any collection (list, tuple, set, etc.).

    Raises:
        ValueError: If the collection has length 0.
    """
    if len(l) == 0:
        raise ValueError('Collection cannot be empty.')