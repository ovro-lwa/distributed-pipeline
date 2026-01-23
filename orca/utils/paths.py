"""Path construction and resource location utilities.

Provides functions for locating package resources such as AOFlagger
strategy files.
"""
from importlib.resources import path as resource_path


def get_aoflagger_strategy(name: str) -> str:
    """Return the full path to an AOFlagger strategy file.

    Locates strategy files bundled with the orca package.

    Args:
        name: Filename of the strategy file (e.g., 'lwa-default.lua').

    Returns:
        Absolute path to the strategy file.
    """
    with resource_path("orca.resources.aoflagger_strategies", name) as p:
        return str(p)
