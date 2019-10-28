"""
Executable paths etc.
"""
from abc import ABC, abstractmethod


class ConfigManager(ABC):
    # this should just be a bunch of properties
    def __init__(self):
        pass


class DefaultConfigManager(ConfigManager):
    def __init__(self):
        super().__init__()
