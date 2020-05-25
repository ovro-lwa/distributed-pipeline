import pytest
from os import path
from orca.configmanager import queue_config, load_yaml


def test_import_config():
    assert 'prefix' in queue_config


def test_parse_yml():
    load_yaml(path.join(path.dirname(__file__), 'resources/test_config.yml'))
