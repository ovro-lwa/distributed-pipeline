import pytest
from mock import patch

from datetime import datetime
from orca.transform import peeling

import json
import tempfile


def test_get_peeling_sources_dict():
    assert len(peeling._get_peeling_sources_list(datetime(2018, 9, 22, 2, 30, 0), True)) == 3
    assert len(peeling._get_peeling_sources_list(datetime(2018, 9, 22, 16, 30, 0), True)) == 2


def test_write_peeling_sources_json():
    with tempfile.TemporaryDirectory() as dir:
        out_json = f'{dir}/sources.json'
        peeling._write_peeling_sources_json(datetime(2018, 9, 22, 2, 30, 0), out_json, True)
        with open(out_json) as f:
            json.load(f)


@patch('orca.utils.coordutils.is_visible')
def test_write_peeling_sources_json_returns_none_with_no_sources(is_visible):
    is_visible.return_value = False
    with tempfile.TemporaryDirectory() as dir:
        out_json = f'{dir}/sources.json'
        assert peeling._write_peeling_sources_json(datetime(2018, 9, 22, 2, 30, 0), out_json, False) is None
