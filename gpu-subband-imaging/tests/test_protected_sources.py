"""Scaife & Heald calibrators are never peeled; per-snapshot peel results are summarised."""
import json

import pytest

from gpu_subband_imaging.config import ConfigError, check_protected, model_source_names
from gpu_subband_imaging.hour_lst import peel_summary


def _model(tmp_path, entries):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(entries))
    return str(p)


def test_protected_by_name_and_position(tmp_path):
    ok = _model(tmp_path, [{"name": "Tau A", "ra": "05h34m31.94s", "dec": "+22d00m52.2s"},
                           {"name": "Cyg A", "components": [{"ra": "19h59m28.3s", "dec": "+40d44m02s"}]}])
    check_protected(ok)
    assert model_source_names(ok) == ["Tau A", "Cyg A"]
    with pytest.raises(ConfigError, match="3C48|3C 48"):
        check_protected(_model(tmp_path, [{"name": "3C48", "ra": "01h00m00s", "dec": "+00d00m00s"}]))
    with pytest.raises(ConfigError, match="3C 147"):
        check_protected(_model(tmp_path, [{"name": "src", "ra": "05h42m40s", "dec": "+49d50m00s"}]))
    check_protected(None)
    check_protected(str(tmp_path / "missing.json"))


def test_peel_summary():
    rows = [{"peel": {"sky:Cas A": True, "sky:Tau A": False}}, {"peel": {"sky:Cas A": True}}, {}]
    assert peel_summary(rows) == {"sky:Cas A": {"selected": 2, "subtracted": 2},
                                  "sky:Tau A": {"selected": 1, "subtracted": 0}}
