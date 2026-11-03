"""Scaife & Heald calibrators are never peeled (config.check_protected, at Config.load)."""
import json

import pytest

from gpu_subband_imaging.config import ConfigError, check_protected, model_source_names


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
    with pytest.raises(ConfigError, match="3C 380"):
        check_protected(_model(tmp_path, {"sources": [{"name": "x", "components": [
            {"ra": "18h29m31.8s", "dec": "+48d44m46s"}]}]}))
    check_protected(None)
    check_protected(str(tmp_path / "missing.json"))


def test_config_load_refuses_a_protected_model(tmp_path, monkeypatch):
    import shutil
    import yaml
    from pathlib import Path
    from gpu_subband_imaging.config import Config
    repo = Path(__file__).resolve().parents[1]
    c = tmp_path / "cfg"
    shutil.copytree(repo / "examples" / "config", c)
    d = yaml.safe_load((c / "pipeline.yaml").read_text())
    d["cal"]["sources"] = _model(tmp_path, [{"name": "3C 286", "ra": "13h31m08.3s", "dec": "+30d30m33s"}])
    (c / "pipeline.yaml").write_text(yaml.safe_dump(d))
    monkeypatch.delenv("GSI_ALLOWED_LUSTRE_WRITE_ROOT", raising=False)
    with pytest.raises(ConfigError, match="Scaife"):
        Config.load(c)
