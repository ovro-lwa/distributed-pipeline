import json
from dataclasses import replace
from pathlib import Path

from gpu_subband_imaging.config import Config
from gpu_subband_imaging.metadata import write_run_metadata


CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"


def test_run_metadata_records_provenance_and_final_status(tmp_path, monkeypatch):
    cfg = Config.load(CONFIG)
    cfg = replace(
        cfg,
        pipeline=replace(cfg.pipeline, output_root=str(tmp_path)),
    )
    monkeypatch.setenv("GSI_CODE_VERSION", "abc123")

    path = write_run_metadata(cfg, str(CONFIG), "running", {"running": 2})
    started = json.loads(path.read_text())["started_utc"]
    write_run_metadata(cfg, str(CONFIG), "completed", {"done": 2})
    data = json.loads(path.read_text())

    assert path == tmp_path / "_metadata" / cfg.pipeline.date / "run.json"
    assert data["status"] == "completed"
    assert data["started_utc"] == started
    assert data["finished_utc"]
    assert data["code"]["revision"] == "abc123"
    assert data["pipeline"]["cal"]["bp"] == cfg.pipeline.cal.bp
    assert data["pipeline"]["peel"]["maxiter"] == 30
    assert data["ledger"] == {"done": 2}
    assert set(data["config_sha256"]) == {
        "cluster.yaml", "subbands.yaml", "pipeline.yaml"}
