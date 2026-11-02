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


def test_git_info_reports_commit_and_uncommitted_changes(tmp_path):
    import subprocess
    from gpu_subband_imaging.metadata import git_info
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("a")
    run("add", "f.txt")
    run("commit", "-q", "-m", "x")
    info = git_info(str(tmp_path))
    assert len(info["commit"]) == 40 and info["uncommitted_files"] == [] and info["on_remote"] is False
    (tmp_path / "f.txt").write_text("b")
    assert git_info(str(tmp_path))["uncommitted_files"] == ["f.txt"]
    assert git_info(str(tmp_path / "nope"))["commit"] == "unknown"


def test_run_metadata_pipeline_block_is_13405ce_at_defaults(tmp_path, monkeypatch):
    """With no new switch set, run_*.json's "pipeline" block has 13405ce's keys: no "v3" block and the
    13405ce "peel" keys (as provenance.json); once a new setting is used, all of it is recorded."""
    from dataclasses import fields
    from gpu_subband_imaging.config import PeelParams, PipelineConfig, V3Params
    from test_peel_params import PEEL_13405CE
    cfg = Config.load(CONFIG)
    cfg = replace(cfg, pipeline=replace(cfg.pipeline, output_root=str(tmp_path)))
    monkeypatch.setenv("GSI_CODE_VERSION", "abc123")
    data = json.loads(write_run_metadata(cfg, str(CONFIG), "running").read_text())
    assert set(data["pipeline"]) == {f.name for f in fields(PipelineConfig)} - {"v3"}
    assert list(data["pipeline"]["peel"]) == sorted(PEEL_13405CE)          # sort_keys
    assert data["pipeline"]["peel"] == json.loads(json.dumps(cfg.pipeline.peel.as_dict()))
    peel = PeelParams(**dict(vars(cfg.pipeline.peel), fast_kernels=True))
    cfg2 = replace(cfg, pipeline=replace(cfg.pipeline, peel=peel, v3=V3Params(stage_attempts=2)))
    data = json.loads(write_run_metadata(cfg2, str(CONFIG), "running").read_text())
    assert data["pipeline"]["v3"]["stage_attempts"] == 2
    assert set(PeelParams.NEW_DEFAULTS) <= set(data["pipeline"]["peel"])
    assert data["pipeline"]["peel"]["fast_kernels"] is True
