import types
from pathlib import Path

from gpu_subband_imaging import cli


def test_freeze_config_dir_copies_yaml_files_to_run_state(tmp_path, monkeypatch):
    config_dir = tmp_path / "config-allbands"
    config_dir.mkdir()
    for name in ("cluster.yaml", "subbands.yaml", "pipeline.yaml"):
        (config_dir / name).write_text(f"{name}: original\n")

    state_dir = tmp_path / "_state"
    cfg = types.SimpleNamespace(
        cluster=types.SimpleNamespace(state_dir=str(state_dir)),
        pipeline=types.SimpleNamespace(date="2026-07-09"),
    )
    monkeypatch.setattr(cli.os, "getpid", lambda: 1234)
    monkeypatch.setattr(cli.time, "strftime", lambda fmt, t: "20260709T184240")

    frozen = Path(cli._freeze_config_dir(str(config_dir), cfg))

    assert frozen == state_dir / "run_configs" / "2026-07-09-20260709T184240-1234"
    assert (frozen / "cluster.yaml").read_text() == "cluster.yaml: original\n"
    assert (frozen / "subbands.yaml").read_text() == "subbands.yaml: original\n"
    assert (frozen / "pipeline.yaml").read_text() == "pipeline.yaml: original\n"
