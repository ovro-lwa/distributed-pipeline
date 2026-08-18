import pytest

from gpu_subband_imaging.config import (
    ClusterConfig,
    ConfigError,
    PipelineConfig,
    CalTables,
    RuntimeEnv,
    PeelParams,
    ImagingParams,
    MovieParams,
    _validate_lustre_write_roots,
)


def _cluster(state_dir="/lustre/pipeline/snapshots_gpu_pipeline/_state",
             work_root="/fast/pipeline/gpu_peel"):
    return ClusterConfig(
        state_dir=state_dir,
        max_feeders=1,
        wsclean_threads=1,
        aoflagger_threads=1,
        free_core_budget=1,
        work_root=work_root,
        nodes=[],
    )


def _pipeline(output_root="/lustre/pipeline/snapshots_gpu_pipeline"):
    return PipelineConfig(
        date="2026-07-08",
        hours=["06"],
        bands=[27],
        batch_size=1,
        batch_timeout_seconds=1,
        max_retries=0,
        data_root="/lustre/pipeline/slow",
        output_root=output_root,
        chanbin=4,
        cal=CalTables(bp="/lustre/pipeline/calibration/bp",
                      xy="/lustre/calibration/xy",
                      sources="/lustre/calibration/sources.json",
                      aoflagger_strategy="/lustre/calibration/strategy.lua"),
        peel=PeelParams(),
        peel_overrides={},
        imaging=ImagingParams(),
        movie=MovieParams(),
        do_aoflag=False,
        do_badants=True,
        env=RuntimeEnv(conda_env="conda", dev_env="dev", julia_bin="julia",
                       ttcalx="ttcalx", shared_depot="depot"),
    )


def test_allows_snapshot_output_tree_and_readonly_lustre_inputs():
    _validate_lustre_write_roots(_cluster(), _pipeline())


def test_allows_configured_lustre_output_tree(monkeypatch):
    root = "/lustre/public/gsi"
    monkeypatch.setenv("GSI_ALLOWED_LUSTRE_WRITE_ROOT", root)
    _validate_lustre_write_roots(
        _cluster(state_dir=f"{root}/_state"),
        _pipeline(output_root=root),
    )


def test_rejects_lustre_output_root_outside_snapshot_tree():
    with pytest.raises(ConfigError, match="pipeline.output_root"):
        _validate_lustre_write_roots(
            _cluster(),
            _pipeline(output_root="/lustre/pipeline/images_test"),
        )


def test_rejects_lustre_state_dir_outside_snapshot_tree():
    with pytest.raises(ConfigError, match="cluster.state_dir"):
        _validate_lustre_write_roots(
            _cluster(state_dir="/lustre/pipeline/images_test/_state"),
            _pipeline(),
        )


def test_rejects_lustre_work_root_outside_snapshot_tree():
    with pytest.raises(ConfigError, match="cluster.work_root"):
        _validate_lustre_write_roots(
            _cluster(work_root="/lustre/tmp/gpu_peel"),
            _pipeline(),
        )
