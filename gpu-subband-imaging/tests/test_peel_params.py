from pathlib import Path

import pytest
import yaml

from gpu_subband_imaging.config import Config, ConfigError, PeelParams

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"

# The ZEST_* variables the original worker passed (vars(PeelParams) before the
# CPU-matched options existed); their values must not change by default.
ORIGINAL_ENV = {
    "ZEST_MAXITER": "30", "ZEST_TOLERANCE": "0.01", "ZEST_MINUVW": "10.0",
    "ZEST_PEELITER": "3", "ZEST_MIN_SOURCE_ELEVATION_DEG": "15.0",
    "ZEST_REQUIRE_CONVERGENCE": "True", "ZEST_MAX_GAIN_AMPLITUDE": "100.0",
    "ZEST_QUALITY_SAMPLE_ROWS": "1024", "ZEST_MIN_VISIBILITY_AMPLITUDE": "1e-12",
    "ZEST_MIN_VISIBILITY_NONZERO_FRACTION": "0.01",
}

CPU_MATCHED = dict(solver="rk4", maxiter=5, tolerance=1e-4, minuvw=5.0,
                   min_source_elevation_deg=0.0, require_convergence=False,
                   model="exact", uvw="geometric", flag_model=True, resubtract=True)


def test_defaults_keep_the_original_daemon_settings():
    env = PeelParams().daemon_env(rfi_sources=None)
    for key, value in ORIGINAL_ENV.items():
        assert env[key] == value
    assert env["ZEST_SOLVER"] == "damped"
    assert env["ZEST_MODEL"] == "kernels"
    assert env["ZEST_UVW"] == "ms"
    assert env["ZEST_DEVICE"] == "gpu"
    assert env["ZEST_FLAG_MODEL"] == "False"
    assert env["ZEST_FAST_IO"] == "False"
    assert env["ZEST_REG_MAX_OFFSET_DEG"] == "0.0"
    assert "ZEST_RFI_SOURCES" not in env and "ZEST_RFI" not in env


def test_cpu_matched_settings():
    env = PeelParams(**CPU_MATCHED).daemon_env(rfi_sources=None)
    assert env["ZEST_SOLVER"] == "rk4"
    assert env["ZEST_MAXITER"] == "5"
    assert env["ZEST_TOLERANCE"] == "0.0001"
    assert env["ZEST_REQUIRE_CONVERGENCE"] == "False"
    assert env["ZEST_MODEL"] == "exact"
    assert env["ZEST_UVW"] == "geometric"


def test_rfi_needs_a_model_path():
    p = PeelParams(rfi=True)
    with pytest.raises(ConfigError, match="rfi_sources"):
        p.daemon_env(rfi_sources=None)
    env = p.daemon_env(rfi_sources="/cal/rfi.json")
    assert env["ZEST_RFI_SOURCES"] == "/cal/rfi.json"
    assert env["ZEST_RFI_MODE"] == "after"


@pytest.mark.parametrize("field,value", [
    ("solver", "newton"), ("model", "fancy"), ("uvw", "lmn"),
    ("rfi_mode", "before"), ("device", "tpu"),
])
def test_invalid_choices_raise(field, value):
    with pytest.raises(ConfigError, match=field):
        PeelParams(**{field: value})


def test_cpu_device_needs_the_exact_model():
    with pytest.raises(ConfigError, match="device=cpu"):
        PeelParams(device="cpu")
    PeelParams(device="cpu", model="exact")


def test_iteration_counts_must_be_positive():
    with pytest.raises(ConfigError):
        PeelParams(maxiter=0)
    with pytest.raises(ConfigError):
        PeelParams(peeliter=0)


def test_yaml_rfi_on_low_bands_only(tmp_path):
    # Copy the shipped config and enable the RFI zest only where the RFI model has flux.
    for name in ("cluster.yaml", "subbands.yaml"):
        (tmp_path / name).write_text((CONFIG / name).read_text())
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    d["cal"]["rfi_sources"] = "/cal/rfi.json"
    d["peel"].update(CPU_MATCHED, rfi=True)
    d["peel_overrides"] = {b: {"rfi": False} for b in (46, 50, 55, 59, 64, 69, 73, 78, 82)}
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(d))
    cfg = Config.load(tmp_path)
    assert cfg.pipeline.cal.rfi_sources == "/cal/rfi.json"
    low = cfg.pipeline.peel_for_band(23).daemon_env(cfg.pipeline.cal.rfi_sources)
    high = cfg.pipeline.peel_for_band(73).daemon_env(cfg.pipeline.cal.rfi_sources)
    assert low["ZEST_RFI_SOURCES"] == "/cal/rfi.json"
    assert "ZEST_RFI_SOURCES" not in high
    assert high["ZEST_SOLVER"] == "rk4"           # overrides keep the other settings


def test_shipped_config_has_no_rfi_model():
    cfg = Config.load(CONFIG)
    assert cfg.pipeline.cal.rfi_sources is None
    assert cfg.pipeline.peel.rfi is False
