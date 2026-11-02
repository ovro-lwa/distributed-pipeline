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


# ---- smoothing switches (docs/peel_config.md) --------------------------------------------

SMOOTH_KEYS = ("ZEST_SMOOTH_BELOW_MHZ", "ZEST_SMOOTH_EXEMPT_FLUX", "ZEST_RFI_SMOOTH_FLUX",
               "ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG")


def test_smoothing_switches_unset_by_default():
    # the default environment is exactly the one before the switches existed
    for p in (PeelParams(), PeelParams(engine="sky_rfi", smooth_flux=1000.0)):
        env = p.daemon_env(rfi_sources=None)
        assert not any(k in env for k in SMOOTH_KEYS)
    assert PeelParams(smooth_flux=1000.0).daemon_env(None)["ZEST_SMOOTH_FLUX"] == "1000.0"


def test_smoothing_switches_env_mapping():
    env = PeelParams(engine="sky_rfi", smooth_flux=1000.0, smooth_below_mhz=65.0,
                     smooth_exempt_flux=500.0, rfi_smooth_flux=0.0).daemon_env(None)
    assert env["ZEST_SMOOTH_FLUX"] == "1000.0"
    assert env["ZEST_SMOOTH_BELOW_MHZ"] == "65.0"
    assert env["ZEST_SMOOTH_EXEMPT_FLUX"] == "500.0"
    assert env["ZEST_RFI_SMOOTH_FLUX"] == "0.0"          # 0 is a value, not "unset"
    only_gate = PeelParams(smooth_below_mhz=65).daemon_env(None)
    assert only_gate["ZEST_SMOOTH_BELOW_MHZ"] == "65"
    assert "ZEST_SMOOTH_EXEMPT_FLUX" not in only_gate and "ZEST_RFI_SMOOTH_FLUX" not in only_gate
    el = PeelParams(smooth_exempt_below_el_deg=30.0).daemon_env(None)
    assert el["ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG"] == "30.0"


@pytest.mark.parametrize("field,value", [
    ("smooth_flux", -1.0), ("smooth_below_mhz", -65.0), ("smooth_exempt_flux", -500.0),
    ("rfi_smooth_flux", -1.0), ("smooth_exempt_flux", float("nan")), ("smooth_below_mhz", "65"),
    ("smooth_flux", True), ("smooth_exempt_below_el_deg", -30.0),
])
def test_smoothing_switches_must_be_non_negative_numbers(field, value):
    with pytest.raises(ConfigError, match=field):
        PeelParams(**{field: value})


def test_yaml_smoothing_gate_and_band_override(tmp_path):
    for name in ("cluster.yaml", "subbands.yaml"):
        (tmp_path / name).write_text((CONFIG / name).read_text())
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    d["peel"].update(CPU_MATCHED, device="cpu", engine="sky_rfi", smooth_flux=1000.0,
                     smooth_exempt_flux=500.0, smooth_below_mhz=65.0)
    d["peel_overrides"] = {82: {"smooth_flux": 0.0}}
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(d))
    cfg = Config.load(tmp_path)
    low = cfg.pipeline.peel_for_band(23).daemon_env(None)
    high = cfg.pipeline.peel_for_band(82).daemon_env(None)
    assert low["ZEST_SMOOTH_FLUX"] == "1000.0" and low["ZEST_SMOOTH_BELOW_MHZ"] == "65.0"
    assert low["ZEST_SMOOTH_EXEMPT_FLUX"] == "500.0"
    assert high["ZEST_SMOOTH_FLUX"] == "0.0" and high["ZEST_SMOOTH_EXEMPT_FLUX"] == "500.0"
    d["peel"]["smooth_below_mhz"] = -1
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="smooth_below_mhz"):
        Config.load(tmp_path)


def test_workspace_switch_maps_to_env_only_when_on():
    from gpu_subband_imaging.config import ConfigError, PeelParams
    assert "ZEST_WORKSPACE" not in PeelParams(engine="sky_rfi").daemon_env(None)
    assert PeelParams(engine="sky_rfi", workspace=True).daemon_env(None)["ZEST_WORKSPACE"] == "True"
    import pytest
    with pytest.raises(ConfigError):
        PeelParams(workspace=True)


# ---- defaults = 13405ce (review M2) ------------------------------------------------------------

# PeelParams fields of 13405ce, in order, with their defaults: vars() of the 13405ce PeelParams()
PEEL_13405CE = dict(
    maxiter=30, tolerance=1e-2, minuvw=10.0, peeliter=3, min_source_elevation_deg=15.0,
    require_convergence=True, max_gain_amplitude=100.0, quality_sample_rows=1024,
    min_visibility_amplitude=1e-12, min_visibility_nonzero_fraction=0.01, solver="damped",
    model="kernels", uvw="ms", flag_model=False, resubtract=False, guard=False, rfi=False,
    rfi_mode="after", fast_io=False, device="gpu", reg_max_offset_deg=0.0, reg_taper_deg=1.0,
    reg_lambda=1.0, flag_zero=False, residual_guard=False, backstop=False)


def _env_13405ce(fields, rfi_sources):
    """13405ce PeelParams.daemon_env, verbatim, on a dict of its fields."""
    opts = {k: v for k, v in fields.items() if k != "rfi"}
    if fields["rfi"]:
        opts["rfi_sources"] = rfi_sources
    return {f"ZEST_{k.upper()}": str(v) for k, v in opts.items()}


PRODUCTION = dict(maxiter=5, tolerance=1e-4, minuvw=5.0, peeliter=3, min_source_elevation_deg=0.0,
                  require_convergence=False, max_gain_amplitude=100.0, quality_sample_rows=1024,
                  min_visibility_amplitude=1e-12, min_visibility_nonzero_fraction=0.01, solver="rk4",
                  model="exact", uvw="geometric", flag_model=True, resubtract=True, fast_io=True,
                  rfi=True, flag_zero=True, residual_guard=True)       # the nightly template's peel block


@pytest.mark.parametrize("settings", [{}, PRODUCTION, CPU_MATCHED])
def test_default_env_and_provenance_equal_13405ce(settings):
    p = PeelParams(**settings)
    fields = dict(PEEL_13405CE, **settings)
    rfi = "/cal/rfi.json" if fields["rfi"] else None
    assert p.daemon_env(rfi) == _env_13405ce(fields, rfi)
    assert list(p.daemon_env(rfi)) == list(_env_13405ce(fields, rfi))      # same order too
    assert p.as_dict() == fields and list(p.as_dict()) == list(fields)


def test_new_fields_appear_once_one_is_set():
    env = PeelParams(**PRODUCTION, engine="sky_rfi", precision="mixed").daemon_env("/r.json")
    assert env["ZEST_ENGINE"] == "sky_rfi" and env["ZEST_PRECISION"] == "mixed"
    assert env["ZEST_RESTORE_BELOW_MHZ"] == "41.0" and env["ZEST_WARM_START"] == "False"
    d = PeelParams(**PRODUCTION, engine="sky_rfi").as_dict()
    assert set(PeelParams.NEW_DEFAULTS) <= set(d)
    # a restore list keeps its 41 MHz gate (the daemon's own default would be Inf)
    assert PeelParams(restore="/j.json").daemon_env(None)["ZEST_RESTORE_BELOW_MHZ"] == "41.0"


def test_fast_kernels_switch_maps_to_env_only_when_on():
    assert "ZEST_FAST_KERNELS" not in PeelParams(**PRODUCTION).daemon_env("/r.json")
    assert "ZEST_FAST_KERNELS" not in PeelParams(**PRODUCTION, engine="sky_rfi").daemon_env("/r.json")
    env = PeelParams(**PRODUCTION, fast_kernels=True).daemon_env("/r.json")
    assert env["ZEST_FAST_KERNELS"] == "True" and env["ZEST_ENGINE"] == "daemon"
    assert PeelParams(**PRODUCTION, fast_kernels=True).as_dict()["fast_kernels"] is True
