from pathlib import Path

import pytest

from gpu_subband_imaging.config import Config, ConfigError, load_subbands

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"


def test_loads_shipped_config():
    cfg = Config.load(CONFIG)
    assert cfg.pipeline.batch_size > 0
    assert cfg.cluster.slots(), "expected at least one enabled slot"
    assert all(not (s.node == "gpu-node-02" and s.gpu == 1)
               for s in cfg.cluster.slots())
    assert {s.gpu for s in cfg.cluster.slots()
            if s.node == "gpu-node-01"} == {0, 1}


def test_subband_geometry_tiers():
    sb = load_subbands(CONFIG / "subbands.yaml")
    assert sb.geom(36).pixels == 1507
    assert sb.geom(41).pixels == 2357
    assert sb.geom(73).pixels == 3122
    # FoV npix*scale ~ constant across tiers
    fovs = [sb.geom(b).pixels * sb.geom(b).scale for b in (36, 41, 73)]
    assert max(fovs) - min(fovs) < 3.0


def test_movie_toggle_defaults_true():
    cfg = Config.load(CONFIG)
    assert cfg.pipeline.movie.enabled is True


def test_all_bands_use_safe_peel_defaults():
    cfg = Config.load(CONFIG)

    assert cfg.pipeline.peel_for_band(23).maxiter == 30
    assert cfg.pipeline.peel_for_band(32).maxiter == 30
    assert cfg.pipeline.peel_for_band(23).peeliter == 3
    assert cfg.pipeline.peel_for_band(36) == cfg.pipeline.peel
    assert cfg.pipeline.peel_for_band(36).maxiter == 30
    assert cfg.pipeline.peel.tolerance == 1e-2
    assert cfg.pipeline.peel.min_source_elevation_deg == 15.0
    assert cfg.pipeline.peel.require_convergence is True
    assert cfg.pipeline.peel.max_gain_amplitude == 100.0
    assert cfg.pipeline.peel.min_visibility_nonzero_fraction == 0.01


def test_unknown_band_raises():
    sb = load_subbands(CONFIG / "subbands.yaml")
    with pytest.raises(ConfigError):
        sb.geom(13)
