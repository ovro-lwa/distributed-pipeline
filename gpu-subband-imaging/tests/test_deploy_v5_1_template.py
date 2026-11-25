"""deploy/config-lst-v5.1-template (the gsi-v5.1 nightly template): the v5 template plus exactly the v5.1 keys, the
config checks pass, and its paths are the gsi-v5.1 deploy's (gsi code, work_root) with the gsi-v5 deploy's TTCalX
v0.5.0 tree, imager venv and CuPy cache."""
from pathlib import Path

import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import Config, v3_params

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
V5 = DEPLOY / "config-lst-v5-template"
V51 = DEPLOY / "config-lst-v5.1-template"

# (file, dotted key) -> v5.1 value; every other value is the v5 template's
V51_KEYS = {
    ("pipeline.yaml", "v3.prep_block_rows"): 2048,
    ("pipeline.yaml", "v3.zest_persistent"): True,
    ("pipeline.yaml", "v3.aoflag_async_io"): True,
    ("pipeline.yaml", "v3.prefetch"): True,
    ("pipeline.yaml", "v3.lookahead"): 16,
    ("cluster.yaml", "defaults.max_feeders"): 16,
    ("cluster.yaml", "defaults.work_root"): "/fast/pipeline/gsi_v5_1",
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def test_v5_1_template_is_v5_plus_the_v5_1_keys():
    assert sorted(p.name for p in V51.iterdir()) == sorted(p.name for p in V5.iterdir())
    for f in sorted(p.name for p in V5.glob("*.yaml")):
        a = _flat(yaml.safe_load((V5 / f).read_text()))
        b = _flat(yaml.safe_load((V51 / f).read_text()))
        want = {k: v for (ff, k), v in V51_KEYS.items() if ff == f}
        assert {k: b[k] for k in b if a.get(k, object()) != b[k]} == want, f
        assert not [k for k in a if k not in b], f                      # nothing dropped


def test_v5_1_template_loads_with_the_v5_1_switches(monkeypatch, caplog):
    monkeypatch.setattr(config_mod, "check_protected", lambda path: None)     # model files live on calim
    cfg = Config.load(V51)
    pl, cl = cfg.pipeline, cfg.cluster
    v3 = v3_params(pl)
    assert (v3.prep_block_rows, v3.zest_persistent, v3.aoflag_async_io, v3.prefetch, v3.lookahead) == \
        (2048, True, True, True, 16)
    assert (cl.max_feeders, cl.zest_recycle_files, v3.prep_procs, v3.prep_gain_cache) == (16, 0, 6, True)
    # the defaults of the options the template leaves out
    assert (v3.zest_persistent_idle_s, v3.aoflag_load_threads, pl.fpack_deterministic) == (1800.0, 2, False)
    # where the switches act: zest_persistent with peel_io shm, aoflag_async_io with aoflag hour, prefetch in a
    # streaming LST run with the hour in the batch
    assert v3.peel_io == "shm" and v3.aoflag == "hour" and v3.hour_source == "memory"
    assert pl.stream_hour_images and pl.layout == "lst"
    assert config_mod._warn_noop(pl) == []                               # no switch is a no-op
    assert config_mod._warn_feeders(cl, pl) is None                      # feeders >= prep processes
    # v5's switches are all still on
    p = pl.peel
    assert p.peel_fast and p.slot_gpu_io and pl.imaging.gpu_env["TTCALX_V5"] == "1"
    assert v3.retry_on_checkpoint_node and pl.max_retries == 2 and pl.dispatch_band_order[0] == 78
    # a prefetched batch gets 2 x batch_timeout_seconds; its own budget must still cover the hour stage
    assert pl.batch_timeout_seconds >= pl.hour_imaging.timeout_seconds + 1800 + v3.budget_margin_s


def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_v5_1_template_paths():
    vals = [v for f in sorted(V51.glob("*.yaml")) for v in _strings(yaml.safe_load(f.read_text()))]
    assert not [v for v in vals if "gsi-v4" in v or "gsi_v4" in v]
    d = yaml.safe_load((V51 / "pipeline.yaml").read_text())
    c = yaml.safe_load((V51 / "cluster.yaml").read_text())
    # reused from the gsi-v5 deploy (TTCalX v0.5.0 is unchanged in v5.1): Julia tree, imager venv, CuPy cache
    assert d["env"]["ttcalx"] == "/opt/devel/pipeline/gsi-v5/TTCalX"
    assert d["hour_imaging"]["ttcalx_image"] == "/opt/devel/pipeline/gsi-v5/venv/bin/ttcalx-image"
    assert d["hour_imaging"]["env"]["CUPY_CACHE_DIR"] == "/fast/pipeline/gsi_v5/cupy_cache"
    # its own node work root (never shared with a gsi-v5 run's per-batch directories)
    assert c["defaults"]["work_root"] == "/fast/pipeline/gsi_v5_1"
