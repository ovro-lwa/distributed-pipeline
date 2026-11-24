"""deploy/config-lst-v5-template (the gsi-v5 nightly template) loads, passes the config checks and turns on exactly
the v5 switches."""
from pathlib import Path

import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import Config, v3_params

TEMPLATE = Path(__file__).resolve().parents[1] / "deploy" / "config-lst-v5-template"


def test_v5_template_loads_with_the_v5_switches(monkeypatch):
    monkeypatch.setattr(config_mod, "check_protected", lambda path: None)     # model files live on calim
    cfg = Config.load(TEMPLATE)
    pl = cfg.pipeline
    p = pl.peel
    assert (p.engine, p.precision, p.workspace, p.fast_kernels, p.slot_gpu_io, p.solver, p.maxiter, p.peel_fast) == \
        ("sky_rfi", "mixed", True, True, True, "rk4", 5, True)
    assert p.daemon_env(pl.cal.rfi_sources)["ZEST_PEEL_FAST"] == "True"
    assert pl.imaging.gpu_env["TTCALX_V5"] == "1"
    v3 = v3_params(pl)
    assert v3.aoflag == "hour" and v3.aoflag_procs == 16 and v3.retry_on_checkpoint_node
    assert v3.checkpoint == "local" and v3.archive == "none" and v3.hour_source == "memory"   # where the pin applies
    assert sorted(pl.dispatch_band_order) == sorted(pl.bands) and pl.dispatch_band_order[0] == 78


def test_v5_template_paths_are_gsi_v5():
    text = "".join(f.read_text() for f in sorted(TEMPLATE.glob("*.yaml")))
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "gsi-v4" not in body and "gsi_v4" not in body
    d = yaml.safe_load((TEMPLATE / "pipeline.yaml").read_text())
    assert d["env"]["ttcalx"] == "/opt/devel/pipeline/gsi-v5/TTCalX"
