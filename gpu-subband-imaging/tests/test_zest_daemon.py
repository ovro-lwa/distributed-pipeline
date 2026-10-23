"""ZestDaemon protocol, with a stand-in daemon script instead of Julia."""
import os
import stat
import sys
from pathlib import Path

import pytest

from gpu_subband_imaging.stages.peel import ZestDaemon

FAKE = '''
import os, sys
print("DAEMON_READY sources=1 device=fake", flush=True)
for line in sys.stdin:
    ms = line.strip()
    if ms == "QUIT":
        break
    if ms.endswith("bad.ms"):
        print(f"FAILED {ms} boom", flush=True)
        continue
    print('PEEL_SOLVE pass=sky name="Cas A" accepted=true', flush=True)
    print("ZEST_TIMING read=0.10 peel=1.00 write=0.20", flush=True)
    print(f"ZESTED {ms} 1.30 {os.environ.get('ZEST_SOLVER')}", flush=True)
print("DAEMON_EXIT", flush=True)
'''


@pytest.fixture
def fake_julia(tmp_path):
    script = tmp_path / "fake_daemon.py"
    script.write_text(FAKE)
    julia = tmp_path / "julia"
    # Same argument order as Julia: --project=... daemon_jl sources
    julia.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
    julia.chmod(julia.stat().st_mode | stat.S_IEXEC)
    return str(julia)


def test_protocol_and_environment(fake_julia, caplog):
    caplog.set_level("INFO", logger="gsi.peel")
    d = ZestDaemon(fake_julia, "proj", "daemon.jl", "sources.json", env=dict(os.environ),
                   zest_env={"ZEST_SOLVER": "rk4"})
    try:
        assert d.zest(Path("/x/good.ms")) == pytest.approx(1.30)
        with pytest.raises(RuntimeError, match="zest failed"):
            d.zest(Path("/x/bad.ms"))
        assert d.zest(Path("/x/good2.ms")) == pytest.approx(1.30)
    finally:
        d.close()
    text = caplog.text
    assert "PEEL_SOLVE" in text and "ZEST_TIMING read=0.10" in text


def test_cpu_device_defaults_julia_threads(fake_julia, monkeypatch):
    seen = {}
    import subprocess
    real = subprocess.Popen

    def spy(*a, **kw):
        seen.update(kw["env"])
        return real(*a, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    env = {k: v for k, v in os.environ.items() if k != "JULIA_NUM_THREADS"}
    d = ZestDaemon(fake_julia, "proj", "daemon.jl", "s.json", env=env,
                   zest_env={"ZEST_DEVICE": "cpu"})
    d.close()
    assert seen["JULIA_NUM_THREADS"] == "auto"
    assert seen["ZEST_DEVICE"] == "cpu"
