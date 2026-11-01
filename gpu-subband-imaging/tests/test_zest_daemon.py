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


# -- gsi v3: SHM slots with a reply deadline ------------------------------------------------

FAKE_SHM = '''
import os, sys, time
print("DAEMON_READY sources=1 device=fake pid=%d" % os.getpid(), flush=True)
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "QUIT":
        break
    shm = cmd.startswith("SHM ")
    name = cmd[4:].strip() if shm else cmd
    if name.endswith("hang"):
        time.sleep(3600)
    if name.endswith("die"):
        sys.exit(3)
    if name.endswith("bad"):
        print(f"FAILED {name} boom", flush=True)
        continue
    print('PEEL_SUMMARY pass=all selected=5 accepted=5 action=sky_rfi', flush=True)
    print("ZEST_TIMING read=0.01 peel=1.00 write=0.01", flush=True)
    print(f"ZESTED {name} {'0.50' if shm else '1.50'} pid={os.getpid()}", flush=True)
print("DAEMON_EXIT", flush=True)
'''


@pytest.fixture
def fake_shm_julia(tmp_path):
    script = tmp_path / "fake_shm_daemon.py"
    script.write_text(FAKE_SHM)
    julia = tmp_path / "julia"
    julia.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
    julia.chmod(julia.stat().st_mode | stat.S_IEXEC)
    return str(julia)


def test_zest_slot_protocol(fake_shm_julia, caplog):
    caplog.set_level("INFO", logger="gsi.peel")
    d = ZestDaemon(fake_shm_julia, "proj", "daemon.jl", "s.json", env=dict(os.environ), zest_env={})
    try:
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0"), timeout=10) == pytest.approx(0.50)
        with pytest.raises(RuntimeError, match="zest failed"):
            d.zest_slot(Path("/dev/shm/gsi3_1_bad"), timeout=10)
        # the MS command still works on the same daemon (through the reader queue)
        assert d.zest(Path("/x/good.ms")) == pytest.approx(1.50)
        assert d.restarts == 0
    finally:
        d.close()
    assert "PEEL_SUMMARY pass=all" in caplog.text


def test_zest_slot_timeout_kills_and_restarts(fake_shm_julia, caplog):
    from gpu_subband_imaging.stages.peel import ZestTimeout
    d = ZestDaemon(fake_shm_julia, "proj", "daemon.jl", "s.json", env=dict(os.environ), zest_env={})
    try:
        old = d._p
        with pytest.raises(ZestTimeout):
            d.zest_slot(Path("/dev/shm/gsi3_1_hang"), timeout=0.5)
        assert old.poll() is not None                  # the stalled daemon was killed
        assert d.restarts == 1 and d.pid != old.pid
        assert d.zest_slot(Path("/dev/shm/gsi3_1_1"), timeout=10) == pytest.approx(0.50)
        with pytest.raises(RuntimeError, match="closed unexpectedly"):
            d.zest_slot(Path("/dev/shm/gsi3_1_die"), timeout=10)
        assert d.restarts == 2                         # restarted after the crash as well
        assert d.zest_slot(Path("/dev/shm/gsi3_1_2"), timeout=10) == pytest.approx(0.50)
    finally:
        d.close()
    assert "ZEST_DAEMON_RESTART" in caplog.text
