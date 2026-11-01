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
if os.environ.get("FAKE_START_SLEEP"):
    time.sleep(float(os.environ["FAKE_START_SLEEP"]))
print("DAEMON_READY sources=1 device=fake pid=%d" % os.getpid(), flush=True)
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "QUIT":
        break
    parts = cmd.split()
    if parts[0] == "RESET_WARM":
        print(f"RESET_WARM_OK id={parts[1] if len(parts) > 1 else ''}", flush=True)
        continue
    shm = parts[0] == "SHM"
    name = parts[1] if shm else cmd
    rid = f" id={parts[2]}" if shm and len(parts) > 2 else ""
    if name.endswith("hang"):
        time.sleep(3600)
    if name.endswith("die"):
        sys.exit(3)
    if name.endswith("exitafter"):            # reply, then die between commands
        print(f"ZESTED {name} 0.50{rid}", flush=True)
        sys.exit(4)
    if name.endswith("bad"):
        print(f"FAILED {name}{rid} boom", flush=True)
        continue
    if name.endswith("stale"):                 # leftovers of an earlier command first
        print("Traceback line that is not a reply", flush=True)
        print(f"ZESTED {name} 9.99 id=999999", flush=True)
        print(f"FAILED {name} id=999998 old failure", flush=True)
    print('PEEL_SUMMARY pass=all selected=5 accepted=5 action=sky_rfi', flush=True)
    print("ZEST_TIMING read=0.01 peel=1.00 write=0.01", flush=True)
    print(f"ZESTED {name} {'0.50' if shm else '1.50'}{rid}", flush=True)
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


def _daemon(julia, **kw):
    return ZestDaemon(julia, "proj", "daemon.jl", "s.json", env=dict(os.environ), zest_env={}, **kw)


def test_zest_slot_protocol(fake_shm_julia, caplog):
    caplog.set_level("INFO", logger="gsi.peel")
    d = _daemon(fake_shm_julia)
    try:
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_0"), timeout=10) == pytest.approx(0.50)
        with pytest.raises(RuntimeError, match="zest failed"):
            d.zest_slot(Path("/dev/shm/gsi3_1_0_bad"), timeout=10)
        d.reset_warm(timeout=10)
        # the MS command still works on the same daemon (through the reader queue)
        assert d.zest(Path("/x/good.ms")) == pytest.approx(1.50)
        assert d.restarts == 0
    finally:
        d.close()
    assert "PEEL_SUMMARY pass=all" in caplog.text


def test_stale_replies_are_skipped(fake_shm_julia, caplog):
    d = _daemon(fake_shm_julia)
    try:
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_stale"), timeout=10) == pytest.approx(0.50)
    finally:
        d.close()
    assert caplog.text.count("ZEST_STALE_REPLY") == 2


def test_daemon_dead_between_commands_is_restarted(fake_shm_julia, caplog):
    d = _daemon(fake_shm_julia)
    try:
        old = d.pid
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_exitafter"), timeout=10) == pytest.approx(0.50)
        d._p.wait(timeout=10)                       # it is gone now
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_1"), timeout=10) == pytest.approx(0.50)
        assert d.restarts == 1 and d.pid != old
    finally:
        d.close()
    assert "daemon had exited (rc=4)" in caplog.text


def test_broken_pipe_is_restarted_once(fake_shm_julia, monkeypatch):
    d = _daemon(fake_shm_julia)
    try:
        real = d._p.stdin

        class Broken:
            def write(self, s):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                pass

        d._p.stdin = Broken()
        d._p.poll = lambda: None                    # looks alive
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_2"), timeout=10) == pytest.approx(0.50)
        assert d.restarts == 1
        real.close()
    finally:
        d.close()


def test_start_deadline(fake_shm_julia, monkeypatch):
    monkeypatch.setenv("FAKE_START_SLEEP", "30")
    t0 = __import__("time").monotonic()
    with pytest.raises(RuntimeError, match="not ready within"):
        _daemon(fake_shm_julia, start_timeout=0.5)
    assert __import__("time").monotonic() - t0 < 20
    monkeypatch.setenv("FAKE_START_SLEEP", "0.2")
    d = _daemon(fake_shm_julia, start_timeout=10)   # in time: works, stdout usable after
    try:
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_3"), timeout=10) == pytest.approx(0.50)
        assert d.zest(Path("/x/a.ms")) == pytest.approx(1.50)
    finally:
        d.close()


def test_kill_wait_is_bounded(fake_shm_julia, caplog):
    import subprocess
    d = _daemon(fake_shm_julia, kill_wait=0.1)
    try:
        real = d._p

        class Stuck:                                # a process that will not die (D state)
            pid = real.pid

            def kill(self):
                pass

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("julia", timeout)

        d._p = Stuck()
        d._kill("test")
        assert d.abandoned == [real.pid]
        assert "ZEST_DAEMON_ABANDONED" in caplog.text
        d._p = real
    finally:
        d.close()


def test_zest_slot_timeout_kills_and_restarts(fake_shm_julia, caplog):
    from gpu_subband_imaging.stages.peel import ZestTimeout
    d = _daemon(fake_shm_julia)
    try:
        old = d._p
        with pytest.raises(ZestTimeout):
            d.zest_slot(Path("/dev/shm/gsi3_1_0_hang"), timeout=0.5)
        assert old.poll() is not None                  # the stalled daemon was killed
        assert d.restarts == 1 and d.pid != old.pid
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_1"), timeout=10) == pytest.approx(0.50)
        with pytest.raises(RuntimeError, match="closed unexpectedly"):
            d.zest_slot(Path("/dev/shm/gsi3_1_0_die"), timeout=10)
        assert d.restarts == 2                         # restarted after the crash as well
        assert d.zest_slot(Path("/dev/shm/gsi3_1_0_2"), timeout=10) == pytest.approx(0.50)
    finally:
        d.close()
    assert "ZEST_DAEMON_RESTART" in caplog.text
