"""v3.peel_shm.peel_ms_via_slot and Worker._zest dispatch (casacore I/O mocked, no daemon)."""
import errno
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_subband_imaging import worker as worker_mod
from gpu_subband_imaging.config import V3Params
from gpu_subband_imaging.v3 import peel_shm, slots

from test_v3_slots import make_snap


class FakeDaemon:
    """zest_slot peels in place like the Julia daemon: data * 2, then state PEELED."""

    def __init__(self, state=slots.PEELED):
        self.calls, self.state = [], state

    def zest_slot(self, slot, timeout=None):
        self.calls.append(("slot", str(slot), timeout))
        v = slots.view_slot(slot)
        with np.errstate(invalid="ignore"):          # make_snap holds a NaN/Inf sample
            v.data[...] *= 2
        del v
        slots.set_state(slot, self.state)
        return 1.25

    def reset_warm(self, timeout=None):
        self.calls.append(("reset_warm",))

    def zest(self, ms):
        self.calls.append(("ms", str(ms)))
        return 2.0


@pytest.fixture
def io(monkeypatch):
    snap = make_snap()
    written = {}
    monkeypatch.setattr(peel_shm, "snap_from_ms", lambda ms: make_snap())
    monkeypatch.setattr(peel_shm, "write_peeled",
                        lambda ms, s: written.update(ms=ms, data=np.array(s.data), flag=np.array(s.flag)))
    return snap, written


def test_peel_ms_via_slot_round_trip(tmp_path, io):
    snap, written = io
    ring = slots.SlotRing(1, root=tmp_path)
    d = FakeDaemon()
    t = {}
    assert peel_shm.peel_ms_via_slot(d, "/x/a.ms", ring, timeout=7, timings=t) == 1.25
    assert written["ms"] == "/x/a.ms"
    with np.errstate(invalid="ignore"):
        assert written["data"].tobytes() == (snap.data * 2).tobytes()
    assert written["flag"].tobytes() == snap.flag.tobytes()
    assert set(t) == {"ms_read", "slot_write", "ms_write"}
    assert d.calls == [("slot", str(ring.paths[0]), 7)]
    assert ring.n_free == 1 and slots.get_state(ring.paths[0]) == slots.FREE
    ring.close()


def test_not_peeled_resets_warm_start(tmp_path, io):
    ring = slots.SlotRing(1, root=tmp_path)
    d = FakeDaemon(state=slots.PREPPED)
    with pytest.raises(RuntimeError, match="PREPPED after ZESTED"):
        peel_shm.peel_ms_via_slot(d, "/x/a.ms", ring)
    assert d.calls[-1] == ("reset_warm",) and ring.n_free == 1
    ring.close()


def test_write_back_failure_resets_warm_start(tmp_path, io, monkeypatch):
    def boom(ms, s):
        raise IOError("disk")
    monkeypatch.setattr(peel_shm, "write_peeled", boom)
    ring = slots.SlotRing(1, root=tmp_path)
    d = FakeDaemon()
    with pytest.raises(IOError):
        peel_shm.peel_ms_via_slot(d, "/x/a.ms", ring)
    assert d.calls[-1] == ("reset_warm",) and ring.n_free == 1
    ring.close()


def _enospc(fd, off, n):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_full_shm_raises_slot_unavailable_before_peel(tmp_path, io, monkeypatch):
    ring = slots.SlotRing(1, root=tmp_path)
    monkeypatch.setattr(slots.os, "posix_fallocate", _enospc, raising=False)
    d = FakeDaemon()
    with pytest.raises(peel_shm.SlotUnavailable):
        peel_shm.peel_ms_via_slot(d, "/x/a.ms", ring)
    assert d.calls == [] and ring.n_free == 1           # the daemon never saw it
    ring.close()


def _ring_in(monkeypatch, root, slot_bytes=0):
    real = slots.SlotRing
    monkeypatch.setattr(slots, "SlotRing", lambda n, **k: real(n, slot_bytes=slot_bytes, root=root))


def _worker(peel_io, tmp_path):
    w = object.__new__(worker_mod.Worker)
    w.band, w.batch, w._slots = 23, 0, None
    w.cfg = SimpleNamespace(pipeline=SimpleNamespace(v3=SimpleNamespace(
        peel_io=peel_io, peel_timeout_s=11.0)))
    return w


def test_worker_zest_dispatch_ms(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(peel_shm, "peel_ms_via_slot", lambda *a, **k: called.append(a))
    w, d = _worker("ms", tmp_path), FakeDaemon()
    w._zest(d, "s", tmp_path / "a.ms")
    assert d.calls == [("ms", str(tmp_path / "a.ms"))] and not called and w._slots is None
    w.cfg = SimpleNamespace(pipeline=SimpleNamespace())     # no v3 section at all: v2
    w._zest(d, "s", tmp_path / "b.ms")
    assert d.calls[-1] == ("ms", str(tmp_path / "b.ms"))


def test_worker_zest_dispatch_shm(tmp_path, monkeypatch, caplog):
    seen = {}

    def fake(daemon, ms, ring, timeout=None, timings=None):
        seen.update(ms=ms, ring=ring, timeout=timeout)
        timings["slot_write"] = 0.01
        return 1.0

    monkeypatch.setattr(peel_shm, "peel_ms_via_slot", fake)
    _ring_in(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="gsi.worker")
    w, d = _worker("shm", tmp_path), FakeDaemon()
    w._zest(d, "stampA", tmp_path / "a.ms")
    assert seen["timeout"] == 11.0 and seen["ring"] is w._slots and d.calls == []
    assert "stamp=stampA phase=slot_write" in caplog.text
    w._slots.close()


def test_worker_falls_back_to_ms_when_shm_full(tmp_path, monkeypatch, caplog):
    def full(*a, **k):
        raise peel_shm.SlotUnavailable("cannot write slot: [Errno 28] No space left on device")

    monkeypatch.setattr(peel_shm, "peel_ms_via_slot", full)
    _ring_in(monkeypatch, tmp_path)
    w, d = _worker("shm", tmp_path), FakeDaemon()
    w._zest(d, "stampB", tmp_path / "b.ms")
    assert d.calls == [("ms", str(tmp_path / "b.ms"))]
    assert "V3_FALLBACK 23MHz stamp=stampB" in caplog.text
    w._slots.close()


def test_worker_falls_back_when_ring_cannot_be_created(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(slots.os, "posix_fallocate", _enospc, raising=False)
    _ring_in(monkeypatch, tmp_path, slot_bytes=1 << 20)
    w, d = _worker("shm", tmp_path), FakeDaemon()
    w._zest(d, "stampC", tmp_path / "c.ms")
    assert d.calls == [("ms", str(tmp_path / "c.ms"))] and "V3_FALLBACK" in caplog.text
    assert w._slots is None and list(tmp_path.iterdir()) == []


def test_v3_peel_timeouts_validated():
    from gpu_subband_imaging.config import ConfigError
    assert V3Params().peel_start_timeout_s == 600.0
    with pytest.raises(ConfigError):
        V3Params(peel_start_timeout_s=0)
