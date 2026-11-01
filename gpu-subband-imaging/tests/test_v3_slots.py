"""v3 shared-memory slots: round trip, alignment, state, ring, stale cleanup (no GPU, no CASA)."""
import json
import os
import struct
import subprocess
import sys
import threading

import numpy as np
import pytest

from gpu_subband_imaging.v3 import slots
from gpu_subband_imaging.v3.snap import SnapArrays


def make_snap(nant=6, nchan=5, seed=0, stem="20261031_101709_23MHz", **meta_extra):
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(nant)          # autos and cross, MS-like order
    a1, a2 = a1.astype(np.int32), a2.astype(np.int32)
    nrow = a1.size
    data = (rng.standard_normal((nrow, nchan, 4)) +
            1j * rng.standard_normal((nrow, nchan, 4))).astype(np.complex64)
    data[0, 0, 0] = np.complex64(complex(np.nan, np.inf))     # odd values survive bit exact
    t = np.full(nrow, 5.2e9 + 0.5)
    t[-1] += 1e-3
    meta = dict(freqs=np.linspace(22.6e6, 27.1e6, nchan), chan_width=np.full(nchan, 95703.125),
                phase_dir=np.array([1.234, 0.6496]), ant_pos=rng.standard_normal((nant, 3)) * 1e6,
                telescope="OVRO_MMA", field_name="zenith", observer="lwa", n_subtables=7)
    meta.update(meta_extra)
    return SnapArrays(
        stem=stem, antenna1=a1, antenna2=a2, uvw=rng.standard_normal((nrow, 3)) * 100,
        data=data, flag=rng.random((nrow, nchan, 4)) < 0.3,
        weight=rng.random((nrow, 4)).astype(np.float32),
        sigma=rng.random((nrow, 4)).astype(np.float32), flag_row=rng.random(nrow) < 0.1,
        time=t, time_centroid=t + 0.25, exposure=np.full(nrow, 10.0), interval=np.full(nrow, 10.0),
        meta=meta)


def assert_same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def test_round_trip_bitwise(tmp_path):
    s = make_snap()
    p = tmp_path / "slot"
    h = slots.write_slot(p, s)
    assert slots.get_state(p) == slots.PREPPED
    v = slots.view_slot(p)
    assert v.stem == s.stem
    for f in slots.FIELDS:
        assert_same_bits(getattr(v, f), getattr(s, f))
    for k in ("freqs", "chan_width", "phase_dir", "ant_pos"):
        assert_same_bits(v.meta[k], s.meta[k])
    assert v.meta["telescope"] == "OVRO_MMA" and v.meta["n_subtables"] == 7
    assert v.meta["_slot"]["nrow"] == s.nrow and h["nchan"] == s.nchan
    assert h["mjd"] == (s.time.min() + s.time.max()) / 2 / 86400


def test_alignment_and_header(tmp_path):
    p = tmp_path / "slot"
    h = slots.write_slot(p, make_snap(nant=9, nchan=7))
    raw = p.read_bytes()
    assert raw[:8] == b"GSISLOT1"
    n, state = struct.unpack("<II", raw[8:16])
    assert state == slots.PREPPED
    assert json.loads(raw[64:64 + n]) == h
    assert h["header_bytes"] % 4096 == 0 and h["header_bytes"] >= 64 + n
    ends = []
    for name, e in h["arrays"].items():
        assert e["offset"] % 64 == 0, name
        assert e["offset"] >= h["header_bytes"]
        nb = int(np.prod(e["shape"])) * np.dtype(e["dtype"]).itemsize
        ends.append((e["offset"], e["offset"] + nb))
    ends.sort()
    assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))   # no overlap
    assert h["nbytes"] >= ends[-1][1] and len(raw) >= h["nbytes"]
    e = h["arrays"]["data"]
    assert e["dtype"] == "<c8" and e["shape"] == [45, 7, 4]
    # C order: element [r, c, p] at offset + ((r * nchan + c) * 4 + p) * 8
    v = slots.view_slot(p)
    assert v.data.flags.c_contiguous
    r, c, q = 3, 2, 1
    off = e["offset"] + ((r * 7 + c) * 4 + q) * 8
    assert raw[off:off + 8] == v.data[r, c, q].tobytes()


def test_view_is_zero_copy_and_shared(tmp_path):
    p = tmp_path / "slot"
    slots.write_slot(p, make_snap())
    v1 = slots.view_slot(p)
    v2 = slots.view_slot(p)
    v1.data[1, 1, 1] = 7 + 8j
    v1.flag[1, 1, :] = True
    assert v2.data[1, 1, 1] == np.complex64(7 + 8j) and v2.flag[1, 1].all()
    ro = slots.view_slot(p, writable=False)
    with pytest.raises(ValueError):
        ro.data[0, 0, 0] = 0


def test_large_meta_grows_header(tmp_path):
    p = tmp_path / "slot"
    s = make_snap(history=["x" * 100] * 80)        # > 4096 bytes of JSON
    h = slots.write_slot(p, s)
    assert h["header_bytes"] >= 8192 and h["header_bytes"] % 4096 == 0
    assert slots.view_slot(p).meta["history"] == ["x" * 100] * 80


def test_state_transitions(tmp_path):
    p = tmp_path / "slot"
    slots.write_slot(p, make_snap())
    for st in (slots.PEELED, slots.FREE, slots.PREPPED):
        slots.set_state(p, st)
        assert slots.get_state(p) == st
        assert slots.read_header(p)["state"] == st
    with pytest.raises(ValueError):
        slots.set_state(p, 9)


def test_reuse_smaller_and_larger(tmp_path):
    p = tmp_path / "slot"
    big, small = make_snap(nant=10, nchan=8, seed=1), make_snap(nant=4, nchan=3, seed=2)
    slots.write_slot(p, big)
    size = p.stat().st_size
    slots.write_slot(p, small)
    assert p.stat().st_size == size                 # never shrunk
    v = slots.view_slot(p)
    assert_same_bits(v.data, small.data)
    slots.write_slot(p, make_snap(nant=14, nchan=8, seed=3))
    assert p.stat().st_size > size


def test_validation(tmp_path):
    s = make_snap()
    s.data = s.data.astype(np.complex128)
    with pytest.raises(TypeError):
        slots.write_slot(tmp_path / "a", s)
    s = make_snap()
    del s.meta["ant_pos"]
    with pytest.raises(ValueError):
        slots.write_slot(tmp_path / "b", s)
    s = make_snap()
    s.antenna1 = s.antenna1.astype(np.int64)        # value-preserving cast is fine
    slots.write_slot(tmp_path / "c", s)
    assert slots.view_slot(tmp_path / "c").antenna1.dtype == np.int32
    (tmp_path / "junk").write_bytes(b"x" * 100)
    with pytest.raises(ValueError):
        slots.read_header(tmp_path / "junk")


class _Exploding:
    """Looks like the flag array to the layout, raises when copied into the slot."""

    def __init__(self, a):
        self.shape, self.dtype, self.nbytes = a.shape, a.dtype, a.nbytes

    def __array__(self, *args, **kw):
        raise RuntimeError("boom")


def test_crash_mid_write_is_not_ready(tmp_path, monkeypatch):
    p = tmp_path / "slot"
    slots.write_slot(p, make_snap())
    assert slots.get_state(p) == slots.PREPPED
    orig = slots._layout

    def bad_layout(snap):
        head, arrays = orig(snap)
        arrays["flag"] = _Exploding(arrays["flag"])
        return head, arrays

    monkeypatch.setattr(slots, "_layout", bad_layout)
    with pytest.raises(RuntimeError, match="boom"):
        slots.write_slot(p, make_snap(seed=5))
    assert slots.get_state(p) == slots.FREE         # a half-written slot never looks ready


def test_ring_acquire_release_and_cleanup(tmp_path):
    ring = slots.SlotRing(2, slot_bytes=1 << 16, root=tmp_path)
    names = sorted(x.name for x in tmp_path.iterdir())
    assert names == [f"gsi3_{os.getpid()}_0", f"gsi3_{os.getpid()}_1"]
    assert all(x.stat().st_size == 1 << 16 for x in tmp_path.iterdir())
    a = ring.acquire()
    b = ring.acquire()
    with pytest.raises(TimeoutError):
        ring.acquire(timeout=0.05)
    slots.write_slot(a, make_snap())
    got = []
    t = threading.Thread(target=lambda: got.append(ring.acquire(timeout=5)))
    t.start()
    ring.release(a)
    t.join()
    assert got == [a] and slots.get_state(a) == slots.FREE
    with pytest.raises(ValueError):
        ring.release(tmp_path / "other")
    ring.release(b)
    with pytest.raises(ValueError):
        ring.release(b)
    ring.close()
    assert list(tmp_path.iterdir()) == []
    ring.close()                                    # idempotent


def test_stale_cleanup_dead_pid_only(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    dead = p.pid                                    # a pid that is gone
    stale = [tmp_path / f"gsi3_{dead}_{k}" for k in range(3)]
    for s in stale:
        s.write_bytes(b"old")
    live = tmp_path / f"gsi3_{os.getppid()}_0"      # alive (our parent)
    live.write_bytes(b"live")
    mine = tmp_path / f"gsi3_{os.getpid()}_7"       # this process
    mine.write_bytes(b"mine")
    other = tmp_path / "gsi3_mem_gpu0.json"         # not a slot name
    other.write_bytes(b"{}")
    ring = slots.SlotRing(1, root=tmp_path)
    assert sorted(ring.removed_stale) == sorted(str(s) for s in stale)
    assert live.exists() and mine.exists() and other.exists()
    ring.close()


def test_ring_removed_at_exit_and_after_crash(tmp_path):
    code = ("import sys; from gpu_subband_imaging.v3 import slots;"
            "r = slots.SlotRing(3, root=sys.argv[1]);"
            "import os; print(os.getpid(), flush=True);"
            "os._exit(0) if sys.argv[2] == 'crash' else None")
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path), "exit"],
                         capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() and list(tmp_path.iterdir()) == []      # atexit removed them
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path), "crash"],
                         capture_output=True, text=True, env=env, check=True)
    pid = int(out.stdout.strip())
    assert len([x for x in tmp_path.iterdir() if x.name.startswith(f"gsi3_{pid}_")]) == 3
    ring = slots.SlotRing(1, root=tmp_path)                            # stale cleanup
    assert len(ring.removed_stale) == 3
    ring.close()
    assert list(tmp_path.iterdir()) == []
