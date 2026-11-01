"""gsi v3 step 7 (minimal): v3/memory.py store policy, reservations, mid-batch guard; HourStore.spill."""
import json
import os

import numpy as np
import pytest

from gpu_subband_imaging.v3 import hourstore, memory

GB = 1e9


def test_mem_available_parses_meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:  527000000 kB\nMemFree: 1000 kB\nMemAvailable:   65000000 kB\n")
    assert memory.mem_available(str(p)) == 65000000 * 1024.0
    assert memory.mem_available(str(tmp_path / "none")) is None


def test_record_estimate_by_band():
    assert memory.est_record_bytes(23) == memory.RECORD_BYTES
    assert memory.est_record_bytes(82) == memory.RECORD_BYTES_HIGH


def test_auto_ram_when_it_fits(tmp_path):
    p = memory.plan_store("auto", 358, 23, 16.0, [str(tmp_path / "sp")], avail=100 * GB, others=0.0)
    assert p.backend == "ram" and p.spill_dir is None
    assert p.need_bytes == pytest.approx(358 * memory.RECORD_BYTES * memory.NEED_FACTOR)


def test_auto_spills_to_the_first_dir_with_room(tmp_path, monkeypatch):
    small, big = tmp_path / "small", tmp_path / "big"
    real = memory.shutil.disk_usage
    monkeypatch.setattr(memory.shutil, "disk_usage", lambda d: type("U", (), dict(
        free=1 * GB if str(d).startswith(str(small)) else 100 * GB))() if str(d).startswith(str(tmp_path))
        else real(d))
    p = memory.plan_store("auto", 358, 23, 16.0, ["/lustre/x", str(small), str(big)], avail=30 * GB, others=0.0)
    assert p.backend == "memmap" and p.spill_dir == str(big) and "need" in p.reason
    # the other process's reservation counts against this one
    p2 = memory.plan_store("auto", 100, 23, 16.0, [str(big)], avail=30 * GB, others=12 * GB)
    assert p2.backend == "memmap"
    p3 = memory.plan_store("auto", 100, 23, 16.0, [str(big)], avail=30 * GB, others=0.0)
    assert p3.backend == "ram"
    with pytest.raises(RuntimeError, match="no v3.spill_dirs entry"):
        memory.plan_store("auto", 358, 23, 16.0, [str(small)], avail=30 * GB, others=0.0)


def test_forced_modes_and_checkpoint_dir(tmp_path, monkeypatch):
    assert memory.plan_store("ram", 358, 23, 16.0, [], avail=1 * GB, others=0).backend == "ram"
    p = memory.plan_store("spill", 3, 23, 16.0, [str(tmp_path / "a")], avail=1000 * GB, others=0)
    assert p.backend == "memmap" and p.spill_dir == str(tmp_path / "a")
    p = memory.plan_store("auto", 3, 23, 16.0, [str(tmp_path / "a")], avail=1000 * GB, others=0, want_file=True)
    assert p.backend == "ram" and p.spill_dir == str(tmp_path / "a")
    p = memory.plan_store("auto", 3, 23, 16.0, ["/lustre/nope"], avail=1000 * GB, others=0, want_file=True)
    assert p.backend == "ram" and p.spill_dir is None and p.notes
    monkeypatch.setattr(memory, "mem_available", lambda *a: None)      # not Linux
    p = memory.plan_store("auto", 3, 23, 16.0, [], others=0)
    assert p.backend == "ram" and "unknown" in p.reason


def test_reservations_of_live_processes_only(tmp_path):
    assert memory.reserve(5 * GB, root=str(tmp_path), band=23) is not None
    assert memory.reserved_by_others(str(tmp_path)) == 0.0          # our own does not count
    me = memory.reservation_path(str(tmp_path))
    other = tmp_path / f"{memory.RES_PREFIX}{os.getuid()}_1.json"   # pid 1: alive (init)
    other.write_text(json.dumps(dict(bytes=7 * GB)))
    dead = tmp_path / f"{memory.RES_PREFIX}{os.getuid()}_999999999.json"
    dead.write_text(json.dumps(dict(bytes=50 * GB)))
    (tmp_path / f"{memory.RES_PREFIX}junk.json").write_text("{")
    assert memory.reserved_by_others(str(tmp_path)) == 7 * GB
    memory.unreserve(str(tmp_path))
    assert not me.exists()


def test_guard_fires_once_every_n(tmp_path):
    g = memory.MemoryGuard(16.0, every=3)
    assert [g.check(avail=100 * GB) for _ in range(3)] == [False, False, False]
    assert [g.check(avail=1 * GB) for _ in range(3)] == [False, False, True]
    assert [g.check(avail=1 * GB) for _ in range(6)] == [False] * 6


def _rec(seed, stem):
    rng = np.random.default_rng(seed)
    nrow, nchan = 10, 3
    ok = rng.random((nrow, nchan)) < 0.6
    n = int(ok.sum())
    meta = dict(kind="iv", stem=stem, path=f"{stem}.ms", time=5.2e9 + seed, tstart=5.2e9 + seed, tmid=5.2e9 + seed,
                pd=[1.0, 0.6], pd_new=None, nrow=nrow, nchan=nchan, n=n, flagfrac=0.4, zero_frac=0.0,
                telescope="t", observer="o", field_name="f", extra={}, ok_shape=[nrow, nchan])
    arrays = dict(I=(rng.normal(size=n) + 1j).astype(np.complex64), V=(rng.normal(size=n) * 1j).astype(np.complex64),
                  uvw=rng.normal(size=(nrow, 3)), natw=np.ones(nrow, np.float32), ok_bits=hourstore.pack_mask(ok),
                  R=np.eye(3), freqs=np.arange(nchan, dtype=float) + 4e7, chan_width=np.ones(nchan),
                  a1=np.zeros(nrow, np.int32), a2=np.arange(nrow, dtype=np.int32), ant_pos=np.ones((nrow, 3)))
    return hourstore.Record(meta, arrays)


def test_hourstore_spill_ram_only(tmp_path):
    st = hourstore.HourStore(None, backend="ram")
    for k in range(3):
        st.append(_rec(k, f"s{k}"))
    st.append(_rec(9, "s1"))                     # newest record of s1
    with pytest.raises(ValueError):
        st.spill()
    p = st.spill(tmp_path / "sp.gsiv")
    assert st.backend == "memmap" and p == tmp_path / "sp.gsiv" and len(st) == 4
    assert isinstance(st[0].arrays["I"].base, np.memmap) or st[0].arrays["I"].base is not None
    lat = st.latest()
    assert sorted(lat) == ["s0", "s1", "s2"]
    assert lat["s1"].arrays["I"].tobytes() == _rec(9, "s1").arrays["I"].tobytes()
    st.append(_rec(4, "s4"))                     # memmap appends go to the file
    st.close()
    assert [r.stem for r in hourstore.read_records(p)[2]] == ["s0", "s1", "s2", "s1", "s4"]


def test_hourstore_spill_with_checkpoint_file(tmp_path):
    p = tmp_path / "ck.gsiv"
    st = hourstore.HourStore(p, backend="ram", header_meta=dict(mode="iv"))
    for k in range(2):
        st.append(_rec(k, f"s{k}"))
    assert st.spill() == p and st.backend == "memmap" and [r.stem for r in st] == ["s0", "s1"]
    st.append(_rec(2, "s2"))
    st.close()
    st2 = hourstore.HourStore(p, backend="ram")
    assert st2.stems == ["s0", "s1", "s2"]
    st3 = hourstore.HourStore(None)
    st3.append(_rec(0, "x"))
    with pytest.raises(hourstore.GSIVError, match="exists"):      # never into another batch's file
        st3.spill(p)
