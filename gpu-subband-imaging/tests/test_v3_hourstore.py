"""gsi v3 step 5: GSIV record files (v3/hourstore.py), the worker hook and the archive hour job.

Format tests need numpy only.  The record-building tests need the imager's msio
(ttcalx_imager >= the gsi v3 change) and python-casacore: they look for the imager source in
$TTCALX_IMAGER_SRC or beside this repo and skip otherwise.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_subband_imaging import hour_lst
from gpu_subband_imaging.config import ConfigError, HourImagingParams, V3Params, v3_params
from gpu_subband_imaging.v3 import hourstore as hs
from gpu_subband_imaging.v3.snap import SnapArrays

REPO_SRC = Path(hs.__file__).resolve().parents[2]


def _imager_src():
    cands = [os.environ.get("TTCALX_IMAGER_SRC", "")]
    here = Path(__file__).resolve()
    for up in here.parents:
        cands += [str(up / "ttcalx_imager" / "src"), str(up / "python" / "ttcalx_imager" / "src")]
    for c in cands:
        if c and (Path(c) / "ttcalx_imager" / "gsiv.py").exists():
            return c
    return None


IMAGER_SRC = _imager_src()
if IMAGER_SRC and IMAGER_SRC not in sys.path:
    sys.path.insert(0, IMAGER_SRC)
try:
    import casacore.tables  # noqa: F401
    CASACORE = True
except Exception:
    CASACORE = False
needs_imager = pytest.mark.skipif(IMAGER_SRC is None, reason="ttcalx_imager source (with gsiv.py) not found")
needs_ms = pytest.mark.skipif(IMAGER_SRC is None or not CASACORE, reason="needs ttcalx_imager source + casacore")


def _rec(seed=0, nrow=40, nchan=6, stem=None):
    rng = np.random.default_rng(seed)
    ok = rng.random((nrow, nchan)) < 0.6
    n = int(ok.sum())
    meta = dict(kind="iv", stem=stem or f"20261031_1010{seed:02d}_55MHz", path=f"/x/{seed}.ms", time=5.3e9 + 10 * seed,
                tstart=5.3e9 + 10 * seed - 5, tmid=5.3e9 + 10 * seed, pd=[1.0, 0.6], pd_new=[1.3, 0.65],
                nrow=nrow, nchan=nchan, n=n, flagfrac=float(1 - ok.mean()), zero_frac=0.0, telescope="OVRO-LWA",
                observer="", field_name="", extra={}, ok_shape=[nrow, nchan])
    arrays = dict(I=(rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64),
                  V=(rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64),
                  uvw=rng.normal(0, 100, (nrow, 3)), natw=rng.random(nrow).astype(np.float32),
                  ok_bits=hs.pack_mask(ok), R=np.eye(3) + 1e-3 * seed, freqs=np.linspace(4e7, 4.1e7, nchan),
                  chan_width=np.full(nchan, 1e5), a1=np.arange(nrow, dtype=np.int32),
                  a2=np.arange(nrow, dtype=np.int32) + 1, ant_pos=np.ones((nrow + 1, 3)))
    return hs.Record(meta, arrays), ok


def _write(path, n=3, **kw):
    recs = [_rec(seed=s, **kw)[0] for s in range(n)]
    w = hs.RecordWriter(path, dict(mode="iv", band=55), {k: recs[0].arrays[k] for k in hs.SHARED_ARRAYS})
    for r in recs:
        w.append(r.meta, r.arrays)
    w.close()
    return recs


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ----------------------------------------------------------------------------- format
def test_mask_packing_roundtrip():
    rng = np.random.default_rng(1)
    for shape in ((61776, 48), (7, 3), (1, 1), (0, 4)):
        ok = rng.random(shape) < 0.5
        assert _eq(hs.unpack_mask(hs.pack_mask(ok), shape), ok)


@pytest.mark.parametrize("backend", ["ram", "memmap"])
def test_roundtrip_bitwise(tmp_path, backend):
    p = tmp_path / "x.gsiv"
    recs = _write(p)
    hm, ha, got = hs.read_records(p, backend=backend)
    assert hm["format"] == "GSIV" and hm["version"] == 1 and hm["mode"] == "iv" and hm["band"] == 55
    assert [g.stem for g in got] == [r.stem for r in recs]
    for r, g in zip(recs, got):
        assert {k: v for k, v in g.meta.items() if k != "written_ns"} == r.meta and g.meta["written_ns"] > 0
        assert set(g.arrays) == set(r.arrays)
        for k in r.arrays:
            assert _eq(g.arrays[k], r.arrays[k]), k
        assert _eq(g.ok(), hs.unpack_mask(r.arrays["ok_bits"], r.meta["ok_shape"]))
        assert not g.arrays["I"].flags.writeable
    sc = hs.scan(p, verify=True)
    assert sc.clean and sc.size == sc.end and len(sc.entries) == 3
    e = sc.entries[1]
    assert (e["seq"], e["stem"], e["n_kept"]) == (2, recs[1].stem, recs[1].n)
    assert e["R"] == recs[1].arrays["R"].tolist() and e["flag_frac"] == recs[1].meta["flagfrac"]
    assert all(x["offset"] % 64 == 0 and x["nbytes"] % 64 == 0 for x in sc.entries)
    # the index line is written after the record: entries == index file lines
    lines = [json.loads(x) for x in hs.idx_path(p).read_text().splitlines()]
    assert lines == sc.entries


def test_encoding_is_deterministic_and_little_endian(tmp_path):
    r, _ = _rec()
    a = r.arrays.copy()
    a["uvw"] = a["uvw"].astype(">f8")                 # big-endian input is stored little-endian
    h1, p1, n1, c1 = hs.encode_block(hs.MAGIC_REC, 1, r.meta, r.arrays)
    h2, p2, n2, c2 = hs.encode_block(hs.MAGIC_REC, 1, r.meta, a)
    assert (h1, n1, c1) == (h2, n2, c2) and b"".join(bytes(x) for x in p1) == b"".join(bytes(x) for x in p2)
    with pytest.raises(hs.GSIVError, match="bit-packed"):
        hs.encode_block(hs.MAGIC_REC, 1, {}, dict(ok=np.ones(3, bool)))


def test_torn_tail_and_resume(tmp_path):
    p = tmp_path / "x.gsiv"
    recs = _write(p)
    e3 = hs.scan(p).entries[2]
    with open(p, "r+b") as fh:
        fh.truncate(e3["offset"] + 100)                 # crash inside record 3
    sc = hs.scan(p)
    assert len(sc.entries) == 2 and sc.torn_bytes == 100 and sc.idx_dropped == 1 and not sc.clean
    with pytest.raises(hs.GSIVError, match="torn"):
        hs.read_records(p)
    rep = {}
    assert [r.stem for r in hs.read_records(p, strict=False)[2]] == [recs[0].stem, recs[1].stem]
    hs.read_blocks(p, strict=False, report=rep)
    assert any("torn" in x for x in rep["problems"]) and any("index line" in x for x in rep["problems"])
    with pytest.raises(hs.GSIVError, match="torn"):
        hs.HourStore(p, backend="memmap")               # strict by default
    st = hs.HourStore(p, backend="memmap", strict=False)   # opening a store does not modify the file
    assert len(st) == 2 and p.stat().st_size == e3["offset"] + 100
    st.append(recs[2])                                  # appending resumes: truncate + rewrite index
    st.close()
    sc = hs.scan(p, verify=True)
    assert sc.clean and [e["stem"] for e in sc.entries] == [r.stem for r in recs]
    assert _eq(st[2].arrays["I"], recs[2].arrays["I"])


def test_record_fsynced_but_index_line_lost(tmp_path):
    p = tmp_path / "x.gsiv"
    _write(p)
    idx = hs.idx_path(p)
    lines = idx.read_text().splitlines()
    idx.write_text(lines[0] + "\n" + lines[1][:30])     # 2nd line torn, 3rd never written
    sc = hs.scan(p)
    assert len(sc.entries) == 3 and sc.unindexed == 2 and sc.idx_dropped == 1 and sc.torn_bytes == 0
    hs.repair(p)
    assert hs.scan(p).clean and [json.loads(x)["seq"] for x in idx.read_text().splitlines()] == [1, 2, 3]
    idx.unlink()                                         # no index at all: the file describes itself
    assert len(hs.scan(p).entries) == 3


def test_crc_failure(tmp_path):
    p = tmp_path / "x.gsiv"
    _write(p)
    e = hs.scan(p).entries[1]
    with open(p, "r+b") as fh:
        fh.seek(e["offset"] + e["nbytes"] - 200)
        b = fh.read(1); fh.seek(-1, 1); fh.write(bytes([b[0] ^ 0x10]))
    assert hs.scan(p).bad_crc == [] and hs.scan(p, verify=True).bad_crc == [2]
    with pytest.raises(hs.GSIVError, match="crc"):
        hs.read_records(p)
    with pytest.raises(hs.GSIVError, match="crc"):
        hs.HourStore(p)
    assert [r.entry["seq"] for r in hs.read_records(p, strict=False)[2]] == [1, 3]
    sc = hs.repair(p)                                    # resume semantics: valid prefix only
    assert len(sc.entries) == 1 and hs.scan(p, verify=True).clean


def test_corrupt_block_header_ends_the_scan(tmp_path):
    p = tmp_path / "x.gsiv"
    _write(p)
    e = hs.scan(p).entries[2]
    hs.idx_path(p).unlink()
    with open(p, "r+b") as fh:                           # garbage where record 3's header is
        fh.seek(e["offset"] + 10); fh.write(b"\x00\x01")
    sc = hs.scan(p)
    assert len(sc.entries) == 2 and sc.torn_bytes == e["nbytes"]


def test_bad_or_foreign_file(tmp_path):
    q = tmp_path / "junk.gsiv"
    q.write_bytes(os.urandom(300))
    with pytest.raises(hs.GSIVError):
        hs.scan(q)
    p = tmp_path / "x.gsiv"
    _write(p, n=1)
    with open(p, "r+b") as fh:
        fh.seek(80); fh.write(b"\xff\xff")
    with pytest.raises(hs.GSIVError, match="header"):
        hs.scan(p)


def test_writer_refuses_mismatched_resume_and_unknown_mode(tmp_path):
    p = tmp_path / "x.gsiv"
    _write(p, n=1)
    with pytest.raises(hs.GSIVError, match="mode"):
        hs.RecordWriter(p, dict(mode="full"))
    with pytest.raises(hs.GSIVError, match="exists"):
        hs.RecordWriter(p, dict(mode="iv"), resume=False)
    with pytest.raises(hs.GSIVError, match="not implemented"):
        hs.RecordWriter(tmp_path / "y.gsiv", dict(mode="full"))


def test_hourstore_ram_and_memmap(tmp_path):
    recs = [_rec(seed=s)[0] for s in range(3)]
    ram = hs.HourStore()                                 # RAM only, no file
    for r in recs:
        ram.append(r)
    assert ram.stems == [r.stem for r in recs] and ram.nbytes() == sum(r.nbytes() for r in recs)
    p = tmp_path / "ck.gsiv"
    for backend in ("ram", "memmap"):
        if p.exists():
            p.unlink()
        st = hs.HourStore(p, backend=backend, header_meta=dict(mode="iv"))
        for r in recs[:2]:
            st.append(r)
        st.close()
        st = hs.HourStore(p, backend=backend)            # reopen + continue (checkpoint)
        st.append(recs[2])
        st.close()
        back = hs.HourStore(p, backend=backend)
        assert back.stems == [r.stem for r in recs]
        for r, g in zip(recs, back):
            for k in r.arrays:
                assert _eq(g.arrays[k], r.arrays[k]), (backend, k)
        if backend == "memmap":
            assert isinstance(back[0].arrays["I"].base, np.memmap) or back[0].arrays["I"].base is not None


def test_shared_arrays_dropped_only_when_equal(tmp_path):
    p = tmp_path / "x.gsiv"
    r0, _ = _rec(seed=0)
    r1, _ = _rec(seed=1)
    r1.arrays["a1"] = r1.arrays["a1"][::-1].copy()        # a different row order is kept in the record
    w = hs.RecordWriter(p, dict(mode="iv"), {k: r0.arrays[k] for k in hs.SHARED_ARRAYS})
    w.append(r0.meta, r0.arrays); w.append(r1.meta, r1.arrays); w.close()
    _, ha, got = hs.read_records(p)
    assert _eq(got[0].arrays["a1"], ha["a1"]) and _eq(got[1].arrays["a1"], r1.arrays["a1"])
    assert got[0].entry["nbytes"] < got[1].entry["nbytes"]


@needs_imager
def test_imager_reader_copy_is_in_sync():
    """ttcalx_imager/gsiv.py carries a verbatim copy of the shared section of hourstore.py."""
    def section(path):
        s = Path(path).read_text()
        a = s.index("# ============================================================================= GSIV v1 codec")
        b = s.index("# END shared section")
        return s[a:b]
    assert section(hs.__file__) == section(Path(IMAGER_SRC) / "ttcalx_imager" / "gsiv.py")


@needs_imager
def test_imager_reads_pipeline_file(tmp_path):
    from ttcalx_imager import gsiv
    p = tmp_path / "x.gsiv"
    recs = _write(p)
    snaps = gsiv.load_archive(p, 1.3, 0.65)
    from ttcalx_imager.msio import expand_coord
    for r, s in zip(recs, snaps):
        ok = hs.unpack_mask(r.arrays["ok_bits"], r.meta["ok_shape"])
        assert _eq(s["u"], expand_coord(r.arrays["uvw"], 0, r.arrays["freqs"], ok))
        assert _eq(s["I"], r.arrays["I"]) and _eq(s["R"], r.arrays["R"]) and s.n == r.n


# ----------------------------------------------------------------------------- records from MS / SnapArrays
def _make_ms(path, seed=0):
    sys.path.insert(0, str(Path(IMAGER_SRC).parents[0] / "tests"))
    try:
        from msmaker import make_ms
    finally:
        sys.path.pop(0)
    return make_ms(str(path), seed=seed)


@needs_ms
def test_records_from_ms_and_snap_equal_load_samples(tmp_path):
    from ttcalx_imager import gsiv
    from ttcalx_imager.msio import load_samples
    p = tmp_path / "20261031_101007_55MHz.ms"
    d = _make_ms(p)
    rm = hs.records_from_ms(p)
    snap = SnapArrays(stem=p.name[:-3], antenna1=d["a1"], antenna2=d["a2"], uvw=d["uvw"], data=d["data"],
                      flag=d["flag"], weight=d["weight"], sigma=d["weight"], flag_row=np.zeros(len(d["a1"]), bool),
                      time=d["time"], time_centroid=d["time"], exposure=d["time"] * 0 + 10,
                      interval=d["time"] * 0 + 10,
                      meta=dict(path=str(p), freqs=d["freqs"], chan_width=d["chan_width"], phase_dir=d["phase_dir"],
                                ant_pos=d["ant_pos"], telescope="OVRO-LWA", observer="test", field_name="zenith"))
    rs = hs.records_from_snap(snap)
    assert rm.meta == rs.meta
    for k in rm.arrays:
        assert _eq(rm.arrays[k], rs.arrays[k]), k
    f = tmp_path / "a.gsiv"
    st = hs.HourStore(f, header_meta=dict(mode="iv"))
    st.append(rm); st.close()
    s = gsiv.load_archive(f)[0]
    ref = load_samples(str(p))
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            assert _eq(s[k], v), k
        else:
            assert s[k] == v and type(s[k]) is type(v), k


@needs_ms
def test_archive_client_serve_end_to_end(tmp_path):
    from ttcalx_imager import gsiv
    from ttcalx_imager.msio import load_samples
    ms = []
    for k in range(3):
        p = tmp_path / f"20261031_1010{k:02d}_55MHz.ms"
        _make_ms(p, seed=k); ms.append(p)
    env = _serve_env()
    out = tmp_path / "arch" / "55MHz-x-04h-peeled-b000.gsiv"
    hdr = hs.header_for(55, "2026-10-31", "04h", None, "Run_T", batch=0, cal_bp="b", cal_xy="x")
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=env, timeout=120)
    assert c.info["imager"].startswith(IMAGER_SRC)
    rs = [c.add(m) for m in ms]
    assert all(r["ok"] for r in rs) and [r["seq"] for r in rs] == [1, 2, 3]
    _make_ms(ms[1], seed=7)                                    # reprocessed: other data, same stem
    r = c.add(ms[1])
    assert r["superseded"] and r["seq"] == 4
    avg = tmp_path / "20261031_101099_55MHz_avg.ms"            # the worker's work-dir name
    _make_ms(avg, seed=9)
    r = c.add(avg, stem="20261031_101099_55MHz", path="/keep/20261031_101099_55MHz.ms")
    assert r["stem"] == "20261031_101099_55MHz" and r["seq"] == 5 and not r["superseded"]
    with pytest.raises(hs.GSIVError):
        c.add(tmp_path / "missing.ms")                         # an error reply; the writer keeps serving
    assert c.alive and c.add(ms[0])["seq"] == 6
    c.close()
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=env, timeout=120)   # rerun: resumes
    assert c.info["n"] == 6
    c.close()
    with pytest.raises(hs.GSIVError, match="cal_bp"):          # another calibration: no resume
        hs.ArchiveClient(sys.executable, out, None, dict(hdr, cal_bp="other"), env=env, timeout=120)
    sc = hs.scan(out, verify=True)
    assert sc.clean and len(sc.entries) == 6 and sc.header_meta["group"] == "04h"
    inv = hs.archive_inventory([out], band=55)
    assert inv["stems"] == sorted([m.name[:-3] for m in ms] + ["20261031_101099_55MHz"]) and not inv["problems"]
    snaps = {s["stem"]: s for s in gsiv.load_archive(out, band=55)}
    assert snaps["20261031_101099_55MHz"]["path"] == "/keep/20261031_101099_55MHz.ms"
    for m in ms:                                                # the newest record of each stem
        ref = load_samples(str(m))
        for k in ("u", "v", "w", "nw", "I", "V", "R", "freqs"):
            assert _eq(snaps[m.name[:-3]][k], ref[k]), k
    assert not Path(f"{out}.lock").exists()


def _serve_env(**kw):
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_SRC), IMAGER_SRC]), CUDA_VISIBLE_DEVICES="", **kw)


@needs_ms
def test_archive_client_timeout_kills_and_restarts(tmp_path):
    ms = []
    for k in range(3):
        p = tmp_path / f"20261031_1010{k:02d}_55MHz.ms"
        _make_ms(p, seed=k); ms.append(p)
    out = tmp_path / "a.gsiv"
    hdr = hs.header_for(55, "2026-10-31", "04h", None, "Run_T")
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=_serve_env(GSIV_SERVE_TEST_SLEEP="2:30"), timeout=3)
    assert c.add(ms[0])["seq"] == 1
    with pytest.raises(hs.GSIVError, match="writer killed"):
        c.add(ms[1])                                            # no reply in time: the child is killed
    assert not c.alive and c.p.poll() is not None
    with pytest.raises(hs.GSIVError, match="not running"):
        c.add(ms[2])                                            # never a stale reply for another request
    c.close()
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=_serve_env(), timeout=60)   # restart: resumes
    assert c.info["n"] == 1
    assert c.add(ms[1])["seq"] == 2 and c.add(ms[2])["seq"] == 3
    c.close()
    assert hs.scan(out, verify=True).stems == [m.name[:-3] for m in ms]


def test_archive_client_matches_replies_to_requests(tmp_path):
    fake = tmp_path / "fake_serve.py"
    fake.write_text("""import json, sys
out = sys.stdout
out.write(json.dumps(dict(id=0, ok=True, n=0, pid=1)) + "\\n"); out.flush()
for line in sys.stdin:
    q = json.loads(line)
    if q.get("op") == "close":
        break
    out.write(json.dumps(dict(id=q["id"] + 1000, ok=False, error="stale")) + "\\n")   # not ours
    out.write("garbage\\n")
    out.write(json.dumps(dict(id=q["id"], ok=True, stem=q["ms"], seq=q["id"])) + "\\n"); out.flush()
""")
    c = hs.ArchiveClient("x", tmp_path / "a.gsiv", None, {}, cmd=[sys.executable, str(fake)], timeout=10)
    assert [c.add(f"m{k}")["stem"] for k in range(3)] == ["m0", "m1", "m2"]
    c.close()


def _fake_imager_pkg(tmp_path):
    d = tmp_path / "old_imager" / "ttcalx_imager"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    (d / "msio.py").write_text("def load_samples(*a): pass\n")       # the production imager: no gsiv
    return str(d.parent)


def test_probe_and_serve_fail_fast_without_gsiv(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_SRC), _fake_imager_pkg(tmp_path)]))
    with pytest.raises(hs.GSIVError, match="no usable ttcalx_imager.gsiv"):
        hs.probe_imager(sys.executable, env)
    with pytest.raises(hs.GSIVError, match="did not start.*gsiv"):
        hs.ArchiveClient(sys.executable, tmp_path / "a.gsiv", None, {}, env=env, timeout=30)
    if IMAGER_SRC:
        assert hs.probe_imager(sys.executable, _serve_env()).startswith(IMAGER_SRC)


def test_worker_fails_fast_at_batch_start_without_gsiv(tmp_path):
    from gpu_subband_imaging import worker as wmod
    cfg = _lst_cfg(tmp_path, tmp_path / "venv" / "bin" / "ttcalx-image")
    cfg.pipeline.imaging = SimpleNamespace(gpu_python=sys.executable)
    cfg.pipeline.hour_imaging = HourImagingParams(env={"PYTHONPATH": _fake_imager_pkg(tmp_path)})
    w = _bare_worker(wmod, cfg)
    with pytest.raises(RuntimeError, match="no usable ttcalx_imager.gsiv"):
        w.run_batch(["/x/20261031_101007_55MHz.ms.tar"])


def test_writer_lock(tmp_path):
    p = tmp_path / "x.gsiv"
    r, _ = _rec()
    w = hs.RecordWriter(p, dict(mode="iv"))
    with pytest.raises(hs.GSIVError, match="being written by pid"):
        hs.RecordWriter(p, dict(mode="iv"))
    w.close()
    assert not Path(f"{p}.lock").exists()
    Path(f"{p}.lock").write_text(json.dumps(dict(pid=2 ** 22 + 12345, host=os.uname().nodename, t=0)))
    w = hs.RecordWriter(p, dict(mode="iv"))                     # a dead writer's lock is taken over
    w.append(r.meta, r.arrays)
    w.close()
    Path(f"{p}.lock").write_text(json.dumps(dict(pid=1, host="elsewhere", t=0)))
    with pytest.raises(hs.GSIVError, match="elsewhere"):
        hs.RecordWriter(p, dict(mode="iv"))
    os.utime(f"{p}.lock", (0, 0))                               # untouched for long: stale
    hs.RecordWriter(p, dict(mode="iv")).close()


def test_decode_bounds_checked_without_crc(tmp_path):
    r, _ = _rec()
    hdr, pieces, nbytes, _ = hs.encode_block(hs.MAGIC_REC, 1, r.meta, r.arrays)
    blk = bytearray(hdr + b"".join(bytes(x) for x in pieces))
    meta_len = hs.decode_header(bytes(blk[:64]))["meta_len"]
    js = bytes(blk[64:64 + meta_len])
    assert b'"offset":0,' in js
    blk[64:64 + meta_len] = js.replace(b'"offset":0,', b'"offset":9,', 1)     # same length, misplaced
    with pytest.raises(hs.GSIVError, match="outside"):
        hs.decode_block(bytes(blk), 0)
    with pytest.raises(hs.GSIVError, match="buffer"):
        hs.decode_block(bytes(blk[:nbytes // 2]), 0, hs.decode_header(bytes(blk[:64])))


def test_newest_record_wins_within_and_across_files(tmp_path):
    a, b = tmp_path / "b000.gsiv", tmp_path / "b001.gsiv"
    r_old, _ = _rec(seed=1, stem="S1")
    r_new, _ = _rec(seed=2, stem="S1")
    r_mid, _ = _rec(seed=3, stem="S1")
    wb = hs.RecordWriter(b, dict(mode="iv"))
    wb.append(r_old.meta, r_old.arrays)                         # oldest, in the later-named file
    wb.close()
    wa = hs.RecordWriter(a, dict(mode="iv"))
    wa.append(r_mid.meta, r_mid.arrays)
    wa.append(r_new.meta, r_new.arrays)                         # newest: second record of a
    wa.close()
    for files in ([a, b], [b, a]):
        inv = hs.archive_inventory(files)
        assert inv["stems"] == ["S1"] and inv["files"]["S1"] == str(a) and len(inv["superseded"]) == 2
    if IMAGER_SRC:
        from ttcalx_imager import gsiv
        for spec in (f"{a},{b}", f"{b},{a}"):
            (s,) = gsiv.load_archive(spec)
            assert _eq(s["I"], r_new.arrays["I"]) and s.source == str(a)


def test_band_and_freqs_checked(tmp_path):
    p = tmp_path / "x.gsiv"
    r55, _ = _rec(seed=0)
    w = hs.RecordWriter(p, dict(mode="iv", band=82), {k: r55.arrays[k] for k in hs.SHARED_ARRAYS + hs.CHECK_ARRAYS})
    w.append(r55.meta, r55.arrays)
    other = dict(r55.arrays, freqs=r55.arrays["freqs"] + 1e6)   # another band's channels
    with pytest.raises(hs.GSIVError, match="another band"):
        w.append(r55.meta, other)
    w.close()
    with pytest.raises(hs.GSIVError, match="band"):
        hs.RecordWriter(p, dict(mode="iv", band=55))
    with pytest.raises(hs.GSIVError, match="another band"):
        hs.RecordWriter(p, dict(mode="iv", band=82), dict(freqs=other["freqs"]))
    assert any("band 82 != 55" in x for x in hs.archive_inventory([p], band=55)["problems"])
    # a structurally valid record of another band written past the writer: refused on load
    sc = hs.scan(p)
    hdr, pieces, nbytes, crc = hs.encode_block(hs.MAGIC_REC, 2, dict(r55.meta, stem="X"), other)
    with open(p, "ab") as fh:
        fh.write(hdr); [fh.write(x) for x in pieces]
    with open(hs.idx_path(p), "a") as fh:
        fh.write(json.dumps(hs._entry(dict(r55.meta, stem="X"), other, 2, sc.end, nbytes, crc)) + "\n")
    assert hs.scan(p).clean
    with pytest.raises(hs.GSIVError, match="another band"):
        hs.read_records(p)
    if IMAGER_SRC:
        from ttcalx_imager import gsiv
        with pytest.raises(gsiv.GSIVError, match="another band"):
            gsiv.load_archive(p)
        rep = {}
        assert [s["stem"] for s in gsiv.load_archive(p, strict=False, report=rep)] == [r55.stem]
        assert rep["problems"]


# ----------------------------------------------------------------------------- config / worker / hour job
def test_v3_archive_config_validation(tmp_path, monkeypatch):
    assert v3_params(SimpleNamespace()) == V3Params()
    with pytest.raises(ConfigError, match="archive_root"):
        V3Params(archive="iv")
    with pytest.raises(ConfigError, match="not implemented"):
        V3Params(archive="full", archive_root="/x")
    with pytest.raises(ConfigError, match="hour_source=archive"):
        V3Params(hour_source="archive")
    with pytest.raises(ConfigError, match="write_kept_ms"):
        V3Params(write_kept_ms=False)
    with pytest.raises(ConfigError, match="cannot replay"):      # v2 worker: no replay of a lost record
        V3Params(archive="iv", archive_root="/x", hour_source="archive", write_kept_ms=False)
    V3Params(archive="iv", archive_root="/x", hour_source="archive", write_kept_ms=False, prep_inmem="on",
             peel_io="shm", snapshot_io="shm")
    monkeypatch.chdir(tmp_path)
    assert V3Params(archive="iv", archive_root="rel/arch").archive_root == str(tmp_path / "rel" / "arch")


def test_archive_hour_source_needs_lst_layout(tmp_path):
    from dataclasses import replace
    from gpu_subband_imaging.config import Config
    cfg = Config.load(Path(__file__).resolve().parents[1] / "examples" / "config")
    v = V3Params(archive="iv", archive_root=str(tmp_path), hour_source="archive")
    with pytest.raises(ConfigError, match="layout=lst"):
        replace(cfg.pipeline, layout="utc", v3=v)
    replace(cfg.pipeline, layout="lst", run_label="Run_T", v3=v)


def _lst_cfg(tmp_path, ttcalx_image, keep=False, **v3):
    v = V3Params(archive="iv", archive_root=str(tmp_path / "arch"), hour_source="archive", **v3)
    pl = SimpleNamespace(date="2026-10-31", output_root=str(tmp_path / "out"), layout="lst", run_label="Run_T",
                         keep_ms_root=str(tmp_path / "keep") if keep else None, keep_ms_local=None, v3=v,
                         sun_cut=True, snapshot_phase_centre="lst",
                         hour_imaging=HourImagingParams(ttcalx_image=str(ttcalx_image), nproc=2, delete_kept_ms=True),
                         movie=SimpleNamespace(enabled=False), cal=SimpleNamespace(bp="b", xy="x", sources="s",
                                                                                   rfi_sources=None),
                         env=SimpleNamespace(ttcalx=str(tmp_path)), imaging=SimpleNamespace(),
                         peel_for_band=lambda b: SimpleNamespace(rfi=False))
    return SimpleNamespace(pipeline=pl, cluster=SimpleNamespace(work_root=str(tmp_path / "work")))


def _bare_worker(wmod, cfg, batch=2):
    w = wmod.Worker.__new__(wmod.Worker)
    # LST hours given (lst.lst_hours needs astropy IERS data, see test_lst_layout)
    w.cfg, w.band, w.batch = cfg, 55, batch
    w._lst = {f"20261031_1010{k}7_55MHz": 4 for k in range(9)}
    w._archives, w._archive_lock, w._archive_failed = {}, __import__("threading").Lock(), False
    w._qa, w._gpu_imager = [], None
    w.work = Path(cfg.cluster.work_root) / "b55_2"
    w.geom = None
    return w


def test_archive_paths(tmp_path):
    cfg = _lst_cfg(tmp_path, "/v/ttcalx-image")
    p = hs.archive_path(cfg, 55, "04h", 3)
    assert p == tmp_path / "arch" / "55MHz" / "2026-10-31" / "Run_T" / "04h" / "55MHz-2026-10-31-04h-peeled-b003.gsiv"
    assert hs.archive_files(cfg, 55, "04h") == []
    assert hs.expected_path(p).name == "55MHz-2026-10-31-04h-peeled-b003.expected.jsonl"


# --- the archive hour job: verified archive, fallback to the kept MSes, never a reduced set
FAKE_IMAGER = """#!{py}
import json, os, sys
sys.path.insert(0, {src!r})
from gpu_subband_imaging.v3 import hourstore as hs
a = sys.argv
if "--help" in a:
    print("usage: ... --archive ARCHIVE ...")
    sys.exit(0)
json.dump(a, open({log!r}, "a")); open({log!r}, "a").write("\\n")
out = a[a.index("--out") + 1]
if "--archive" in a:            # what ttcalx_imager.gsiv.load_archive does: strict, newest per stem
    files = [l.strip() for l in open(a[a.index("--archive") + 1][1:]) if l.strip()]
    try:
        per = [(f, hs.read_blocks(f)[2]) for f in files]
    except hs.GSIVError as e:
        print("GSIVError", e); sys.exit(1)
    stems = sorted(hs.newest(per)[0])
else:
    stems = sorted(os.path.basename(l.strip())[:-3] for l in open(a[2][1:]) if l.strip())
stems = stems[:int(os.environ.get("FAKE_LOAD_ONLY", len(stems)))]
open(out + "-I-Deep-Taper-Robust-0-image.fits", "w").write("x")
json.dump(dict(timing=dict(TOTAL=1.0), n_loaded=len(stems), rejected=[], phase_centre=[1, 2], products={{}},
               qa=[dict(stem=s) for s in stems]), open(out + "-summary.json", "w"))
"""


def _hour_setup(tmp_path, monkeypatch, n=8, n_records=None, kept=True, expected=True, archive=True, env=None):
    log = tmp_path / "imager_calls.jsonl"
    fake = tmp_path / "fake-ttcalx-image"
    fake.write_text(FAKE_IMAGER.format(py=sys.executable, src=str(REPO_SRC), log=str(log)))
    fake.chmod(0o755)
    cfg = _lst_cfg(tmp_path, fake, keep=True)
    if env:
        cfg.pipeline.hour_imaging = HourImagingParams(ttcalx_image=str(fake), nproc=2, delete_kept_ms=True, env=env)
    monkeypatch.setattr("gpu_subband_imaging.metadata.software_versions", lambda cfg: {})
    stems = [f"20261031_10{10 + k:02d}07_55MHz" for k in range(n)]      # valid times for the product names
    f = hs.archive_path(cfg, 55, "04h", 0)
    if archive:
        r0, _ = _rec(seed=0)
        w = hs.RecordWriter(f, dict(mode="iv", band=55, phase_centre="04h30m00s 37d12m57.057s"),
                            {k: r0.arrays[k] for k in hs.SHARED_ARRAYS + hs.CHECK_ARRAYS})
        for k, s in enumerate(stems[::-1][:n if n_records is None else n_records]):
            r, _ = _rec(seed=k, stem=s)
            w.append(r.meta, r.arrays)
        w.close()
    if expected:
        for s in stems:
            hs.add_expected(f, s, batch=0)
    kd = Path(cfg.pipeline.keep_ms_root) / "55MHz" / "2026-10-31" / "04h"
    if kept:
        for s in stems:
            (kd / f"{s}.ms").mkdir(parents=True)
    return cfg, stems, f, kd, log


def _calls(log):
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_run_band_archive_command(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, n=3, kept=False)
    bdir = hour_lst._band_dir(cfg, 55, 4)
    st = bdir / "QA" / ".snapshot_stats"
    st.mkdir(parents=True)
    rows = [dict(ts=s[:15], rms=r, peak=1.0, file=f"{s}.fits") for s, r in zip(stems * 4, [1.0] * 11 + [50.0])]
    rows = [dict(r, ts=f"20261031_1{k:05d}") for k, r in enumerate(rows)]
    rows[-1]["ts"] = stems[1][:15]                            # the outlier is the middle archived stamp
    (st / "b0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rec = hour_lst.run_band(cfg, 55, "04h", gpu=0)
    (argv,) = _calls(log)
    assert argv[1] == "hour" and argv[2] == "--archive" and argv[3].startswith("@")
    assert "--lst-hour" in argv and argv[argv.index("--lst-hour") + 1] == "4" and "--exclude" not in argv
    xs = argv[argv.index("--exclude-stems") + 1]
    assert xs.startswith("@") and Path(xs[1:]).name == "exclude-stems.txt"
    assert rec["n_records"] == 3 and rec["archive"] == [str(f)] and rec["hour_source"] == "archive"
    assert rec["status"] == "ok" and rec["n_ms"] == 0
    excluded = json.loads((bdir / "QA" / "55MHz-hour-timing.json").read_text())["qa"]["excluded_from_hour"]
    assert excluded == 1
    prov = json.loads((bdir / "provenance.json").read_text())
    assert prov["v3"]["used"] == "archive" and prov["v3"]["files"][0]["path"] == str(f)
    assert list((bdir / "I" / "deep").glob("55MHz-I-Deep-Taper-Robust-0-image-*.fits"))


def test_run_band_archive_excludes_the_rejected_stem(tmp_path, monkeypatch):
    """The stem written to --exclude-stems is the QA-rejected one (the work dir is removed after
    the job, so the fake imager records the file's content)."""
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, n=3, kept=False)
    seen = tmp_path / "xs.txt"
    fake = Path(cfg.pipeline.hour_imaging.ttcalx_image)
    fake.write_text(fake.read_text().replace(
        'out = a[a.index("--out") + 1]',
        f'out = a[a.index("--out") + 1]\nif "--exclude-stems" in a: open({str(seen)!r}, "w").write('
        f'open(a[a.index("--exclude-stems") + 1][1:]).read())'))
    bdir = hour_lst._band_dir(cfg, 55, 4)
    st = bdir / "QA" / ".snapshot_stats"
    st.mkdir(parents=True)
    rows = [dict(ts=f"20261031_1{k:05d}", rms=1.0, peak=1.0, file="x") for k in range(11)]
    rows.append(dict(ts=stems[1][:15], rms=50.0, peak=1.0, file="x"))
    (st / "b0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert seen.read_text().split() == [stems[1]]


def test_run_band_clean_archive_deletes_only_imaged_kept_ms(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch)
    (kd.parent / "05h").mkdir()
    rec = hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert rec["hour_source"] == "archive" and rec["status"] == "ok" and _calls(log)[0][2] == "--archive"
    assert not kd.exists() and (kd.parent / "05h").exists()


def test_run_band_nothing_at_all_is_no_data(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, kept=False, expected=False, archive=False)
    rec = hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert rec["status"] == "no-data" and not _calls(log)


@pytest.mark.parametrize("case", ["missing", "torn", "partial", "no_manifest_extra", "bitflip"])
def test_run_band_unusable_archive_falls_back_to_kept_ms(tmp_path, monkeypatch, case):
    kw = dict(missing=dict(archive=False), torn={}, partial=dict(n_records=4), no_manifest_extra={},
              bitflip={})[case]
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, **kw)
    if case == "bitflip":                                   # only a crc check sees it
        e = hs.scan(f).entries[2]
        with open(f, "r+b") as fh:
            fh.seek(e["offset"] + e["nbytes"] - 300)
            b = fh.read(1); fh.seek(-1, 1); fh.write(bytes([b[0] ^ 4]))
    if case == "torn":
        e = hs.scan(f).entries[-1]
        with open(f, "r+b") as fh:
            fh.truncate(e["offset"] + 100)
    if case == "no_manifest_extra":                       # a stamp archived as FITS but not as a record
        (kd / "20261031_105007_55MHz.ms").mkdir()
        fits = hour_lst._band_dir(cfg, 55, 4) / "snapshots_clean"
        fits.mkdir(parents=True)
        (fits / "55MHz-Clean-Snapshot-20261031_105007-image.fits.fz").write_text("x")
    rec = hour_lst.run_band(cfg, 55, "04h", gpu=0)
    (argv,) = _calls(log)
    assert rec["hour_source"] == "kept_ms" and rec["archive_problems"] and argv[2].startswith("@")
    assert "--archive" not in argv and rec["status"] == "ok"
    assert not kd.exists()                                  # all of them imaged (from the MSes)


@pytest.mark.parametrize("case", ["missing", "torn", "partial"])
def test_run_band_unusable_archive_without_kept_ms_fails_loudly(tmp_path, monkeypatch, case):
    kw = dict(missing=dict(archive=False), torn={}, partial=dict(n_records=4))[case]
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, kept=False, **kw)
    if case == "torn":
        e = hs.scan(f).entries[-1]
        with open(f, "r+b") as fh:
            fh.truncate(e["offset"] + 100)
    with pytest.raises(RuntimeError, match="unusable"):
        hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert not _calls(log)
    t = json.loads((hour_lst._band_dir(cfg, 55, 4) / "QA" / "55MHz-hour-timing.json").read_text())
    assert t["status"] == "failed" and t["archive_problems"]


def test_run_band_partial_kept_ms_never_fall_back_to_a_reduced_set(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, n_records=4)
    for s in stems[:3]:
        os.rmdir(kd / f"{s}.ms")                            # kept MSes do not cover the expected set
    with pytest.raises(RuntimeError, match="unusable"):
        hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert sorted(p.name[:-3] for p in kd.iterdir()) == stems[3:]


def test_run_band_imager_loading_fewer_deletes_nothing(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, env={"FAKE_LOAD_ONLY": "4"})
    with pytest.raises(RuntimeError, match="rc=-3"):
        hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert sorted(p.name[:-3] for p in kd.iterdir()) == stems


def _old_hour_lst():
    """hour_lst.py of 13405ce (tests/data/hour_lst_13405ce.py, verbatim) as a module of the package."""
    import importlib.util
    path = Path(__file__).parent / "data" / "hour_lst_13405ce.py"
    spec = importlib.util.spec_from_file_location("gpu_subband_imaging._hour_lst_13405ce", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*")) if Path(root).exists() else []


@pytest.mark.parametrize("case", ["qa_rejection", "no_data", "no_summary", "load_fewer"])
def test_defaults_delete_exactly_what_13405ce_deleted(tmp_path, monkeypatch, case):
    """Defaults (hour_source=kept_ms): new and old run_band on identical fixtures leave identical
    kept trees -- QA-rejected MSes, MSes the imager did not load, hidden leftovers included."""
    old = _old_hour_lst()
    left = {}
    for tag, mod in (("old", old), ("new", hour_lst)):
        root = tmp_path / tag
        root.mkdir()
        env = {"FAKE_LOAD_ONLY": "5"} if case == "load_fewer" else None
        cfg, stems, f, kd, log = _hour_setup(root, monkeypatch, archive=False, expected=False,
                                             kept=case != "no_data", env=env)
        cfg.pipeline.v3 = V3Params()                             # defaults
        kd.mkdir(parents=True, exist_ok=True)
        (kd / f".{stems[0]}.ms.tmp123").mkdir()                  # a leftover of an interrupted keep_ms
        (kd.parent / "05h" / "x.ms").mkdir(parents=True)         # another hour: untouched
        if case == "no_summary":                                 # rc 0 but no summary JSON
            fake = Path(cfg.pipeline.hour_imaging.ttcalx_image)
            fake.write_text(fake.read_text().replace('open(out + "-summary.json", "w"))', 'open(os.devnull, "w"))'))
        bdir = hour_lst._band_dir(cfg, 55, 4)
        st = bdir / "QA" / ".snapshot_stats"
        st.mkdir(parents=True)
        rows = [dict(ts=s[:15], rms=1.0, peak=1.0, file="x") for s in stems] + \
               [dict(ts=f"20261031_1{k:05d}", rms=1.0, peak=1.0, file="x") for k in range(4)]
        rows[2]["rms"] = 50.0                                    # stems[2] is QA-rejected (excluded)
        (st / "b0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        rec = mod.run_band(cfg, 55, "04h", gpu=0)
        if case == "qa_rejection":
            assert rec["qa"]["excluded_from_hour"] == 1
        left[tag] = (_tree(Path(cfg.pipeline.keep_ms_root)), rec["status"])
    assert left["new"] == left["old"]
    if case == "no_data":
        assert left["new"][1] == "no-data" and any(".ms.tmp123" in p for p in left["new"][0])
    else:
        assert left["new"][0] == ["55MHz", "55MHz/2026-10-31", "55MHz/2026-10-31/05h", "55MHz/2026-10-31/05h/x.ms"]


def test_archive_mode_imager_loading_fewer_deletes_nothing_it_did_not_image(tmp_path, monkeypatch):
    cfg, stems, f, kd, log = _hour_setup(tmp_path, monkeypatch, archive=False, env={"FAKE_LOAD_ONLY": "5"})
    with pytest.raises(RuntimeError, match="rc=-3"):          # archive mode, fallback to MSes, fewer loaded
        hour_lst.run_band(cfg, 55, "04h", gpu=0)
    assert len(list(kd.iterdir())) == 8


def test_worker_image_archive_order_and_record(tmp_path, monkeypatch):
    """_image_archive: FITS archived -> expected list -> keep_ms -> record of the kept MS."""
    from gpu_subband_imaging import worker as wmod
    calls = []
    cfg = _lst_cfg(tmp_path, tmp_path / "venv" / "bin" / "ttcalx-image", keep=True)
    cfg.pipeline.imaging = SimpleNamespace(gpu_python=None)
    cfg.pipeline.snapshot_phase_centre = "zenith"
    cfg.pipeline.movie = SimpleNamespace(enabled=False, fixed_vmax=20.0)
    cfg.pipeline.keep_ms_local_min_free_gb = 0
    w = _bare_worker(wmod, cfg)
    stamp = "20261031_101007_55MHz"

    class FakeClient:
        alive = True

        def __init__(self, python, out, pc, hdr, env=None, **kw):
            calls.append(("start", python, str(out), pc, hdr["group"], hdr["cal_bp"],
                          env["PYTHONPATH"].split(os.pathsep)[0], env["CUDA_VISIBLE_DEVICES"]))
            self.info = dict(pid=1, n=0)

        def add(self, ms, stem=None, path=None):
            fits = w._fits_path_for_stamp(stamp)
            exp = hs.expected_stems(cfg, 55, "04h") or set()
            calls.append(("add", str(ms), stem, path, fits.exists(), Path(ms).is_dir(), stem in exp))
            return dict(ok=True, nbytes=10, n_kept=5)

        def close(self):
            calls.append(("close",))

    class Imager:
        def image(self, src, prefix, geom, imaging, timings=None):
            out = [Path(f"{prefix}-I-image.fits"), Path(f"{prefix}-V-image.fits")]
            for p in out:
                p.write_text("fits")
            return out
    monkeypatch.setattr(hs, "ArchiveClient", FakeClient)
    monkeypatch.setattr(wmod.image, "v_stats", lambda f: dict(rms=1.0, peak=2.0))

    def fpack(f):
        fz = Path(f"{f}.fz"); f.rename(fz); return fz
    monkeypatch.setattr(wmod.compress, "fpack", fpack)
    w._gpu_imager = Imager()
    avg = w.work / stamp / f"{stamp}_avg.ms"
    avg.mkdir(parents=True)
    (avg / "table.dat").write_text("x")
    w._image_archive(stamp, avg)
    w._close_archives()
    kept = w.kept_ms_path(stamp)
    assert kept.is_dir() and not avg.exists()
    start, add = calls[0], calls[1]
    assert start[1] == str(tmp_path / "venv" / "bin" / "python")
    assert start[2].endswith("55MHz-2026-10-31-04h-peeled-b002.gsiv")
    assert start[3:6] == ("04h30m00s 37d12m57.057s", "04h", "b") and start[6] == str(REPO_SRC) and start[7] == ""
    # the record is made from the kept MS, after the FITS are in place and the stem is listed
    assert add[1:] == (str(kept), stamp, str(kept), True, True, True)
    assert calls[-1] == ("close",)

    class Failing(FakeClient):
        def add(self, ms, stem=None, path=None):
            raise hs.GSIVError("disk full")
    monkeypatch.setattr(hs, "ArchiveClient", Failing)
    s2 = "20261031_101017_55MHz"
    with pytest.raises(hs.GSIVError):                          # hour_source=archive: the snapshot fails
        w._archive_record(s2, tmp_path / "avg3.ms")
    assert not w._archives                                     # the failed writer is dropped
    w.cfg.pipeline.v3 = V3Params(archive="iv", archive_root=str(tmp_path / "arch"))
    w._archive_record("20261031_101027_55MHz", tmp_path / "avg4.ms")   # kept_ms hour source: logged
    w._archive_record("20261031_101037_55MHz", tmp_path / "avg5.ms")
    assert w._archive_failed and not w._archives
