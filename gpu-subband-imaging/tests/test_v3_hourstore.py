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
    sys.path.append(IMAGER_SRC)
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
        assert g.meta == r.meta
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
    assert [r.stem for r in hs.read_records(p)[2]] == [recs[0].stem, recs[1].stem]
    st = hs.HourStore(p, backend="memmap")              # opening a store does not modify the file
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
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_SRC), IMAGER_SRC]), CUDA_VISIBLE_DEVICES="")
    out = tmp_path / "arch" / "55MHz-x-04h-peeled-b000.gsiv"
    hdr = hs.header_for(55, "2026-10-31", "04h", None, "Run_T", batch=0)
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=env, timeout=120)
    rs = [c.add(m) for m in ms]
    assert all(r["ok"] for r in rs) and [r["seq"] for r in rs] == [1, 2, 3]
    assert c.add(ms[1])["skipped"] == "duplicate"
    with pytest.raises(hs.GSIVError):
        c.add(tmp_path / "missing.ms")                  # an error reply; the writer keeps serving
    c.close()
    c = hs.ArchiveClient(sys.executable, out, None, hdr, env=env, timeout=120)   # rerun: resumes
    assert c.info["n"] == 3
    c.close()
    sc = hs.scan(out, verify=True)
    assert sc.clean and sc.stems == [m.name[:-3] for m in ms] and sc.header_meta["group"] == "04h"
    for m, s in zip(ms, gsiv.load_archive(out)):
        ref = load_samples(str(m))
        for k in ("u", "v", "w", "nw", "I", "V", "R", "freqs"):
            assert _eq(s[k], ref[k]), k


# ----------------------------------------------------------------------------- config / worker / hour job
def test_v3_archive_config_validation():
    assert v3_params(SimpleNamespace()) == V3Params()
    with pytest.raises(ConfigError, match="archive_root"):
        V3Params(archive="iv")
    with pytest.raises(ConfigError, match="not implemented"):
        V3Params(archive="full", archive_root="/x")
    with pytest.raises(ConfigError, match="hour_source=archive"):
        V3Params(hour_source="archive")
    with pytest.raises(ConfigError, match="write_kept_ms"):
        V3Params(write_kept_ms=False)
    V3Params(archive="iv", archive_root="/x", hour_source="archive", write_kept_ms=False)


def _lst_cfg(tmp_path, ttcalx_image, **v3):
    v = V3Params(archive="iv", archive_root=str(tmp_path / "arch"), hour_source="archive", **v3)
    pl = SimpleNamespace(date="2026-10-31", output_root=str(tmp_path / "out"), layout="lst", run_label="Run_T",
                         keep_ms_root=None, keep_ms_local=None, v3=v, sun_cut=True, snapshot_phase_centre="lst",
                         hour_imaging=HourImagingParams(ttcalx_image=str(ttcalx_image), nproc=2, delete_kept_ms=True),
                         movie=SimpleNamespace(enabled=False), cal=SimpleNamespace(bp="b", xy="x", sources="s",
                                                                                   rfi_sources=None),
                         env=SimpleNamespace(ttcalx=str(tmp_path)), imaging=SimpleNamespace(),
                         peel_for_band=lambda b: SimpleNamespace(rfi=False))
    return SimpleNamespace(pipeline=pl, cluster=SimpleNamespace(work_root=str(tmp_path / "work")))


def test_archive_paths(tmp_path):
    cfg = _lst_cfg(tmp_path, "/v/ttcalx-image")
    p = hs.archive_path(cfg, 55, "04h", 3)
    assert p == tmp_path / "arch" / "55MHz" / "2026-10-31" / "Run_T" / "04h" / "55MHz-2026-10-31-04h-peeled-b003.gsiv"
    assert hs.archive_files(cfg, 55, "04h") == []


def test_run_band_archive_command(tmp_path, monkeypatch):
    fake = tmp_path / "fake-ttcalx-image"
    fake.write_text(f"""#!{sys.executable}
import json, sys
a = sys.argv
json.dump(a, open({str(tmp_path / 'argv.json')!r}, "w"))
out = a[a.index("--out") + 1]
open(out + "-I-Deep-Taper-Robust-0-image.fits", "w").write("x")
json.dump(dict(timing=dict(TOTAL=1.0), n_loaded=3, rejected=[1], phase_centre=[1, 2], products={{}}),
          open(out + "-summary.json", "w"))
""")
    fake.chmod(0o755)
    cfg = _lst_cfg(tmp_path, fake)
    monkeypatch.setattr("gpu_subband_imaging.metadata.software_versions", lambda cfg: {})
    stems = [f"20261031_1010{k}7_55MHz" for k in range(3)]
    f = hs.archive_path(cfg, 55, "04h", 0)
    w = hs.RecordWriter(f, dict(mode="iv"))
    for k, s in enumerate(stems[::-1]):                       # archive order != time order
        r, _ = _rec(seed=k, stem=s)
        w.append(r.meta, r.arrays)
    w.close()
    bdir = hour_lst._band_dir(cfg, 55, 4)
    st = bdir / "QA" / ".snapshot_stats"
    st.mkdir(parents=True)
    rows = [dict(ts=s[:15], rms=r, peak=1.0, file=f"{s}.fits") for s, r in zip(stems * 4, [1.0] * 11 + [50.0])]
    rows = [dict(r, ts=f"20261031_1{k:05d}") for k, r in enumerate(rows)]
    rows[-1]["ts"] = stems[1][:15]                            # the outlier is the middle archived stamp
    (st / "b0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rec = hour_lst.run_band(cfg, 55, "04h", gpu=0)
    argv = json.load(open(tmp_path / "argv.json"))
    assert argv[1] == "hour" and argv[2] == "--archive" and argv[3].startswith("@")
    assert "--lst-hour" in argv and argv[argv.index("--lst-hour") + 1] == "4" and "--exclude" not in argv
    xs = argv[argv.index("--exclude-stems") + 1]
    assert xs.startswith("@") and rec["n_records"] == 3 and rec["archive"] == [str(f)]
    assert rec["status"] == "ok" and rec["n_ms"] == 0
    prov = json.loads((bdir / "provenance.json").read_text())
    assert prov["v3"]["hour_source"] == "archive" and prov["v3"]["files"][0]["path"] == str(f)
    assert list((bdir / "I" / "deep").glob("55MHz-I-Deep-Taper-Robust-0-image-*.fits"))


def test_worker_archive_hook(tmp_path, monkeypatch):
    """_image_archive appends the record before the FITS move; kept MSes follow write_kept_ms."""
    from gpu_subband_imaging import worker as wmod
    calls = []

    class FakeClient:
        def __init__(self, python, out, pc, hdr, env=None, **kw):
            calls.append(("start", python, str(out), pc, hdr["group"], env["PYTHONPATH"].split(os.pathsep)[0],
                          env["CUDA_VISIBLE_DEVICES"]))
            self.info = dict(pid=1, n=0)

        def add(self, ms):
            calls.append(("add", str(ms)))
            return dict(ok=True, nbytes=10, n_kept=5)

        def close(self):
            calls.append(("close",))
    monkeypatch.setattr(hs, "ArchiveClient", FakeClient)
    cfg = _lst_cfg(tmp_path, tmp_path / "venv" / "bin" / "ttcalx-image")
    cfg.pipeline.imaging = SimpleNamespace(gpu_python=None)
    w = wmod.Worker.__new__(wmod.Worker)
    w.cfg, w.band, w.batch, w._lst = cfg, 55, 2, {}
    w._archives, w._archive_lock = {}, __import__("threading").Lock()
    w._archive_record("20261031_101007_55MHz", tmp_path / "avg.ms")
    w._archive_record("20261031_101017_55MHz", tmp_path / "avg2.ms")
    w._close_archives()
    start = calls[0]
    assert start[1] == str(tmp_path / "venv" / "bin" / "python")
    assert start[2].endswith("55MHz-2026-10-31-04h-peeled-b002.gsiv")
    assert start[3] == "04h30m00s 37d12m57.057s" and start[4] == "04h"
    assert start[5] == str(REPO_SRC) and start[6] == ""
    assert [c[0] for c in calls] == ["start", "add", "add", "close"]

    class Failing(FakeClient):
        def add(self, ms):
            raise hs.GSIVError("disk full")
    monkeypatch.setattr(hs, "ArchiveClient", Failing)
    with pytest.raises(hs.GSIVError):                          # hour_source=archive: the snapshot fails
        w._archive_record("20261031_101027_55MHz", tmp_path / "avg3.ms")
    w._close_archives()
    w.cfg.pipeline.v3 = V3Params(archive="iv", archive_root=str(tmp_path / "arch"))
    w._archive_record("20261031_101037_55MHz", tmp_path / "avg4.ms")   # kept_ms hour source: only logged
    w._close_archives()
