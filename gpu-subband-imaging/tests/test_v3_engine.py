"""gsi v3 steps 4/6/7: workers/gsi_v3_engine.py and v3/engine_client.py without a GPU.

The engine runs as a real child process with its fake backend (GSI_V3_ENGINE_BACKEND=fake: tiny
FITS instead of the GPU imager); the records, the hour store (RAM, checkpoint file, spill), the
protocol (ids, deadlines, restarts) and the hour op's plumbing are the real ones.  Needs the
ttcalx_imager source (msio, gsiv, cli, rotate of the gsi v3 imager) as test_v3_hourstore.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gpu_subband_imaging.v3 import engine_client, hourstore, slots

from test_v3_hourstore import CASACORE, IMAGER_SRC
from test_v3_slots import make_snap

REPO = Path(__file__).resolve().parents[1]
ENGINE = str(REPO / "workers" / "gsi_v3_engine.py")
pytestmark = pytest.mark.skipif(IMAGER_SRC is None, reason="ttcalx_imager source (gsi v3) not found")


def _has_engine_imager():
    if IMAGER_SRC is None:
        return False
    return "def run_hour_snaps" in (Path(IMAGER_SRC) / "ttcalx_imager" / "cli.py").read_text()


needs_v3_imager = pytest.mark.skipif(not _has_engine_imager(), reason="imager without run_hour_snaps")


def _env(**kw):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), IMAGER_SRC or ""]),
               GSI_V3_ENGINE_BACKEND="fake", CUDA_VISIBLE_DEVICES="")
    for k in ("GSI_V3_FAKE_SLEEP", "GSI_V3_FAKE_FAIL", "GSI_V3_FAKE_DIE", "GSI_V3_FAKE_FATAL", "GSI_V3_FAKE_HOUR"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _client(**kw):
    on_restart = kw.pop("on_restart", None)
    return engine_client.EngineClient(sys.executable, ENGINE, _env(**kw), start_timeout=120, kill_wait=10,
                                      on_restart=on_restart)


def _peeled_slot(path, seed=0, stem="20261031_101709_23MHz", state=slots.PEELED):
    s = make_snap(seed=seed, stem=stem)
    s.data[0, 0, 0] = 1 + 1j                                  # no NaN (records hold real values)
    slots.write_slot(path, s)
    slots.set_state(path, state)
    return s


def _req(slot, prefix, stem, **kw):
    d = dict(slot=str(slot), prefix=str(prefix), pixels=16, scale=1.0, niter=10, phase_centre=None,
             record_phase_centre=None, image=True, record=True, stem=stem, path=f"/kept/{stem}.ms")
    d.update(kw)
    return d


pytest.importorskip("astropy")


@needs_v3_imager
def test_image_slot_and_record(tmp_path):
    c = _client()
    try:
        assert c.info["backend"] == "fake" and c.info["status"] == "ready"
        r = c.request("store_open", path=None, backend="ram", header=None)
        assert r["n"] == 0 and r["path"] is None
        snap = _peeled_slot(tmp_path / "s0")
        r = c.request("image_slot", timeout=60, **_req(tmp_path / "s0", tmp_path / "img_a", "20261031_101709_23MHz"))
        assert r["status"] == "ok"
        assert sorted(r["image"]["files"]) == [str(tmp_path / "img_a-I-image.fits"), str(tmp_path / "img_a-V-image.fits")]
        assert all(Path(f).exists() for f in r["image"]["files"])
        assert r["record"]["stem"] == "20261031_101709_23MHz" and r["record"]["n_kept"] > 0
        # the engine's record is the in-process records_from_snap (the imager's expressions)
        ref = hourstore.records_from_snap(snap, stem="20261031_101709_23MHz", path="/kept/20261031_101709_23MHz.ms")
        assert r["record"]["n_kept"] == ref.n and r["record"]["nbytes"] == ref.nbytes()
        info = c.request("info")
        assert info["store"]["n"] == 1 and info["n_images"] == 1
    finally:
        c.close()
    assert not c.alive


@needs_v3_imager
def test_record_only_and_state_checks(tmp_path):
    c = _client()
    try:
        c.request("store_open", path=None, backend="ram")
        _peeled_slot(tmp_path / "s0", state=slots.PREPPED)
        with pytest.raises(engine_client.EngineError, match="PREPPED, expected PEELED"):
            c.request("image_slot", **_req(tmp_path / "s0", tmp_path / "x", "20261031_101709_23MHz"))
        _peeled_slot(tmp_path / "s1")
        r = c.request("image_slot", **_req(tmp_path / "s1", tmp_path / "y", "20261031_101709_23MHz", image=False))
        assert "image" not in r and r["record"]["n_kept"] > 0 and not (tmp_path / "y-I-image.fits").exists()
        with pytest.raises(engine_client.EngineError, match="holds"):
            c.request("image_slot", **_req(tmp_path / "s1", tmp_path / "z", "20261031_999999_23MHz"))
        assert c.alive and c.restarts == 0
    finally:
        c.close()


@needs_v3_imager
def test_checkpoint_resume_and_store_header(tmp_path):
    p = tmp_path / "ck" / "b.gsiv"
    hdr = hourstore.header_for(23, "2026-10-31", "05h", None, "Run_x", batch=0)
    c = _client()
    try:
        c.request("store_open", path=str(p), backend="ram", header=hdr)
        for k in range(3):
            _peeled_slot(tmp_path / f"s{k}", seed=k, stem=f"20261031_10170{k}_23MHz")
            c.request("image_slot", **_req(tmp_path / f"s{k}", tmp_path / f"i{k}", f"20261031_10170{k}_23MHz"))
    finally:
        c.close()
    sc = hourstore.scan(p, verify=True)
    assert sc.clean and [e["stem"] for e in sc.entries] == [f"20261031_10170{k}_23MHz" for k in range(3)]
    assert sc.header_meta["band"] == 23 and sc.header_meta["imager"] != ""
    # a torn tail (engine killed mid-append) is repaired at open; the valid records come back
    with open(p, "ab") as fh:
        fh.write(b"GSIVREC1" + b"\1" * 1000)
    c = _client()
    try:
        r = c.request("store_open", path=str(p), backend="memmap", header=hdr)
        assert r["n"] == 3 and r["stems"] == [f"20261031_10170{k}_23MHz" for k in range(3)] and r["backend"] == "memmap"
        # another band is refused on resume
        with pytest.raises(engine_client.EngineError, match="cannot resume"):
            c.request("store_open", path=str(p), backend="ram",
                      header=hourstore.header_for(55, "2026-10-31", "05h", None, "Run_x"))
            _peeled_slot(tmp_path / "s9", seed=9, stem="20261031_101799_23MHz")
            c.request("image_slot", **_req(tmp_path / "s9", tmp_path / "i9", "20261031_101799_23MHz"))
    finally:
        c.close()


@needs_v3_imager
def test_spill_switches_to_memmap_with_the_same_records(tmp_path):
    c = _client()
    try:
        c.request("store_open", path=None, backend="ram")
        for k in range(2):
            _peeled_slot(tmp_path / f"s{k}", seed=k, stem=f"20261031_10170{k}_23MHz")
            c.request("image_slot", **_req(tmp_path / f"s{k}", tmp_path / f"i{k}", f"20261031_10170{k}_23MHz"))
        r = c.request("spill", path=str(tmp_path / "sp" / "x.gsiv"))
        assert r["before"] == "ram" and r["backend"] == "memmap" and r["n"] == 2
        _peeled_slot(tmp_path / "s2", seed=2, stem="20261031_101702_23MHz")
        c.request("image_slot", **_req(tmp_path / "s2", tmp_path / "i2", "20261031_101702_23MHz"))
        assert c.request("info")["store"]["n"] == 3
    finally:
        c.close()
    recs = hourstore.read_records(tmp_path / "sp" / "x.gsiv")[2]
    assert [r.stem for r in recs] == [f"20261031_10170{k}_23MHz" for k in range(3)]
    ref = hourstore.records_from_snap(_peeled_slot(tmp_path / "t", seed=1, stem="20261031_101701_23MHz"),
                                      stem="20261031_101701_23MHz", path="/kept/20261031_101701_23MHz.ms")
    for k in ("I", "V", "uvw", "natw", "ok_bits", "R"):
        assert recs[1].arrays[k].tobytes() == ref.arrays[k].tobytes(), k


@needs_v3_imager
def test_image_failure_is_reported_after_the_record(tmp_path):
    c = _client(GSI_V3_FAKE_FAIL="20261031_101709_23MHz")
    try:
        c.request("store_open", path=None, backend="ram")
        _peeled_slot(tmp_path / "s0")
        with pytest.raises(engine_client.EngineError, match="fake failure") as e:
            c.request("image_slot", **_req(tmp_path / "s0", tmp_path / "i", "20261031_101709_23MHz"))
        assert e.value.reply["record"]["n_kept"] > 0 and c.alive
    finally:
        c.close()


@needs_v3_imager
def test_engine_death_restart_reopens_the_store(tmp_path):
    p = tmp_path / "ck.gsiv"
    opened = []

    def reopen(cl):
        opened.append(cl.request("store_open", path=str(p), backend="ram", header={"mode": "iv"}))
    c = _client(GSI_V3_FAKE_DIE="20261031_101702_23MHz", on_restart=reopen)
    try:
        c.request("store_open", path=str(p), backend="ram", header={"mode": "iv"})
        for k in range(2):
            _peeled_slot(tmp_path / f"s{k}", seed=k, stem=f"20261031_10170{k}_23MHz")
            c.request("image_slot", **_req(tmp_path / f"s{k}", tmp_path / f"i{k}", f"20261031_10170{k}_23MHz"))
        pid = c.pid
        _peeled_slot(tmp_path / "s2", seed=2, stem="20261031_101702_23MHz")
        with pytest.raises(engine_client.EngineDied):
            c.request("image_slot", **_req(tmp_path / "s2", tmp_path / "i2", "20261031_101702_23MHz"))
        r = c.request("info")                          # restarted, store re-opened from the checkpoint
        # the dying request's record may or may not have reached the file (store thread vs exit)
        assert c.pid != pid and c.restarts == 1 and opened[0]["n"] in (2, 3) and r["store"]["n"] == opened[0]["n"]
        assert opened[0]["stems"][:2] == ["20261031_101700_23MHz", "20261031_101701_23MHz"]
    finally:
        c.close()


@needs_v3_imager
def test_timeout_kills_and_restarts(tmp_path):
    c = _client(GSI_V3_FAKE_SLEEP="30")
    try:
        c.request("store_open", path=None, backend="ram")
        _peeled_slot(tmp_path / "s0")
        pid = c.pid
        t0 = time.monotonic()
        with pytest.raises(engine_client.EngineTimeout):
            c.request("image_slot", timeout=2, **_req(tmp_path / "s0", tmp_path / "i", "20261031_101709_23MHz"))
        assert time.monotonic() - t0 < 20 and not c.alive
        assert c.request("info")["status"] == "ok" and c.pid != pid and c.restarts == 1
    finally:
        c.close()


@needs_v3_imager
def test_fatal_cuda_error_restarts(tmp_path):
    c = _client(GSI_V3_FAKE_FATAL="20261031_101709_23MHz")
    try:
        c.request("store_open", path=None, backend="ram")
        _peeled_slot(tmp_path / "s0")
        with pytest.raises(engine_client.EngineDied, match="fatal"):
            c.request("image_slot", **_req(tmp_path / "s0", tmp_path / "i", "20261031_101709_23MHz"))
        assert not c.alive
        c.request("info")
        assert c.restarts == 1
    finally:
        c.close()


@needs_v3_imager
def test_stale_reply_of_a_killed_engine_is_never_taken(tmp_path):
    c = _client(GSI_V3_FAKE_SLEEP="3")
    try:
        c.request("store_open", path=None, backend="ram")
        _peeled_slot(tmp_path / "s0")
        h = c.submit("image_slot", **_req(tmp_path / "s0", tmp_path / "i", "20261031_101709_23MHz"))
        c.restart("test")
        with pytest.raises(engine_client.EngineDied):
            h.result(10)
    finally:
        c.close()


@needs_v3_imager
def test_restart_limit(tmp_path):
    c = _client()
    c.max_restarts = 1
    try:
        c.restart("one")
        c.request("info")
        c.restart("two")
        with pytest.raises(engine_client.EngineDied, match="giving up"):
            c.request("info")
    finally:
        c.close()


def test_bad_imager_fails_at_start(tmp_path):
    d = tmp_path / "old" / "ttcalx_imager"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(tmp_path / "old")])
    with pytest.raises(engine_client.EngineDied, match="did not start"):
        engine_client.EngineClient(sys.executable, ENGINE, env, start_timeout=60)


@needs_v3_imager
def test_hour_op_uses_the_newest_record_of_the_expected_stems(tmp_path):
    from ttcalx_imager import rotate  # noqa: F401
    c = _client(GSI_V3_FAKE_HOUR="1")
    try:
        c.request("store_open", path=None, backend="ram")
        stems = [f"20261031_10170{k}_23MHz" for k in range(3)]
        for k, st in enumerate(stems):
            _peeled_slot(tmp_path / f"s{k}", seed=k, stem=st)
            c.request("image_slot", **_req(tmp_path / f"s{k}", tmp_path / f"i{k}", st))
        _peeled_slot(tmp_path / "s9", seed=9, stem=stems[1])           # a rerun of stem 1: newest wins
        c.request("image_slot", **_req(tmp_path / "s9", tmp_path / "i9", stems[1]))
        out = tmp_path / "hw" / "23MHz"
        argv = ["--band", "23", "--out", str(out), "--products", "I-Deep-Taper-Robust-0,V-Taper-Deep",
                "--lst-hour", "5", "--qa", "none", "-v", "--exclude-stems", stems[2]]
        with pytest.raises(engine_client.EngineError, match="no record"):
            c.request("hour", argv=argv, expected=stems + ["20261031_101799_23MHz"])
        if not CASACORE:
            pytest.skip("records without R at 05h30m need casacore measures for lazy_snaps")
        r = c.request("hour", timeout=300, argv=argv, expected=stems[:2] + [stems[2]], log=str(tmp_path / "h.log"))
        assert r["n_loaded"] == 3 and r["qa_stems"] == stems and r["rejected"] == [2]
        assert sorted(p.name for p in out.parent.glob("*.fits")) == ["23MHz-I-Deep-Taper-Robust-0-image.fits",
                                                                     "23MHz-V-Taper-Deep-image.fits"]
        s = json.loads((out.parent / "23MHz-summary.json").read_text())
        n_new = hourstore.records_from_snap(_peeled_slot(tmp_path / "t", seed=9, stem=stems[1])).n
        n0 = hourstore.records_from_snap(_peeled_slot(tmp_path / "t", seed=0, stem=stems[0])).n
        assert s["n_samples"] == n0 + n_new
        r = c.request("hour", timeout=300, argv=argv, expected=stems[:2])
        assert r["extra_records"] == [stems[2]] and r["n_loaded"] == 2
    finally:
        c.close()


@needs_v3_imager
def test_failed_store_reopen_fails_the_engine_for_good(tmp_path):
    calls = []

    def reopen(cl):
        calls.append(1)
        raise RuntimeError("store_open failed")
    c = _client(on_restart=reopen)
    try:
        c.request("info")
        c.restart("test")
        with pytest.raises(engine_client.EngineFailed, match="restart failed"):
            c.request("info")
        with pytest.raises(engine_client.EngineFailed):          # no further restarts
            c.request("info")
        assert calls == [1] and not c.alive
    finally:
        c.close()


@needs_v3_imager
def test_engine_rotation_and_rephase_equal_chgcentre(tmp_path):
    """The UVW, DATA and PHASE_DIR the engine hands the imager (prepare(): rotate.chgcentre_uvw,
    rotate.rephase, chgcentre's parse; numpy in the fake backend, the same code on CuPy) against a
    chgcentre'd copy of the same MS."""
    import shutil
    import subprocess
    if not CASACORE or not os.path.exists("/opt/bin/chgcentre"):
        pytest.skip("needs casacore measures and /opt/bin/chgcentre")
    from casacore.tables import table
    from gpu_subband_imaging.v3 import peel_shm
    from v3_synth import make_raw_ms
    ms = tmp_path / "20261031_101007_23MHz.ms"
    make_raw_ms(ms, nant=12, nchan=8, f0=50e6, df=1e5)
    pc = "05h30m00s 37d12m57.057s"
    snap = peel_shm.snap_from_ms(ms)
    slots.write_slot(tmp_path / "s", snap)
    slots.set_state(tmp_path / "s", slots.PEELED)
    c = _client(GSI_V3_FAKE_DUMP="1")
    try:
        c.request("store_open", path=None, backend="ram")
        c.request("image_slot", **_req(tmp_path / "s", tmp_path / "img", "20261031_101007_23MHz", phase_centre=pc,
                                       record_phase_centre=pc))
    finally:
        c.close()
    got = np.load(tmp_path / "img-prep.npz")
    ref = tmp_path / "chg.ms"
    shutil.copytree(ms, ref)
    r = subprocess.run(["/opt/bin/chgcentre", str(ref)] + pc.split(), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with table(str(ref), ack=False) as t:
        uvw, data, a1, a2 = t.getcol("UVW"), t.getcol("DATA"), t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
    with table(str(ref) + "/FIELD", ack=False) as f:
        pd = f.getcol("PHASE_DIR")[0, 0]
    keep = a1 != a2
    assert np.abs(got["uvw"][keep] - uvw[keep]).max() < 1e-6
    rms = np.sqrt(np.mean(np.abs(data[keep]) ** 2))
    assert np.abs(got["data"][keep] - data[keep]).max() / rms < 1e-5
    assert np.array_equal(got["a1"], a1)
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:      # x87: chgcentre's parse is exact
        assert list(got["phase_dir"]) == list(pd)
    assert np.abs(got["phase_dir"] - pd).max() < 1e-15
