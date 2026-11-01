"""gsi v3 full batch (v3.snapshot_io: shm, hour_source: memory) without a GPU or Julia.

The prep (a synthetic SnapArrays written into the slot), the peel (data x 2, state PEELED) and
fpack are faked; the slot ring, the engine (a real child process, fake imaging backend, real
records and hour store), the FITS post-processing, the QA, place_products and the provenance
are real.  Plus the orchestrator / worklist rules of hour_source=memory.
"""
import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging import dispatch, hour_lst, lst, metadata, worklist
from gpu_subband_imaging.config import Config, ConfigError, Slot, V3Params
from gpu_subband_imaging.ledger import DONE, Job, Ledger
from gpu_subband_imaging.orchestrator import Orchestrator, is_hour_job
from gpu_subband_imaging.stages import compress
from gpu_subband_imaging.v3 import fastprep, hourstore, slots
from gpu_subband_imaging.worker import Worker

from test_v3_engine import needs_v3_imager
from test_v3_hourstore import CASACORE, IMAGER_SRC
from test_v3_slots import make_snap

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "examples" / "config"
STAMPS = [f"20261031_1010{s:02d}_23MHz" for s in (7, 17, 27, 37, 47)]
PC = "05h30m00s 37d12m57.057s"


# -- orchestrator / worklist ----------------------------------------------------------------------

def _pl(**v3):
    return SimpleNamespace(layout="lst", lst_hours=[5, 6], sun_cut=False, max_files_per_hour=None, batch_size=2,
                           bands=[23], data_root="/d", date="2026-10-31", hours="all", stream_hour_images=True,
                           v3=V3Params(**v3) if v3 else V3Params())


def test_worklist_one_batch_per_band_hour_in_memory_mode(tmp_path, monkeypatch):
    tars = [f"/d/23MHz/2026-10-31/10/{s}.ms.tar" for s in STAMPS] + ["/d/23MHz/2026-10-31/11/20261031_110000_23MHz.ms.tar"]
    monkeypatch.setattr(worklist, "_list_inputs", lambda *a: tars)
    monkeypatch.setattr(lst, "group", lambda names, hours, sun: {5: list(names[:5]), 6: list(names[5:])})
    v2 = worklist.build(_pl(), "n", tmp_path / "m2")
    assert [(b.group, len(b.tars)) for b in v2] == [("05h", 2), ("05h", 2), ("05h", 1), ("06h", 1)]
    mem = _pl(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", write_kept_ms=False)
    v3 = worklist.build(mem, "n", tmp_path / "m3")
    assert [(b.group, len(b.tars)) for b in v3] == [("05h", 5), ("06h", 1)]


def test_plan_seeds_no_hour_jobs_in_memory_mode(tmp_path, monkeypatch):
    for v3, n_hour in (({}, 2), (dict(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory",
                                      write_kept_ms=False), 0)):
        orch = Orchestrator.__new__(Orchestrator)
        pl = _pl(**v3)
        pl.run_key = f"k{n_hour}"
        orch.cfg = SimpleNamespace(pipeline=pl)
        orch.list_node = "n"
        orch.state = tmp_path / f"s{n_hour}"
        orch.ledger = Ledger(orch.state / "l.sqlite")
        b = [worklist.Batch(23, 0, ["a"], "05h"), worklist.Batch(23, 1, ["b"], "06h")]
        monkeypatch.setattr(worklist, "build", lambda *a: b)
        orch.plan()
        rows = orch.ledger.all()
        assert sum(is_hour_job(j) for j in rows) == n_hour and sum(not is_hour_job(j) for j in rows) == 2
        orch.ledger.close()


def test_wideband_counts_the_memory_batch_as_the_hour(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    pl = _pl(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", write_kept_ms=False)
    pl.wideband_per_hour = True
    orch.cfg = SimpleNamespace(pipeline=pl)
    orch.ledger = Ledger(tmp_path / "l.sqlite")
    orch.ledger.seed([Job(23, 0, 5), Job(55, 1, 5), Job(23, 2, 1)])
    orch.group_of = {(23, 0): "05h", (55, 1): "05h", (23, 2): "06h"}
    orch._wideband_done, orch._wideband_pool = set(), None
    ran = []
    orch._wideband = ran.append

    class Pool:
        def submit(self, fn, *a):
            fn(*a)
    orch._wideband_pool = Pool()
    orch._maybe_wideband()
    assert ran == []
    for j in orch.ledger.all():
        if j.batch in (0, 1):
            orch.ledger.mark_done(j)
    orch._maybe_wideband()
    assert ran == ["05h"]
    orch.ledger.close()


def test_v3_config_rules(tmp_path):
    with pytest.raises(ConfigError, match="checkpoint=archive"):
        V3Params(checkpoint="archive")
    with pytest.raises(ConfigError, match="spill_dirs"):
        V3Params(spill_dirs=["/lustre/x"])
    with pytest.raises(ConfigError, match="engine_timeout_s"):
        V3Params(engine_timeout_s=0)


# -- the batch ----------------------------------------------------------------------------------------

class FakeDaemon:
    """The Julia daemon's slot peel: data x 2, PEELED; optionally fails one stamp."""

    def __init__(self, fail=None, sleep=0.0):
        self.fail, self.calls, self.closed, self.sleep = fail, [], 0, sleep

    def zest_slot(self, slot, timeout=None):
        import time
        time.sleep(self.sleep)
        v = slots.view_slot(slot)
        self.calls.append(v.stem)
        if v.stem == self.fail:
            del v
            raise RuntimeError("zest failed: FAILED")
        v.data[...] *= 2
        del v
        slots.set_state(slot, slots.PEELED)
        return 1.0

    def reset_warm(self, timeout=None):
        pass

    def close(self):
        self.closed += 1


def _snap_of(stamp):
    s = make_snap(nant=8, nchan=6, seed=STAMPS.index(stamp) if stamp in STAMPS else 99, stem=stamp)
    s.data[0, 0, 0] = 1 + 1j
    return s


def fake_prep_slot_task(raw_ms, bp, xy, badants, chanbin, fix_uvw_mode, sample_rows, min_amplitude, slot,
                        min_nonzero=0.0):
    stamp = Path(raw_ms).name[:-3]
    snap = _snap_of(stamp)
    snap.meta.update(chanbin=4, fix_uvw=None, fix_uvw_error=None)
    h = slots.write_slot(slot, snap)
    return dict(stats=dict(sampled_values=100, usable_values=90, nonzero_count=80, nonzero_fraction=0.9,
                           median_amplitude=1.0, max_amplitude=9.0),
                slot_written=True, slot_nbytes=h["nbytes"], meta=dict(timings={"read": {"wall": 0.1, "cpu": 0.1}}),
                task_wall=0.2, maxrss_mb=1.0, pid=os.getpid())


def _config(tmp_path, keep=False, **v3):
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    v = dict(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", write_kept_ms=keep,
             store="ram", spill_dirs=[str(tmp_path / "spill")], checkpoint="local", lookahead=2,
             engine_timeout_s=120, engine_start_timeout_s=120)
    v.update(v3)
    d.update(date="2026-10-31", bands=[23], layout="lst", lst_hours=[5], run_label="Run_t", sun_cut=False,
             snapshot_phase_centre="lst", stream_hour_images=True, prep="numpy", do_aoflag=False, do_badants=False,
             output_root=str(tmp_path / "out"), data_root=str(tmp_path / "in"),
             keep_ms_root=str(tmp_path / "kept") if keep else None,
             imaging=dict(imager="ttcalx", mode="clean", gpu_python=sys.executable,
                          gpu_env=dict(PYTHONPATH=IMAGER_SRC or "", GSI_V3_ENGINE_BACKEND="fake", GSI_V3_FAKE_HOUR="1",
                                       CUDA_VISIBLE_DEVICES="")),
             hour_imaging=dict(ttcalx_image=str(tmp_path / "bin" / "ttcalx-image"),
                               products=["I-Deep-Taper-Robust-0", "V-Taper-Deep", "I-Taper-10min"],
                               delete_kept_ms=True, timeout_seconds=300),
             movie=dict(enabled=False), v3=v)
    d["cal"].update(bp=str(tmp_path / "bp"), xy=str(tmp_path / "xy"))
    c = tmp_path / "cfg"
    c.mkdir(exist_ok=True)
    (c / "pipeline.yaml").write_text(yaml.safe_dump(d))
    cl = yaml.safe_load((CONFIG / "cluster.yaml").read_text())
    cl["control"]["state_dir"] = str(tmp_path / "state")
    cl["defaults"].update(work_root=str(tmp_path / "work"), max_feeders=2, zest_recycle_files=0)
    (c / "cluster.yaml").write_text(yaml.safe_dump(cl))
    shutil.copy(CONFIG / "subbands.yaml", c / "subbands.yaml")
    return Config.load(c)


@pytest.fixture
def env(tmp_path, monkeypatch):
    if not CASACORE:
        pytest.skip("needs casacore measures (the rotations)")
    tars = []
    for s in STAMPS:
        t = tmp_path / "in" / f"{s}.ms.tar"
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(b"x")
        tars.append(str(t))
    monkeypatch.setattr(Worker, "_stage_cal", lambda self, src: src)
    monkeypatch.setattr(Worker, "_v3_start_pool", lambda self: None)
    monkeypatch.setattr(Worker, "_v3_run", lambda self, fn, *a: fn(*a))
    monkeypatch.setattr(fastprep, "prep_slot_task", fake_prep_slot_task)

    def stage(self, tar, d, stamp):
        ms = d / f"{stamp}.ms"
        ms.mkdir(parents=True, exist_ok=True)
        return ms
    monkeypatch.setattr(Worker, "_stage", stage)

    def fpack(f):
        fz = Path(f"{f}.fz")
        shutil.copy(f, fz)
        return fz
    monkeypatch.setattr(compress, "fpack", fpack)
    monkeypatch.setattr(metadata, "software_versions", lambda cfg: {})
    monkeypatch.setenv("GSI_WORKERS", str(REPO / "workers"))
    return SimpleNamespace(tars=tars, tmp=tmp_path)


def _worker(cfg, daemon):
    w = Worker(cfg, 23, 0, 0)
    w._lst = {s: 5 for s in STAMPS}
    w._start_daemon = lambda: daemon
    return w


def _bdir(tmp_path):
    return tmp_path / "out" / "05h" / "2026-10-31" / "Run_t" / "23MHz"


@needs_v3_imager
def test_memory_batch_end_to_end(env, caplog):
    tmp = env.tmp
    cfg = _config(tmp, delete_checkpoint=False)
    d = FakeDaemon()
    w = _worker(cfg, d)
    with caplog.at_level(logging.INFO, logger="gsi"):
        w.run_batch(env.tars)
    b = _bdir(tmp)
    assert sorted(p.name for p in (b / "snapshots_clean").iterdir()) == \
        sorted(f"{lst.clean_name(23, lst.timestamp(s))}.fz" for s in STAMPS)
    assert len(list((b / "snapshots").glob("*.fits.fz"))) == 5
    deep = sorted(p.name for p in (b / "I" / "deep").iterdir())
    assert len(deep) == 1 and deep[0].startswith("23MHz-I-Deep-Taper-Robust-0-image-")
    # 5 snapshots in 6 intervals: the empty one is named with the deep mid time, as one other (v2 rule)
    assert len(list((b / "V" / "deep").iterdir())) == 1 and len(list((b / "I" / "10min").iterdir())) == 5
    prov = json.loads((b / "provenance.json").read_text())
    h = prov["hour"]
    assert h["status"] == "ok" and h["hour_source"] == "memory" and h["n_loaded"] == 5 and h["n_records"] == 5
    assert prov["v3"]["hour_source"] == "memory" and prov["v3"]["store"]["backend"] == "ram"
    assert (b / "QA" / "snapshot_qa_23MHz.csv").exists() and (b / "QA" / "23MHz-summary.json").exists()
    assert len(list((b / "QA" / ".snapshot_stats").glob("*.jsonl"))) == 1
    # the checkpoint holds exactly the records of the peeled snapshots (data x 2)
    ck = Path(prov["v3"]["store_path"])
    recs = {r.stem: r for r in hourstore.read_records(ck)[2]}
    assert sorted(recs) == STAMPS
    from ttcalx_imager import rotate
    ra, dec = rotate.parse_phase_centre(PC)
    for s in STAMPS:
        p = _snap_of(s)
        p.data[...] *= 2
        ref = hourstore.records_from_snap(p, ra, dec, stem=s, path=f"{s}.ms")
        for k in ("I", "V", "uvw", "natw", "ok_bits", "freqs", "chan_width"):
            assert recs[s].arrays[k].tobytes() == ref.arrays[k].tobytes(), (s, k)
        assert np.allclose(recs[s].arrays["R"], ref.arrays["R"], rtol=0, atol=1e-9)
        assert recs[s].meta["pd_new"] == [ra, dec]
    # no slot, no work dir, no untar dir left; the daemon and the engine were closed
    assert not list(Path("/dev/shm").glob(f"gsi3_{os.getpid()}_*")) and not w.work.exists() and d.closed >= 1
    assert "V3_REPLAY" not in caplog.text and "phase=hour_total" in caplog.text
    assert "ring of 4 slots" in caplog.text
    # the rerun has nothing to do (FITS and hour products there)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gsi"):
        _worker(cfg, FakeDaemon()).run_batch(env.tars)
    assert "nothing to do" in caplog.text


@needs_v3_imager
def test_rerun_replays_missing_records_and_redoes_the_hour(env, caplog):
    tmp = env.tmp
    cfg = _config(tmp, delete_checkpoint=False)
    _worker(cfg, FakeDaemon()).run_batch(env.tars)
    b = _bdir(tmp)
    prov = json.loads((b / "provenance.json").read_text())
    ck = Path(prov["v3"]["store_path"])
    # the hour must be redone (provenance gone) and the records are lost (another node)
    (b / "provenance.json").unlink()
    for f in (ck, hourstore.idx_path(ck)):
        f.unlink()
    d = FakeDaemon()
    with caplog.at_level(logging.INFO, logger="gsi"):
        _worker(cfg, d).run_batch(env.tars)
    assert "V3_REPLAY" in caplog.text and sorted(d.calls) == STAMPS
    prov = json.loads((b / "provenance.json").read_text())
    assert prov["hour"]["status"] == "ok" and prov["hour"]["n_loaded"] == 5
    assert len(list((b / "snapshots_clean").iterdir())) == 5
    # a record still in the checkpoint is reused: only the missing one is replayed
    (b / "provenance.json").unlink()
    st = hourstore.HourStore(ck)
    keep = [r for r in st.records if r.stem != STAMPS[2]]
    st.close()
    for f in (ck, hourstore.idx_path(ck)):
        f.unlink()
    d = FakeDaemon()
    w = _worker(cfg, d)
    caplog.clear()
    # rebuild the checkpoint with 4 of the 5 records under the batch's own header
    from gpu_subband_imaging.v3 import worker_v3
    vb = worker_v3.V3Batch(w, env.tars)
    hdr = hourstore.header_for(23, "2026-10-31", "05h", PC, "Run_t", batch=0, layout="lst", cal_bp=str(cfg.pipeline.cal.bp),
                               cal_xy=str(cfg.pipeline.cal.xy))
    from ttcalx_imager import rotate
    import ttcalx_imager
    hdr.update(pd_new=list(rotate.parse_phase_centre(PC)), imager=ttcalx_imager.__file__)
    st = hourstore.HourStore(ck, header_meta=hdr)
    for r in keep:
        st.append(hourstore.Record(r.meta, r.arrays))
    st.close()
    with caplog.at_level(logging.INFO, logger="gsi"):
        vb.run()
    assert d.calls == [STAMPS[2]] and "1 snapshot(s) have FITS but no record" in caplog.text
    assert json.loads((b / "provenance.json").read_text())["hour"]["n_loaded"] == 5


@needs_v3_imager
def test_failed_peel_fails_the_batch_then_the_retry_makes_the_hour(env, caplog):
    """A snapshot that could have been imaged is missing -> no hour products, the batch fails; the
    retry redoes only that snapshot and makes the hour (then the checkpoint is deleted)."""
    tmp = env.tmp
    cfg = _config(tmp)
    with caplog.at_level(logging.INFO, logger="gsi"), pytest.raises(RuntimeError, match="V3_INCOMPLETE"):
        _worker(cfg, FakeDaemon(fail=STAMPS[1])).run_batch(env.tars)
    b = _bdir(tmp)
    assert not (b / "provenance.json").exists() and len(list((b / "snapshots_clean").iterdir())) == 4
    d = FakeDaemon()
    _worker(cfg, d).run_batch(env.tars)
    prov = json.loads((b / "provenance.json").read_text())
    assert d.calls == [STAMPS[1]] and prov["hour"]["n_loaded"] == 5
    assert not Path(prov["v3"]["store_path"]).exists()          # checkpoint deleted after the hour


@needs_v3_imager
def test_hour_failure_fails_the_batch_and_keeps_the_checkpoint(env, monkeypatch):
    tmp = env.tmp
    cfg = _config(tmp)
    cfg.pipeline.imaging.gpu_env["GSI_V3_FAKE_HOUR"] = "fail"
    with pytest.raises(RuntimeError, match="memory"):
        _worker(cfg, FakeDaemon()).run_batch(env.tars)
    b = _bdir(tmp)
    prov = json.loads((b / "provenance.json").read_text())
    assert prov["hour"]["status"] == "failed" and "fake hour failure" in prov["hour"]["error"]
    assert Path(prov["v3"]["store_path"]).exists()
    assert len(list((b / "snapshots_clean").iterdir())) == 5     # the snapshot products stay


@needs_v3_imager
def test_write_kept_ms_from_the_slot(env, monkeypatch):
    tmp = env.tmp
    cfg = _config(tmp, keep=True)
    tpl = tmp / "tpl.ms"
    tpl.mkdir()
    monkeypatch.setattr(Worker, "_v3_template_for", lambda self, ms: setattr(self, "_v3_template", tpl) or tpl)
    monkeypatch.setattr(fastprep, "check_template", lambda *a: None)
    written = {}
    lock = threading.Lock()

    def kept(template, raw_ms, slot, out):
        v = slots.view_slot(slot, writable=False)
        assert slots.get_state(slot) == slots.PEELED and Path(raw_ms).is_dir()
        Path(out).mkdir()
        np.save(Path(out) / "data.npy", np.asarray(v.data))
        with lock:
            written[v.stem] = out
        return dict(wall=0.0)
    monkeypatch.setattr(fastprep, "kept_from_slot_task", kept)
    deleted = []
    monkeypatch.setattr(hour_lst, "_delete_kept", lambda cfg, band, group, imaged: deleted.append(sorted(imaged)))
    _worker(cfg, FakeDaemon()).run_batch(env.tars)
    assert sorted(written) == STAMPS
    for s in STAMPS:
        k = tmp / "kept" / "23MHz" / "2026-10-31" / "05h" / f"{s}.ms"
        p = _snap_of(s)
        assert np.load(k / "data.npy").tobytes() == (p.data * 2).tobytes()
    assert deleted == [STAMPS]


@needs_v3_imager
def test_rerun_with_only_skipped_inputs_still_makes_the_hour(env, monkeypatch, caplog):
    """Found on the real band-hour: two snapshots are always skipped (no data); the hour rerun
    must not fail with "no outputs" because nothing is left to image."""
    tmp = env.tmp
    cfg = _config(tmp, delete_checkpoint=False)
    extra = tmp / "in" / "20261031_101057_23MHz.ms.tar"
    extra.write_bytes(b"x")
    tars = env.tars + [str(extra)]

    def prep(raw_ms, *a, **k):
        if Path(raw_ms).name.startswith("20261031_101057"):
            return dict(stats=dict(sampled_values=100, usable_values=0, nonzero_count=0, nonzero_fraction=0.0,
                                   median_amplitude=0.0, max_amplitude=0.0), slot_written=False,
                        meta=dict(timings={}), task_wall=0.1, maxrss_mb=1.0, pid=os.getpid())
        return fake_prep_slot_task(raw_ms, *a, **k)
    monkeypatch.setattr(fastprep, "prep_slot_task", prep)
    w = _worker(cfg, FakeDaemon())
    w._lst["20261031_101057_23MHz"] = 5
    w.run_batch(tars)
    b = _bdir(tmp)
    (b / "provenance.json").unlink()
    w = _worker(cfg, FakeDaemon())
    w._lst["20261031_101057_23MHz"] = 5
    with caplog.at_level(logging.INFO, logger="gsi"):
        w.run_batch(tars)
    # the input skipped before is not prepped again (finding 5)
    assert "0 to image, 0 to replay" in caplog.text and "1 input(s) skipped before" in caplog.text
    assert json.loads((b / "provenance.json").read_text())["hour"]["status"] == "ok"
    # and a done hour with an always-skipped input has nothing to do at all
    caplog.clear()
    d = FakeDaemon()
    w = _worker(cfg, d)
    with caplog.at_level(logging.INFO, logger="gsi"):
        w.run_batch(tars)
    assert "nothing to do (hour products done)" in caplog.text and d.calls == []


@needs_v3_imager
def test_engine_death_does_not_take_a_healthy_sibling_and_no_hour_is_made(env, caplog):
    """Review finding 1: the engine dies on snapshot 4 of 5 while others are in flight.  The healthy
    ones are retried alone and succeed, the culprit (dies again alone) fails, and the hour is NOT
    made (one snapshot that could be imaged is missing); a rerun then completes the band-hour."""
    tmp = env.tmp
    cfg = _config(tmp, delete_checkpoint=False)
    cfg.pipeline.imaging.gpu_env.update(GSI_V3_FAKE_DIE=STAMPS[3], GSI_V3_FAKE_SLEEP="0.3")
    with caplog.at_level(logging.INFO, logger="gsi"), pytest.raises(RuntimeError, match="V3_INCOMPLETE"):
        _worker(cfg, FakeDaemon()).run_batch(env.tars)
    b = _bdir(tmp)
    done = sorted(p.name for p in (b / "snapshots_clean").iterdir())
    assert done == sorted(f"{lst.clean_name(23, lst.timestamp(s))}.fz" for s in STAMPS if s != STAMPS[3])
    assert "V3_ENGINE_CULPRIT" in caplog.text and STAMPS[3] in caplog.text.split("V3_ENGINE_CULPRIT")[1][:200]
    assert caplog.text.count("V3_ENGINE_CULPRIT") == 1
    assert not (b / "provenance.json").exists()
    # the pilot rows of the archived snapshots were persisted as they came
    rows = [ln for f in (b / "QA" / ".snapshot_stats").glob("*.jsonl") for ln in f.read_text().splitlines()]
    assert len(rows) == 4
    del cfg.pipeline.imaging.gpu_env["GSI_V3_FAKE_DIE"]
    caplog.clear()
    d = FakeDaemon()
    with caplog.at_level(logging.INFO, logger="gsi"):
        _worker(cfg, d).run_batch(env.tars)
    assert d.calls == [STAMPS[3]] and "1 to image, 0 to replay" in caplog.text
    h = json.loads((b / "provenance.json").read_text())["hour"]
    assert h["status"] == "ok" and h["n_loaded"] == 5


@needs_v3_imager
def test_exhausted_engine_restarts_abort_the_batch_at_once(env, caplog):
    tmp = env.tmp
    cfg = _config(tmp, engine_max_restarts=1)
    cfg.pipeline.imaging.gpu_env.update(GSI_V3_FAKE_DIE=STAMPS[0])
    d = FakeDaemon(sleep=0.5)
    from gpu_subband_imaging.v3.worker_v3 import BatchAborted
    with caplog.at_level(logging.INFO, logger="gsi"), pytest.raises(BatchAborted, match="engine failed"):
        _worker(cfg, d).run_batch(env.tars)
    assert "V3_ABORT" in caplog.text and len(d.calls) < len(STAMPS)
    assert not list(Path("/dev/shm").glob(f"gsi3_{os.getpid()}_*"))


def test_memory_mode_has_its_own_run_key_and_ledger(tmp_path):
    """Review finding 2: a v2 ledger of the same run (hour-job rows, batch_size batches) is never
    reused by hour_source=memory (INSERT OR IGNORE on (band, batch) would skip band-hours)."""
    base = dict(date="2026-10-31", hours="all", bands=[23], batch_size=90, batch_timeout_seconds=None,
                max_retries=1, data_root="/d", output_root="/o", chanbin=4, cal=None, peel=None, peel_overrides={},
                imaging=config_mod.ImagingParams(mode="clean", imager="ttcalx"), movie=None, do_aoflag=False,
                do_badants=True, env=None, layout="lst", lst_hours=[5], run_label="Run_x", stream_hour_images=True,
                prep="numpy")
    v2 = config_mod.PipelineConfig(**base)
    mem = config_mod.PipelineConfig(**base, v3=V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm",
                                                        hour_source="memory"))
    assert v2.run_key == "2026-10-31_Run_x_lst05" and mem.run_key == "2026-10-31_Run_x_lst05_v3mem"
    led = Ledger(tmp_path / f"ledger_{v2.run_key}.sqlite")
    led.seed([Job(23, 0, 90), Job(23, 1, 90), Job(23, -1005, 2)])
    for j in led.all():
        led.mark_done(j)
    led.close()
    m = Ledger(tmp_path / f"ledger_{mem.run_key}.sqlite")
    m.seed([Job(23, 0, 358)])
    assert [(j.band, j.batch, j.n_files, j.status) for j in m.all()] == [(23, 0, 358, "queued")]
    m.close()


def test_memory_mode_config_rules():
    v = V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory")
    assert v.write_kept_ms is False and V3Params().write_kept_ms is True
    assert V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory",
                    write_kept_ms=True).write_kept_ms is True
    with pytest.raises(ConfigError, match="checkpoint"):
        V3Params(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", checkpoint="none")
    base = dict(date="2026-10-31", hours="all", bands=[23], batch_size=90, max_retries=1, data_root="/d",
                output_root="/o", chanbin=4, cal=None, peel=None, peel_overrides={},
                imaging=config_mod.ImagingParams(mode="clean", imager="ttcalx"), movie=None, do_aoflag=False,
                do_badants=True, env=None, layout="lst", lst_hours=[5], run_label="Run_x", stream_hour_images=True,
                prep="numpy", v3=v)
    with pytest.raises(ConfigError, match="batch_timeout_seconds"):       # prod: 7200 / 7200
        config_mod.PipelineConfig(**base, batch_timeout_seconds=7200,
                                  hour_imaging=config_mod.HourImagingParams(timeout_seconds=7200))
    config_mod.PipelineConfig(**base, batch_timeout_seconds=7200 + 1800 + 300,
                              hour_imaging=config_mod.HourImagingParams(timeout_seconds=7200))


@needs_v3_imager
def test_rerun_plan_counts_existing_records(env, monkeypatch, caplog):
    """Review finding 4: n_todo = 0 on a rerun but the checkpoint holds 12 GB, 1 GB free: spill."""
    from gpu_subband_imaging.v3 import memory, worker_v3
    cfg = _config(env.tmp, store="auto")
    w = _worker(cfg, FakeDaemon())
    vb = worker_v3.V3Batch(w, env.tars)
    ck = vb._ck_name(cfg.pipeline.v3.spill_dirs[0])
    monkeypatch.setattr(worker_v3.V3Batch, "_existing_store", lambda self: (ck, 350, 12e9))
    monkeypatch.setattr(memory, "mem_available", lambda *a: (16 + 1) * 1e9)
    monkeypatch.setattr(memory, "reserved_by_others", lambda *a: 0.0)
    with caplog.at_level(logging.INFO, logger="gsi"):
        vb._plan_store(0)
    assert vb.plan.backend == "memmap" and vb.store_path == ck and "V3_MEM_SPILL" in caplog.text
    memory.unreserve()


def test_zest_env_memory_switches(tmp_path, monkeypatch):
    """v3.zest_soft_memory_limit / zest_hard_memory_limit / zest_gc_between reach the daemon's env;
    the defaults keep v2's 4GiB soft limit (GPU imager) and set nothing else."""
    monkeypatch.delenv("JULIA_CUDA_SOFT_MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("JULIA_CUDA_HARD_MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("ZEST_GC_BETWEEN", raising=False)

    def env(**v3):
        pl = SimpleNamespace(imaging=SimpleNamespace(imager="ttcalx"), v3=V3Params(**v3))
        w = Worker.__new__(Worker)
        w.cfg = SimpleNamespace(pipeline=pl)
        return w._zest_env()
    e = env()
    assert e["JULIA_CUDA_SOFT_MEMORY_LIMIT"] == "4GiB" and "JULIA_CUDA_HARD_MEMORY_LIMIT" not in e \
        and "ZEST_GC_BETWEEN" not in e
    assert "JULIA_CUDA_SOFT_MEMORY_LIMIT" not in env(zest_soft_memory_limit="none")
    e = env(zest_soft_memory_limit="6GiB", zest_hard_memory_limit="7GiB", zest_gc_between="full")
    assert (e["JULIA_CUDA_SOFT_MEMORY_LIMIT"], e["JULIA_CUDA_HARD_MEMORY_LIMIT"], e["ZEST_GC_BETWEEN"]) == \
        ("6GiB", "7GiB", "full")
    with pytest.raises(ConfigError):
        V3Params(zest_gc_between="always")


@needs_v3_imager
def test_image_overlap_false_serialises_the_engine_behind_the_peel(env, caplog):
    cfg = _config(env.tmp, image_overlap=False)
    with caplog.at_level(logging.INFO, logger="gsi"):
        _worker(cfg, FakeDaemon()).run_batch(env.tars)
    assert caplog.text.count("phase=image_gate_wait") == len(STAMPS) - 1
    assert json.loads((_bdir(env.tmp) / "provenance.json").read_text())["hour"]["status"] == "ok"
