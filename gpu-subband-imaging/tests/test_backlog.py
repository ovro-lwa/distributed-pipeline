"""Backlog re-runs (feature/gsi-streaming, ported): night-time averaged inputs, cal.badants,
peel.save_solutions on the v2 and the v3 path, hour_imaging.qa_exclude, the backlog provenance
keys, the GPU imager restart and scripts/run_backlog.py.  No GPU, no Julia."""
import importlib.util
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_subband_imaging import hour_lst
from gpu_subband_imaging.config import CalTables, ConfigError, HourImagingParams, PeelParams, _badants
from gpu_subband_imaging.stages import gpu_image, peel_solutions
from gpu_subband_imaging.v3 import peel_shm, slots
from gpu_subband_imaging.worker import Worker, _stamp

from test_v3_slots import make_snap

REPO = Path(__file__).resolve().parents[1]


def _run_backlog():
    spec = importlib.util.spec_from_file_location("run_backlog", REPO / "scripts" / "run_backlog.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    return rb


# -- inputs, bad antennas -----------------------------------------------------------------------------

def test_stamp_of_averaged_input():
    p = "/lustre/pipeline/night-time/averaged/55MHz/2024-12-20/03/20241220_030005_55MHz_averaged.ms"
    assert _stamp(p) == "20241220_030005_55MHz"
    assert _stamp("/x/20261026_031144_55MHz.ms.tar") == "20261026_031144_55MHz"


def test_badants_per_band_and_band_hour():
    per = CalTables("bp", "xy", "s", "ao", badants=_badants({"55": "1,2", 82: "3"}))
    assert per.badants_for(55) == "1,2" and per.badants_for(82) == "3" and per.badants_for(18) is None
    assert CalTables("bp", "xy", "s", "ao", badants=_badants("4,5")).badants_for(18) == "4,5"
    assert CalTables("bp", "xy", "s", "ao").badants_for(18) is None
    c = CalTables("bp", "xy", "s", "ao", badants=_badants({"05h/55": "1", "06h/55": "2", 82: "3"}))
    assert c.badants_for(55, "05h") == "1" and c.badants_for(55, "06h") == "2"
    assert c.badants_for(82, "05h") == "3" and c.badants_for(55) is None


# -- peel.save_solutions: config ----------------------------------------------------------------------

def test_save_solutions_validation_and_env():
    assert PeelParams(save_solutions="sky, rfi").solution_passes == ["sky", "rfi"]
    assert PeelParams().solution_passes == []
    with pytest.raises(ConfigError):
        PeelParams(save_solutions="sky,cal")
    # default: no ZEST_SAVE_SOLUTIONS and no provenance key; set: only that key is added (the
    # v2 legacy environment stays 13405ce's otherwise)
    base, on = PeelParams(), PeelParams(save_solutions="sky")
    assert "ZEST_SAVE_SOLUTIONS" not in base.daemon_env(None) and "save_solutions" not in base.as_dict()
    env = on.daemon_env(None)
    assert env.pop("ZEST_SAVE_SOLUTIONS") == "sky" and env == base.daemon_env(None)
    assert on.as_dict()["save_solutions"] == "sky"
    sky_rfi = PeelParams(engine="sky_rfi", fast_io=True, model="exact", uvw="geometric", residual_guard=True,
                         flag_model=True, save_solutions="sky,rfi")
    assert sky_rfi.daemon_env("r.json")["ZEST_SAVE_SOLUTIONS"] == "sky,rfi"


def _pipeline_yaml(tmp_path, **kw):
    import shutil
    import yaml
    cfg = REPO / "examples" / "config"
    d = yaml.safe_load((cfg / "pipeline.yaml").read_text())
    d.update(kw)
    c = tmp_path / "cfg"
    c.mkdir(exist_ok=True)
    (c / "pipeline.yaml").write_text(yaml.safe_dump(d))
    for n in ("cluster.yaml", "subbands.yaml"):
        shutil.copy(cfg / n, c / n)
    return c


def test_save_solutions_and_chanbin_config_rules(tmp_path, monkeypatch):
    from gpu_subband_imaging.config import Config
    monkeypatch.delenv("GSI_ALLOWED_LUSTRE_WRITE_ROOT", raising=False)
    with pytest.raises(ConfigError, match="layout=lst"):
        Config.load(_pipeline_yaml(tmp_path, layout="utc", peel=dict(save_solutions="sky")))
    with pytest.raises(ConfigError, match="do_aoflag"):
        Config.load(_pipeline_yaml(tmp_path, chanbin=1, do_aoflag=True))
    c = Config.load(_pipeline_yaml(tmp_path, layout="lst", run_label="Run_x", lst_hours=[6], chanbin=1,
                                   do_aoflag=False, description="backlog test",
                                   peel=dict(save_solutions="sky,rfi")))
    assert c.pipeline.description == "backlog test" and c.pipeline.peel.solution_passes == ["sky", "rfi"]
    d = Config.load(_pipeline_yaml(tmp_path))
    assert d.pipeline.description == "" and d.pipeline.hour_imaging.qa_exclude is True


# -- convert / merge ------------------------------------------------------------------------------

def _daemon_output(d: Path, names, gains, accepted, passname="sky", elev=(30.0, 12.5), reverted=False):
    """What zest_daemon.jl write_solutions leaves for one pass."""
    d.mkdir(parents=True, exist_ok=True)
    gains.astype("<c8").ravel(order="F").tofile(d / f"{passname}.bin")
    lines = ["# shape " + " ".join(map(str, gains.shape)),
             "index\tname\televation_deg\taccepted\tconverged\tniters"]
    for k, (n, a) in enumerate(zip(names, accepted)):
        lines.append(f"{k * 2}\t{n}\t{elev[k]:.4f}\t{str(a).lower()}\ttrue\t{5 + k}")
    (d / f"{passname}.tsv").write_text("\n".join(lines) + "\n")
    if reverted:
        (d / f"{passname}.reverted").touch()


def _gains(rng, nsrc, nant=3, nfreq=4):
    shape = (nsrc, 4, nant, nfreq)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def test_convert_merge_and_summary(tmp_path):
    rng = np.random.default_rng(1)
    freq = np.linspace(50e6, 51e6, 4)
    g1, g2 = _gains(rng, 2), _gains(rng, 1)
    bdir = tmp_path / "55MHz"
    stems = ["20241220_030005_55MHz", "20241220_030015_55MHz", "20241220_030025_55MHz"]
    for t, (names, g, acc) in enumerate(((["Cas A", "Cyg A"], g1, [True, False]), (["Cas A"], g2, [True]),
                                         (["Cas A"], g2, [True]))):
        src = tmp_path / f"slot{t}.peelsol"
        _daemon_output(src, names, g, acc, reverted=(t == 1))
        files = peel_solutions.convert(src, peel_solutions.parts_dir(bdir) / stems[t], {"band": "55MHz"},
                                       (freq, 1000.0 + 10 * t), ms_name=f"{stems[t]}_avg.ms")
        assert [f.name for f in files] == ["sky.npz"] and not src.exists()
        with np.load(files[0]) as z:
            np.testing.assert_array_equal(z["gains"][..., 0], g)
            assert z["gains"].shape == g.shape + (1,) and z["accepted"].tolist() == acc
            md = json.loads(str(z["metadata_json"]))
            assert md["schema_version"] == 1 and md["ms"] == f"{stems[t]}_avg.ms"
    # the third snapshot is not one of the band-hour's: it stays
    merged = peel_solutions.merge_band_hour(bdir, stems[:2], "55MHz_x_03h")
    assert [p.name for p in merged] == ["55MHz_x_03h_sky.npz"]
    assert merged[0].parent == bdir / "peeling_solutions"
    assert sorted(p.name for p in peel_solutions.parts_dir(bdir).iterdir()) == [stems[2]]
    with np.load(merged[0]) as z:
        assert z["source_names"].tolist() == ["Cas A", "Cyg A"]
        assert z["gains"].shape == (2, 4, 3, 4, 2)
        np.testing.assert_array_equal(z["gains"][0, ..., 1], g2[0])
        assert np.isnan(z["gains"][1, ..., 1]).all()
        assert z["solved"].tolist() == [[True, True], [True, False]]
        assert z["accepted"].tolist() == [[True, True], [False, False]]
        assert z["niters"][0].tolist() == [5, 5] and z["times_mjd_seconds"].tolist() == [1000.0, 1010.0]
        assert z["ms_names"].tolist() == [f"{s}_avg.ms" for s in stems[:2]]
        assert z["backstop_reverted"].tolist() == [False, True]
    assert peel_solutions.peel_summary(merged) == {"sky:Cas A": {"selected": 2, "subtracted": 1},
                                                   "sky:Cyg A": {"selected": 1, "subtracted": 0}}
    # nothing left to merge for these stems: the merged file stays
    assert peel_solutions.merge_band_hour(bdir, stems[:2], "55MHz_x_03h") == [] and merged[0].exists()


def test_slot_solutions_follow_the_ms_in_peel_ms_via_slot(tmp_path, monkeypatch):
    """v2 worker with v3.peel_io=shm: the daemon writes <slot>.peelsol; it ends up as <ms>.peelsol."""
    monkeypatch.setattr(peel_shm, "snap_from_ms", lambda ms: make_snap())
    monkeypatch.setattr(peel_shm, "write_peeled", lambda ms, s: None)

    class D:
        def zest_slot(self, slot, timeout=None):
            _daemon_output(Path(f"{slot}.peelsol"), ["Cas A"], _gains(np.random.default_rng(0), 1), [True])
            slots.set_state(slot, slots.PEELED)
            return 1.0

        def reset_warm(self, timeout=None):
            pass
    ms = tmp_path / "x_avg.ms"
    ms.mkdir()
    with slots.SlotRing(1) as ring:
        peel_shm.peel_ms_via_slot(D(), ms, ring)
        assert not Path(f"{ring.paths[0]}.peelsol").exists()
    assert (Path(f"{ms}.peelsol") / "sky.tsv").exists()


def test_ring_close_and_stale_cleanup_remove_slot_solutions(tmp_path):
    ring = slots.SlotRing(2, root=tmp_path)
    sol = Path(f"{ring.paths[1]}.peelsol")
    sol.mkdir()
    ring.close()
    assert not sol.exists()
    stale = tmp_path / "gsi3_999999_1_0.peelsol"            # a dead pid's
    stale.mkdir()
    (tmp_path / "gsi3_999999_1_0").write_bytes(b"x")
    removed = slots.cleanup_stale(tmp_path)
    assert str(stale) in removed and not stale.exists()


def _w(tmp_path, peel=None):
    pl = SimpleNamespace(date="2024-12-23", output_root=str(tmp_path / "out"), layout="lst", run_label="Run_b",
                         keep_ms_root=None, peel_for_band=lambda b: peel or PeelParams(save_solutions="sky,rfi"),
                         cal=SimpleNamespace(bp="b", xy="x", sources="s.json", rfi_sources="r.json"))
    w = Worker.__new__(Worker)
    w.cfg, w.band, w.batch = SimpleNamespace(pipeline=pl), 55, 0
    w._lst = {}
    return w


def test_worker_saves_solutions_in_the_band_dir(tmp_path):
    """v2 (target = the peeled MS) and v3 (target = the slot) both write the per-snapshot NPZs below
    <band dir>/peeling_solutions/.snapshots/<stamp>/."""
    w = _w(tmp_path)
    stamp = "20241223_074350_55MHz"
    w._lst[stamp] = 6
    target = tmp_path / "gsi3_1_1_0"
    _daemon_output(Path(f"{target}.peelsol"), ["Cas A", "Cyg A"], _gains(np.random.default_rng(2), 2), [True, True])
    files = w._save_solutions(stamp, target, (np.linspace(50e6, 51e6, 4), 5.0))
    b = w.band_dir(stamp)
    assert files == [peel_solutions.parts_dir(b) / stamp / "sky.npz"] and files[0].exists()
    with np.load(files[0]) as z:
        md = json.loads(str(z["metadata_json"]))
    assert md["band"] == "55MHz" and md["peel"]["save_solutions"] == "sky,rfi" and md["ms"] == f"{stamp}_avg.ms"
    with pytest.raises(RuntimeError, match="no solutions"):
        w._save_solutions(stamp, target, (np.ones(4), 5.0))


def test_slot_axes_are_the_ms_axes(tmp_path):
    snap = make_snap(nchan=5)
    p = tmp_path / "slot"
    slots.write_slot(p, snap)
    freq, t = peel_solutions.slot_axes(p)
    assert np.array_equal(freq, snap.meta["freqs"]) and t == float(snap.time[0])


# -- v3 batch with save_solutions (the engine is a real child process, fake imaging) -------------------

from test_v3_batch import STAMPS, FakeDaemon, _bdir, _config, _worker, env  # noqa: E402,F401
from test_v3_engine import needs_v3_imager  # noqa: E402


class SolDaemon(FakeDaemon):
    """FakeDaemon that also leaves the daemon's solution files beside the slot."""

    def zest_slot(self, slot, timeout=None):
        r = super().zest_slot(slot, timeout)
        rng = np.random.default_rng(len(self.calls))
        d = Path(f"{slot}.peelsol")
        _daemon_output(d, ["Cas A", "Cyg A"], _gains(rng, 2, nant=8, nfreq=6), [True, len(self.calls) % 2 == 0])
        _daemon_output(d, ["sat"], _gains(rng, 1, nant=8, nfreq=6), [True], passname="rfi")
        return r


@needs_v3_imager
def test_v3_batch_saves_and_merges_solutions(env):
    cfg = _config(env.tmp, delete_checkpoint=False)
    pl = replace(cfg.pipeline, peel=replace(cfg.pipeline.peel, save_solutions="sky,rfi"), description="bl test")
    cfg = replace(cfg, pipeline=pl)
    _worker(cfg, SolDaemon()).run_batch(env.tars)
    b = _bdir(env.tmp)
    sols = sorted((b / "peeling_solutions").glob("*.npz"))
    assert [p.name for p in sols] == ["23MHz_2026-10-31_05h_rfi.npz", "23MHz_2026-10-31_05h_sky.npz"]
    assert not peel_solutions.parts_dir(b).exists()
    with np.load(sols[1]) as z:
        assert z["gains"].shape == (2, 4, 8, 6, len(STAMPS)) and z["solved"].all()
        assert z["ms_names"].tolist() == [f"{s}_avg.ms" for s in STAMPS]
        assert z["frequencies_hz"].shape == (6,)
    prov = json.loads((b / "provenance.json").read_text())
    assert prov["description"] == "bl test" and prov["peeling_solutions"] == [str(p) for p in sols[::-1]]
    assert prov["peeled_sources"]["per_snapshot"]["sky:Cas A"] == {"selected": 5, "subtracted": 5}
    assert prov["hour"]["qa"]["rejected_by_v_qa"] >= 0 and prov["hour_imaging"]["v_qa_excludes_snapshots"]
    assert not list(Path("/dev/shm").glob(f"gsi3_{os.getpid()}_*"))


@needs_v3_imager
def test_v3_batch_default_provenance_has_no_backlog_keys(env):
    _worker(_config(env.tmp, delete_checkpoint=False), FakeDaemon()).run_batch(env.tars)
    b = _bdir(env.tmp)
    prov = json.loads((b / "provenance.json").read_text())
    assert not {"description", "peeling_solutions", "peeled_sources", "bad_antennas", "data_root"} & set(prov)
    assert "rejected_by_v_qa" not in prov["hour"]["qa"] and not (b / "peeling_solutions").exists()


# -- qa_exclude, provenance keys ----------------------------------------------------------------------

def _pl(**kw):
    peel = kw.pop("peel", PeelParams())
    d = dict(description="", cal=SimpleNamespace(badants=None), hour_imaging=HourImagingParams(),
             peel_for_band=lambda b: peel)
    d.update(kw)
    return SimpleNamespace(**d)


def test_backlog_keys_only_with_a_backlog_switch():
    assert not hour_lst.backlog_keys(_pl(), 55)
    assert hour_lst.backlog_keys(_pl(description="x"), 55)
    assert hour_lst.backlog_keys(_pl(cal=SimpleNamespace(badants="1,2")), 55)
    assert hour_lst.backlog_keys(_pl(peel=PeelParams(save_solutions="sky")), 55)
    assert hour_lst.backlog_keys(_pl(hour_imaging=HourImagingParams(qa_exclude=False)), 55)


def test_qa_exclude():
    stamps = ["20241223_074350_55MHz", "20241223_074400_55MHz"]
    bad = {"20241223_074350"}
    rec = {"qa": {}}
    assert hour_lst.apply_qa_exclude(_pl(), 55, bad, stamps, rec) == bad and rec == {"qa": {}}
    rec = {"qa": {}}
    keep = _pl(hour_imaging=HourImagingParams(qa_exclude=False))
    assert hour_lst.apply_qa_exclude(keep, 55, bad, stamps, rec) == set()
    assert rec["qa"]["rejected_by_v_qa"] == 1


# -- GPU imager restart (8646127) ---------------------------------------------------------------------

FAKE = r'''
import json, os, sys
print("GSI_SNAP " + json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(json.dumps(dict(req, pid=os.getpid())) + "\n")
    out = {"status": "ok", "files": [], "timing": {"TOTAL": 0.5}}
    if "gpufail" in req["prefix"] and os.getpid() == int(open(os.environ["FIRST_PID"]).read() or 0):
        out = {"status": "error", "error": "RuntimeError: Error setting non-uniform points"}
    if "id" in req:
        out["id"] = req["id"]
    print("GSI_SNAP " + json.dumps(out), flush=True)
'''


@pytest.mark.parametrize("pipelined", [False, True])
def test_gpu_allocation_failure_retries_in_a_fresh_imager(tmp_path, pipelined):
    script = tmp_path / "fake.py"
    script.write_text(FAKE.replace('{"status": "ready"}', '{"status": "ready", "pipeline": True}')
                      if pipelined else FAKE)
    log, first = tmp_path / "req.jsonl", tmp_path / "first_pid"
    env = dict(os.environ, FAKE_LOG=str(log), FIRST_PID=str(first), **({"GSI_SNAP_PIPELINE": "1"} if pipelined else {}))
    d = gpu_image.SnapshotDaemon(sys.executable, str(script), env)
    geom, par = SimpleNamespace(pixels=8, scale=1.0), SimpleNamespace(clean_niter=1)
    d.image(Path("a.ms"), Path("/x/ok"), geom, par)
    first.write_text(str(json.loads(log.read_text().splitlines()[0])["pid"]))
    if pipelined:      # the pipelined client retries in the same process (others may be in flight)
        with pytest.raises(RuntimeError, match="non-uniform"):
            d.image(Path("b.ms"), Path("/x/gpufail"), geom, par)
    else:
        assert d.image(Path("b.ms"), Path("/x/gpufail"), geom, par)
    d.close()
    pids = [json.loads(x)["pid"] for x in log.read_text().splitlines()]
    assert len(pids) == 3 and len(set(pids)) == (1 if pipelined else 2)


# -- scripts/run_backlog.py ---------------------------------------------------------------------------

def test_backlog_window_and_grouping():
    from datetime import datetime, timezone
    rb = _run_backlog()
    t = lambda hh, mm: datetime(2026, 10, 29, hh, mm, tzinfo=timezone.utc)     # noqa: E731
    assert rb.may_start(t(16, 0), "05:30", "15:00", 60)
    assert rb.may_start(t(4, 0), "05:30", "15:00", 60)
    assert not rb.may_start(t(4, 45), "05:30", "15:00", 60)     # would run into the nightly
    assert not rb.may_start(t(9, 0), "05:30", "15:00", 60)      # nightly hours
    plan = [("d1", 1, "bp", [18, 55], {}, ""), ("d1", 2, "bp", [18, 55], {}, ""),
            ("d1", 3, "bp", [18, 55], {}, ""), ("d1", 4, "bp", [55], {}, ""), ("d2", 1, "bp", [55], {}, "")]
    assert [[g[1] for g in grp] for grp in rb.group_plan(plan, 2)] == [[1, 2], [3], [4], [1]]


def test_backlog_config_from_template(tmp_path):
    import yaml
    rb = _run_backlog()
    tpl = _pipeline_yaml(tmp_path, description="GPU backlog.")
    grp = [("2024-12-23", 6, "/bp/B.flagged", [55, 82], {55: "1,2", 82: "3"}, "Compare with A."),
           ("2024-12-23", 7, "/bp/B.flagged", [55, 82], {55: "4", 82: "5"}, "Compare with B.")]
    out = rb.write_group_config(tpl, tmp_path / "out", "Run_GSIbacklog_v1", grp)
    d = yaml.safe_load((out / "pipeline.yaml").read_text())
    assert d["lst_hours"] == [6, 7] and d["bands"] == [55, 82] and d["cal"]["bp"] == "/bp/B.flagged"
    assert d["cal"]["badants"] == {"06h/55": "1,2", "06h/82": "3", "07h/55": "4", "07h/82": "5"}
    assert d["description"] == "GPU backlog. Compare with A. Compare with B."
    assert d["run_label"] == "Run_GSIbacklog_v1" and d["layout"] == "lst"


def test_bandpass_fallback(tmp_path):
    rb = _run_backlog()
    for lab in ("01h", "03h"):
        p = tmp_path / "2024-12-18" / lab / "successful" / "r1" / "tables"
        p.mkdir(parents=True)
        (p / f"calibration_2024-12-18_{lab}.B.flagged").mkdir()
    assert rb.exopipe_bandpass(str(tmp_path), "2024-12-18").endswith("calibration_2024-12-18_01h.B.flagged")
    assert rb.exopipe_bandpass(str(tmp_path), "2024-12-19") is None
