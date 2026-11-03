"""No v3 switch set: the worker, orchestrator and imager client behave as 13405ce (review H1-H4, M1-M4).

The v3-only extras (QA row times, the per-row QA append, the safe FITS move, env forwarding) follow
V3Params.active.  See docs/default_path_vs_13405ce.md.
"""
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_subband_imaging import worker as wmod
from gpu_subband_imaging.config import HourImagingParams, V3Params

REPO = Path(__file__).resolve().parents[1]


def _worker(tmp_path, v3):
    pl = SimpleNamespace(date="2026-10-31", output_root=str(tmp_path / "out"), layout="lst", run_label="Run_T",
                         keep_ms_root=None, keep_ms_local=None, v3=v3, sun_cut=True, snapshot_phase_centre="zenith",
                         hour_imaging=HourImagingParams(ttcalx_image="/v/ttcalx-image"),
                         movie=SimpleNamespace(enabled=False, fixed_vmax=20.0), imaging=SimpleNamespace(gpu_python=None),
                         cal=SimpleNamespace(bp="b", xy="x", sources="s", rfi_sources=None))
    cfg = SimpleNamespace(pipeline=pl, cluster=SimpleNamespace(work_root=str(tmp_path / "work")))
    w = wmod.Worker.__new__(wmod.Worker)
    w.cfg, w.band, w.batch = cfg, 55, 2
    w._lst = {f"20261031_1010{k}7_55MHz": 4 for k in range(9)}
    w._archives, w._archive_lock, w._archive_failed = {}, threading.Lock(), False
    w._qa, w._gpu_imager = [], None
    w.work = Path(cfg.cluster.work_root) / "b55_2"
    return w


def _post(tmp_path, monkeypatch, v3):
    w = _worker(tmp_path, v3)
    stamp = "20261031_101007_55MHz"
    moves = []
    monkeypatch.setattr(wmod.image, "v_stats", lambda f: dict(rms=1.0, peak=2.0))

    def fpack(f):
        fz = Path(f"{f}.fz")
        f.rename(fz)
        return fz
    monkeypatch.setattr(wmod.compress, "fpack", fpack)
    real_move = wmod.shutil.move
    monkeypatch.setattr(wmod.shutil, "move", lambda a, b: (moves.append(Path(b).name), real_move(a, b))[1])
    d = w.work / stamp
    d.mkdir(parents=True)
    files = [d / f"img_{stamp}-I-image.fits", d / f"img_{stamp}-V-image.fits"]
    for f in files:
        f.write_text("fits")
    w._post_image(stamp, files)
    return w, stamp, moves


def test_default_qa_row_and_fits_move_are_13405ce(tmp_path, monkeypatch):
    w, stamp, moves = _post(tmp_path, monkeypatch, V3Params())
    assert w._qa == [dict(stamp=stamp, ts="20261031_101007", file=w._archive_paths(stamp)[1].name, rms=1.0, peak=2.0)]
    band = w.band_dir(stamp)
    assert not (band / "QA" / ".snapshot_stats").exists()           # no per-row append
    i, v = (Path(f"{p}.fz") for p in w._archive_paths(stamp))
    assert moves == [i.name, v.name] and i.exists() and v.exists()  # I then V, straight to the final names


def test_v3_qa_row_append_and_safe_move(tmp_path, monkeypatch):
    w, stamp, moves = _post(tmp_path, monkeypatch, V3Params(untar_to=str(tmp_path / "shm")))
    assert "t" in w._qa[0]
    st = list((w.band_dir(stamp) / "QA" / ".snapshot_stats").glob("*.jsonl"))
    assert len(st) == 1 and json.loads(st[0].read_text())["ts"] == "20261031_101007"
    i, v = (Path(f"{p}.fz") for p in w._archive_paths(stamp))
    # V first, the CLEAN I (the "done" mark) last, each through a temporary name
    assert moves == [f".{v.name}.part", f".{i.name}.part"] and i.exists() and v.exists()
    assert not list(i.parent.glob(".*.part")) and not list(v.parent.glob(".*.part"))


def test_orchestrator_forwards_v3_env_only_in_v3_mode(tmp_path, monkeypatch):
    from dataclasses import replace
    from gpu_subband_imaging import orchestrator
    from gpu_subband_imaging.config import Config
    cfg = Config.load(REPO / "examples" / "config")
    for k in ("GSI_PREP_LOOKAHEAD", "XDG_CACHE_HOME", "GSI_V3_ALLOW_PROFILE", "CASARCFILES"):
        monkeypatch.setenv(k, "/x")
    v3keys = {"GSI_PREP_LOOKAHEAD", "XDG_CACHE_HOME", "GSI_V3_ALLOW_PROFILE"}
    cl = replace(cfg.cluster, state_dir=str(tmp_path))
    o = orchestrator.Orchestrator(replace(cfg, cluster=cl), str(tmp_path), "w.sh", list_via_node="n")
    assert "CASARCFILES" in o.worker_env and not v3keys & set(o.worker_env)
    pl3 = replace(cfg.pipeline, v3=V3Params(untar_to="/dev/shm/x"))
    o3 = orchestrator.Orchestrator(replace(cfg, cluster=cl, pipeline=pl3), str(tmp_path), "w.sh", list_via_node="n")
    assert v3keys <= set(o3.worker_env)


def test_v3_defaults_are_inactive():
    v = V3Params()
    assert not v.active and v.stage_attempts == 1


def test_snapshot_daemon_default_imports_only_13405ce_modules():
    """workers/ttcalx_snapshot_daemon.py: the default path imports what 13405ce imported; the pipelined
    path's snapload / image_many are imported only with GSI_SNAP_PIPELINE=1."""
    import ast
    src = (REPO / "workers" / "ttcalx_snapshot_daemon.py").read_text()
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top for a in n.names}
    assert names == {"json", "math", "os", "sys", "traceback", "cupy", "bands", "gridder", "snapshot", "SnapshotImager"}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.get_source_segment(src, main)
    assert "gridder._SVD_CACHE.clear()" in body and "image_many" not in body and "snapload" not in body
