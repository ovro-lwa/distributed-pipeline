"""Clean snapshot imaging, kept peeled MSes and the hour-imaging step (all default off)."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_subband_imaging import hour_image
from gpu_subband_imaging.config import (Config, ConfigError, HourImagingParams, ImagingGeom,
                                        ImagingParams, _validate_lustre_write_path)
from gpu_subband_imaging.stages import image
from gpu_subband_imaging.worker import Worker

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"
GEOM = ImagingGeom(2357, 0.049)


def test_defaults_keep_production_behaviour():
    cfg = Config.load(CONFIG)
    assert cfg.pipeline.imaging.mode == "dirty"
    assert cfg.pipeline.keep_ms_root is None
    assert cfg.cluster.archive_workers == 1
    assert cfg.pipeline.hour_imaging == HourImagingParams()


def test_dirty_command_unchanged():
    cmd = image.dirty_iv_cmd(Path("x.ms"), Path("/w/img_s"), GEOM, ImagingParams(), 4)
    assert cmd[:8] == ["wsclean", "-j", "4", "-pol", "IV", "-niter", "0", "-mem"]
    for flag, val in (("-size", "2357"), ("-scale", "0.049"), ("-taper-inner-tukey", "30.0"),
                      ("-name", "/w/img_s")):
        assert cmd[cmd.index(flag) + 1] == val
    assert cmd[cmd.index("-weight") + 1:cmd.index("-weight") + 3] == ["briggs", "0.0"]
    assert cmd[-1] == "x.ms"


def test_clean_command_matches_orca_snapshot_clean_i():
    p = ImagingParams(mode="clean")
    cmd = image.clean_i_cmd(Path("x.ms"), "/w/n", GEOM, p, 4)
    s = " ".join(cmd)
    for part in ("-pol I", "-niter 50000", "-mgain 0.9999", "-auto-mask 5.0", "-auto-threshold 1.0",
                 "-local-rms", "-horizon-mask 10deg", "-mem 50", "-no-dirty", "-size 2357 2357",
                 "-scale 0.049", "-taper-inner-tukey 30.0", "-weight briggs 0.0",
                 "-no-update-model-required", "-name /w/n x.ms"):
        assert part in s, part
    v = " ".join(image.dirty_v_cmd(Path("x.ms"), "/w/v", GEOM, p, 4))
    for part in ("-pol V", "-niter 0", "-mem 20", "-no-dirty", "-taper-inner-tukey 30.0",
                 "-weight briggs 0.0", "-size 2357 2357"):
        assert part in v, part


def test_imaging_mode_validated():
    with pytest.raises(ConfigError):
        ImagingParams(mode="cleaner")


def test_clean_image_renames_and_times(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        name = cmd[cmd.index("-name") + 1]
        Path(f"{name}-image.fits").write_text("img")
        if "-pol" in cmd and cmd[cmd.index("-pol") + 1] == "I":
            for s in ("psf", "model", "residual"):
                Path(f"{name}-{s}.fits").write_text(s)

    monkeypatch.setattr(image, "_run", fake_run)
    t = {}
    out = image.image(Path("x.ms"), tmp_path / "img_s", GEOM, ImagingParams(mode="clean"), 4,
                      timings=t)
    assert [p.name for p in out] == ["img_s-I-image.fits", "img_s-V-image.fits"]
    assert all(p.exists() for p in out)
    assert sorted(os.listdir(tmp_path)) == ["img_s-I-image.fits", "img_s-V-image.fits"]
    assert set(t) == {"image_clean_i", "image_dirty_v"} and len(calls) == 2


def test_multiple_lustre_write_roots(monkeypatch):
    monkeypatch.setenv("GSI_ALLOWED_LUSTRE_WRITE_ROOT", "/lustre/a:/lustre/b")
    _validate_lustre_write_path("x", "/lustre/b/keep")
    _validate_lustre_write_path("x", "/lustre/a/out")
    with pytest.raises(ConfigError):
        _validate_lustre_write_path("x", "/lustre/c")


def _cfg(tmp_path, **pl):
    p = dict(date="2026-10-24", output_root=str(tmp_path / "out"), hours=["10"], bands=[41, 82],
             keep_ms_root=str(tmp_path / "keep"), hour_imaging=HourImagingParams(
                 ttcalx_image="/venv/bin/ttcalx-image", env={"CUPY_CACHE_DIR": "/c"}),
             movie=SimpleNamespace(), cal=SimpleNamespace(bp="bp", xy="xy"))
    p.update(pl)
    return SimpleNamespace(
        pipeline=SimpleNamespace(**p),
        cluster=SimpleNamespace(work_root=str(tmp_path / "work"), slots=lambda: []),
        subbands=SimpleNamespace(geom=lambda b: ImagingGeom(3122 if b > 60 else 2357, 0.04)))


def test_keep_ms_moves_peeled_ms(tmp_path):
    cfg = _cfg(tmp_path)
    w = Worker(cfg, 41, 0, 0)
    avg = tmp_path / "work" / "s" / "20261024_100003_41MHz_avg.ms"
    (avg / "sub").mkdir(parents=True)
    (avg / "table.dat").write_text("x")
    dst = w.keep_ms("20261024_100003_41MHz", avg)
    assert dst == tmp_path / "keep" / "41MHz" / "2026-10-24" / "10" / "20261024_100003_41MHz.ms"
    assert (dst / "table.dat").read_text() == "x" and not avg.exists()
    assert w.kept_ms_path("20261024_100003") == dst


def test_hour_image_plan_and_command(tmp_path):
    cfg = _cfg(tmp_path)
    for b in (41, 82):
        d = hour_image.kept_dir(cfg, b, "10")
        for s in ("20261024_100013", "20261024_100003"):
            (d / f"{s}_{b}MHz.ms").mkdir(parents=True)
        (d / f".tmp_{b}.ms").mkdir()
    assert hour_image.plan(cfg) == [(82, "10"), (41, "10")]       # largest images first
    ms = hour_image.list_kept(cfg, 41, "10")
    assert [Path(m).name for m in ms] == ["20261024_100003_41MHz.ms", "20261024_100013_41MHz.ms"]
    cmd = hour_image.build_cmd(cfg, 41, Path("/l.txt"), Path("/o/41MHz"))
    assert cmd[:5] == ["/venv/bin/ttcalx-image", "hour", "@/l.txt", "--band", "41"]
    assert cmd[cmd.index("--products") + 1].split(",") == HourImagingParams().products
    env = hour_image.imager_env(cfg, {"PYTHONPATH": "/src", "PATH": "/bin"})
    assert "PYTHONPATH" not in env and env["NUMPY_MADVISE_HUGEPAGE"] == "0"
    assert env["CUPY_CACHE_DIR"] == "/c" and env["PATH"] == "/bin"


def test_hour_image_run_band_writes_timing_json(tmp_path):
    fake = tmp_path / "fake-ttcalx-image"
    fake.write_text(f"""#!{sys.executable}
import json, sys
out = sys.argv[sys.argv.index("--out") + 1]
json.dump(dict(timing=dict(TOTAL=1.5), n_loaded=2, rejected=[],
               products={{"I-Deep-Taper-Robust-0": [dict(files=[out + "-I-Deep-Taper-Robust-0-image.fits"])]}}),
          open(out + "-summary.json", "w"))
""")
    fake.chmod(0o755)
    cfg = _cfg(tmp_path, hour_imaging=HourImagingParams(ttcalx_image=str(fake)))
    d = hour_image.kept_dir(cfg, 41, "10")
    (d / "20261024_100003_41MHz.ms").mkdir(parents=True)
    rec = hour_image.run_band(cfg, 41, "10", gpu=1)
    od = hour_image.out_dir(cfg, 41, "10")
    assert od == tmp_path / "out" / "41MHz" / "2026-10-24" / "10" / "hour_images"
    t = json.loads((od / "41MHz-hour-timing.json").read_text())
    assert t["status"] == "ok" and t["n_ms"] == 1 and t["imager_timing_s"]["TOTAL"] == 1.5
    assert t["files"]["I-Deep-Taper-Robust-0"][0].endswith("41MHz-I-Deep-Taper-Robust-0-image.fits")
    assert rec["wall_s"] >= 0


def test_pipeline_yaml_parses_new_sections(tmp_path):
    for name in ("cluster.yaml", "subbands.yaml"):
        (tmp_path / name).write_text((CONFIG / name).read_text())
    d = yaml.safe_load((CONFIG / "pipeline.yaml").read_text())
    d["imaging"]["mode"] = "clean"
    d["keep_ms_root"] = "/fast/keep"
    d["hour_imaging"] = {"ttcalx_image": "/v/bin/ttcalx-image", "products": ["I-Taper-10min"],
                         "nproc": 4}
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(d))
    c = yaml.safe_load((CONFIG / "cluster.yaml").read_text())
    c.setdefault("defaults", {})["archive_workers"] = 2
    (tmp_path / "cluster.yaml").write_text(yaml.safe_dump(c))
    cfg = Config.load(tmp_path)
    assert cfg.pipeline.imaging.mode == "clean" and cfg.pipeline.keep_ms_root == "/fast/keep"
    assert cfg.pipeline.hour_imaging.products == ["I-Taper-10min"]
    assert cfg.pipeline.hour_imaging.nproc == 4 and cfg.cluster.archive_workers == 2
