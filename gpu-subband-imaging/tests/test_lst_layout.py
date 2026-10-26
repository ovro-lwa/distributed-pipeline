"""layout=lst: LST grouping, Sun cut, main-pipeline paths, QA and product names."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_subband_imaging import hour_lst, lst
from gpu_subband_imaging.config import ConfigError, PipelineConfig
from gpu_subband_imaging.worker import Worker


def test_lst_hours_and_phase_centre():
    # 2026-10-24: LST 04h at OVRO-LWA is ~09:44-10:44 UTC
    assert lst.lst_hours(["20261024_094000_55MHz.ms.tar", "20261024_095000_55MHz.ms.tar",
                          "20261024_104000_55MHz.ms.tar", "20261024_105000_55MHz.ms.tar"]) == [3, 4, 4, 5]
    assert lst.phase_centre(4) == "04h30m00s 37d12m57.057s"
    assert lst.label(4) == "04h"
    assert lst.timestamp("/x/20261024_100003_55MHz.ms.tar") == "20261024_100003"


def test_sun_cut_matches_main_policy():
    # 2026-10-24: 06:00 UTC is dark; 02:00 UTC (Sun setting, above -18 deg) and 20:00 are not
    assert lst.sun_ok(["20261024_060000", "20261024_020000", "20261024_200000"]) == [True, False, False]


def test_group_keeps_requested_lst_hours_after_sun_cut():
    names = [f"/d/20261024_{h:02d}0003_55MHz.ms.tar" for h in (6, 10, 20)]
    g = lst.group(names, hours=[4], sun_cut=True)
    assert g == {4: [names[1]]}


def test_band_dir_is_main_pipeline_layout(tmp_path):
    d = lst.band_dir(str(tmp_path), 4, "2026-10-24", "Run_20261026_031500", 55)
    assert d == tmp_path / "04h" / "2026-10-24" / "Run_20261026_031500" / "55MHz"
    assert lst.pilot_name(55, "20261024_100003") == "55MHz-Pilot-Snapshot-20261024_100003-image.fits"
    assert lst.clean_name(55, "20261024_100003") == "55MHz-Clean-Snapshot-20261024_100003-image.fits"


def _worker(tmp_path, layout="lst"):
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(date="2026-10-24", output_root=str(tmp_path / "out"),
                                 layout=layout, run_label="Run_X", movie=SimpleNamespace(),
                                 keep_ms_root=str(tmp_path / "keep"),
                                 cal=SimpleNamespace(bp="bp", xy="xy")),
        cluster=SimpleNamespace(state_dir=str(tmp_path / "state"), work_root=str(tmp_path / "work")),
        subbands=SimpleNamespace(geom=lambda band: SimpleNamespace()),
    )
    return Worker(cfg, 55, 0, 0)


def test_worker_lst_paths(tmp_path):
    w = _worker(tmp_path)
    s = "20261024_100003_55MHz"
    bd = tmp_path / "out" / "04h" / "2026-10-24" / "Run_X" / "55MHz"
    i, v = w._archive_paths(s)
    assert i == bd / "snapshots_clean" / "55MHz-Clean-Snapshot-20261024_100003-image.fits"
    assert v == bd / "snapshots" / "55MHz-Pilot-Snapshot-20261024_100003-image.fits"
    assert w._fits_path_for_stamp(s) == Path(f"{i}.fz")
    assert w._frame_dir_for_stamp(s) == bd / "Movies" / "frames"
    assert w.kept_ms_path(s) == tmp_path / "keep" / "55MHz" / "2026-10-24" / "04h" / f"{s}.ms"


def test_worker_utc_paths_unchanged(tmp_path):
    w = _worker(tmp_path, layout="utc")
    s = "20261024_100003_55MHz"
    assert w._fits_path_for_stamp(s) == (tmp_path / "out" / "55MHz" / "2026-10-24" / "10" / "fits" /
                                         f"img_{s}-I-image.fits.fz")


def test_qa_reject_is_orca_rule():
    rms = [1.0] * 20 + [5.0, 0.4]
    bad, med, sd = hour_lst.qa_reject(rms)
    assert bad == [20, 21] and med == 1.0


def test_write_qa_files(tmp_path):
    rows = [dict(ts=f"20261024_1000{i:02d}", rms=1.0 + 0.01 * i, peak=10.0,
                 file=f"55MHz-Pilot-Snapshot-20261024_1000{i:02d}-image.fits") for i in range(10)]
    rows.append(dict(ts="20261024_100100", rms=50.0, peak=99.0, file="x.fits"))
    (tmp_path / ".snapshot_stats").mkdir()
    (tmp_path / ".snapshot_stats" / "b0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))
    got = hour_lst.read_stats(tmp_path)
    assert [r["ts"] for r in got] == sorted(r["ts"] for r in rows)
    bad, summ = hour_lst.write_qa(tmp_path, 55, got)
    assert bad == {"20261024_100100"} and summ["rejected"] == 1
    lines = (tmp_path / "snapshot_qa_55MHz.csv").read_text().splitlines()
    assert lines[0] == "snapshot_index,scan_number,timestamp_utc,rms_jy_per_beam,peak_jy_per_beam,flagged,filename"
    assert lines[-1].startswith("10,11,2026-10-24T10:01:00Z,50.0,99.0,true,")
    assert (tmp_path / "pilot_qa_rejected.txt").read_text().count("\n") == 2


def test_place_products_names(tmp_path):
    work, bd = tmp_path / "w", tmp_path / "b"
    work.mkdir()
    for n in ["55MHz-I-Deep-Taper-Robust-0.75-image.fits", "55MHz-I-Deep-NoTaper-Robust-0-psf.fits",
              "55MHz-V-Taper-Deep-image.fits", "55MHz-I-Taper-10min-t0005-model.fits",
              "55MHz-V-Taper-10min-t0000-image.fits", "55MHz-summary.json"]:
        (work / n).write_text("x")
    stamps = [f"20261024_10{m:02d}03_55MHz" for m in range(60)]
    moved = sorted(Path(p).relative_to(bd).as_posix() for p in hour_lst.place_products(work, bd, stamps))
    assert moved == [
        "I/10min/55MHz-I-Taper-10min-20261024_105438-model.fits",
        "I/deep/55MHz-I-Deep-NoTaper-Robust-0-psf-20261024_102938.fits",
        "I/deep/55MHz-I-Deep-Taper-Robust-0.75-image-20261024_102938.fits",
        "V/10min/55MHz-V-Taper-10min-20261024_100438-image.fits",
        "V/deep/55MHz-V-Taper-Deep-image-20261024_102938.fits",
    ]


def test_config_lst_validation():
    base = dict(date="2026-10-24", hours="all", bands=[55], batch_size=10, batch_timeout_seconds=None,
                max_retries=1, data_root="/d", output_root="/o", chanbin=4, cal=None, peel=None,
                peel_overrides={}, imaging=None, movie=None, do_aoflag=False, do_badants=True, env=None)
    with pytest.raises(ConfigError):
        PipelineConfig(**base, layout="lst")                        # no run_label
    with pytest.raises(ConfigError):
        PipelineConfig(**base, snapshot_phase_centre="lst")         # needs layout=lst
    p = PipelineConfig(**base, layout="lst", run_label="Run_X", lst_hours=[4, 5])
    assert p.run_key == "2026-10-24_lst04-05"
    assert PipelineConfig(**base).run_key == "2026-10-24"
