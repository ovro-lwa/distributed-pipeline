from pathlib import Path
from types import SimpleNamespace

from gpu_subband_imaging.worker import Worker, _hour, _stamp
from gpu_subband_imaging.stages import movie


def _worker(tmp_path: Path) -> Worker:
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(
            date="2026-07-07",
            output_root=str(tmp_path / "out"),
            movie=SimpleNamespace(),
            cal=SimpleNamespace(bp="bp", xy="xy"),
        ),
        cluster=SimpleNamespace(state_dir=str(tmp_path / "state"),
                                work_root=str(tmp_path / "work")),
        subbands=SimpleNamespace(geom=lambda band: SimpleNamespace()),
    )
    return Worker(cfg, 73, 2, 0)


def test_hour_from_snapshot_stamp():
    assert _hour("20260707_073026") == "07"


def test_stamp_supports_archive_and_raw_ms_paths():
    assert _stamp("/data/20260707_073026_73MHz.ms.tar") == "20260707_073026_73MHz"
    assert _stamp("/data/20260707_073026_73MHz.ms") == "20260707_073026_73MHz"


def test_worker_archives_products_under_band_date_hour(tmp_path):
    w = _worker(tmp_path)

    assert w.hour_dir("06") == tmp_path / "out" / "73MHz" / "2026-07-07" / "06"
    assert w.fits_dir("06") == w.hour_dir("06") / "fits"
    assert w.frame_dir("06") == w.hour_dir("06") / "images"


def test_stitch_hours_creates_one_movie_per_hour_and_removes_frames(tmp_path, monkeypatch):
    w = _worker(tmp_path)
    w.cfg.pipeline.movie = SimpleNamespace(fps=20, crf=28, keep_frames=False, enabled=True)
    for hour in ("06", "07"):
        frame_dir = w.frame_dir(hour)
        frame_dir.mkdir(parents=True)
        (frame_dir / f"20260707_{hour}0000.png").write_text("png")

    stitched = []

    def fake_stitch(frame_dir, out_mp4, fps, crf):
        stitched.append((frame_dir, out_mp4, fps, crf))
        out_mp4.write_text("movie")
        return out_mp4

    monkeypatch.setattr(movie, "stitch", fake_stitch)

    w.stitch()

    assert [call[1].name for call in stitched] == [
        "73MHz_2026-07-07_06UTC.mp4",
        "73MHz_2026-07-07_07UTC.mp4",
    ]
    assert not w.frame_dir("06").exists()
    assert not w.frame_dir("07").exists()
