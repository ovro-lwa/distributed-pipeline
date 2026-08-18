import numpy as np

from gpu_subband_imaging.stages import movie
from gpu_subband_imaging.stages.movie import (
    _downsample,
    _finite_rms,
    _horizon_mask,
    _utc_label,
    robust_rms,
)


def test_downsample_shape_and_mean():
    a = np.ones((1024, 1024))
    d = _downsample(a, 256)
    assert d.shape[0] <= 256 and np.isclose(d.mean(), 1.0)


def test_robust_rms_on_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 2.0, (512, 512))
    assert abs(robust_rms(a) - 2.0) < 0.2


def test_horizon_mask_blanks_corners_and_keeps_center():
    a = np.ones((100, 100))
    masked = _horizon_mask(a, 0.49)
    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[0, -1])
    assert np.isnan(masked[-1, 0])
    assert np.isnan(masked[-1, -1])
    assert masked[50, 50] == 1.0


def test_finite_rms_ignores_nan_mask():
    a = np.array([0.0, 1.0, 2.0, np.nan])
    assert abs(_finite_rms(a) - 1.4826) < 1e-6


def test_utc_label_uses_fits_filename_timestamp():
    path = "/tmp/img_20260720_071206_23MHz-I-image.fits"
    assert _utc_label(path) == "UTC time: 2026-07-20 07:12:06"
    assert _utc_label("/tmp/no_timestamp.fits") is None


def test_render_frame_adds_timestamp_and_colorbar(tmp_path, monkeypatch):
    img = np.zeros((100, 100))
    monkeypatch.setattr(movie, "_load2d", lambda path: img)
    out = tmp_path / "frame.png"

    movie.render_frame("img_20260720_071206_23MHz-I-image.fits", out,
                       vmax=1.0, max_px=100, horizon_mask=True)

    import matplotlib.image as mpimg
    rendered = mpimg.imread(out)
    assert rendered.shape[0] == 100
    assert rendered.shape[1] > 100
    assert np.max(rendered[:25, :90, :3]) > 0.9
    assert np.max(rendered[:, 100:, :3]) > 0.9
    assert np.max(rendered[0, -1, :3]) < 0.05


def test_band_vmax_defaults_to_ten_rms_over_masked_disk(monkeypatch):
    img = np.zeros((100, 100))
    img[60:80, 40:60] = 2.0
    monkeypatch.setattr(movie, "_load2d", lambda path: img)

    vmax = movie.band_vmax(["dummy.fits"], "adaptive_i", 20.0, 100,
                           horizon_mask=True, horizon_radius_fraction=0.49)

    masked = movie._horizon_mask(img, 0.49)
    assert np.isclose(vmax, max(1.0, 10.0 * movie._finite_rms(masked)))


def test_stitch_crops_odd_dimensions_and_replaces_atomically(tmp_path, monkeypatch):
    frame_dir = tmp_path / "images"
    frame_dir.mkdir()
    (frame_dir / "frame.png").write_text("png")
    out = tmp_path / "movie.mp4"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        tmp_out = cmd[-1]
        assert tmp_out.endswith(".tmp.mp4")
        with open(tmp_out, "w") as f:
            f.write("movie")

    monkeypatch.setattr(movie.subprocess, "run", fake_run)

    assert movie.stitch(frame_dir, out, fps=20, crf=28, ffmpeg_bin="ffmpeg") == out

    cmd = calls[0][0]
    assert "-vf" in cmd
    assert "crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" in cmd
    assert out.read_text() == "movie"
    assert not (tmp_path / ".movie.mp4.tmp.mp4").exists()
