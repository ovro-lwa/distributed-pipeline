"""gsi v3 step 1: stream the snapshot tar into tmpfs (stage_ms.stage(untar_to=...))."""
import io
import tarfile

import pytest

from gpu_subband_imaging.stages import stage_ms

STAMP = "20261031_101007_23MHz"


def _fake_ms(root):
    ms = root / f"{STAMP}.ms"
    (ms / "ANTENNA").mkdir(parents=True)
    (ms / "table.dat").write_bytes(b"main" * 1000)
    (ms / "table.f0").write_bytes(bytes(range(256)) * 300)
    (ms / "ANTENNA" / "table.dat").write_bytes(b"ant" * 77)
    (ms / "empty").write_bytes(b"")
    return ms


def _tar(tmp_path):
    src = tmp_path / "src"
    ms = _fake_ms(src)
    tar = tmp_path / "lustre" / f"{STAMP}.ms.tar"
    tar.parent.mkdir()
    with tarfile.open(tar, "w") as t:
        t.add(ms, arcname=ms.name)
    return tar, ms


def _tree(root):
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def test_stream_untar_equals_copy_then_extract(tmp_path):
    tar, ms = _tar(tmp_path)
    v2 = stage_ms.stage(str(tar), tmp_path / "work_v2")
    v3 = stage_ms.stage(str(tar), tmp_path / "work_v3", untar_to=tmp_path / "shm" / STAMP)
    assert v3 == tmp_path / "shm" / STAMP / f"{STAMP}.ms"
    assert _tree(v3) == _tree(v2) == _tree(ms)
    assert (tmp_path / "work_v3").is_dir()                     # averaged MS goes there
    assert list((tmp_path / "work_v3").iterdir()) == []        # no local tar copy
    assert tar.exists()


def test_truncated_tar_is_rejected(tmp_path, caplog):
    tar, _ = _tar(tmp_path)
    data = tar.read_bytes()
    # cut at a header boundary (after the first member), where tarfile itself sees no error
    with tarfile.open(tar) as t:
        members = t.getmembers()
    cut = members[2].offset
    tar.write_bytes(data[:cut])
    with pytest.raises(tarfile.ReadError):
        stage_ms.stage(str(tar), tmp_path / "w", untar_to=tmp_path / "shm")
    assert caplog.text.count("STAGE_RETRY") == 2               # three attempts
    assert not (tmp_path / "shm" / f"{STAMP}.ms").exists()
    tar.write_bytes(data[:members[3].offset_data + 100])     # inside a member's data
    with pytest.raises(tarfile.ReadError):
        stage_ms.stage(str(tar), tmp_path / "w", untar_to=tmp_path / "shm")


def test_unexpected_members_are_rejected(tmp_path):
    tar = tmp_path / f"{STAMP}.ms.tar"
    with tarfile.open(tar, "w") as t:
        info = tarfile.TarInfo("../evil")
        info.size = 3
        t.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(tarfile.ReadError, match="unexpected member"):
        stage_ms.stream_untar(tar, tmp_path / "shm")
    assert not (tmp_path / "evil").exists()


def test_missing_tar_fails_without_retries(tmp_path, caplog):
    with pytest.raises(FileNotFoundError):
        stage_ms.stage(str(tmp_path / f"{STAMP}.ms.tar"), tmp_path / "w", untar_to=tmp_path / "shm")
    assert "STAGE_RETRY" not in caplog.text


def test_transient_read_error_is_retried(tmp_path, monkeypatch, caplog):
    tar, ms = _tar(tmp_path)
    real, calls = stage_ms._extract_stream, []

    def flaky(source, dest, name):
        calls.append(1)
        if len(calls) == 1:
            (dest / name).mkdir(parents=True)
            (dest / name / "partial").write_bytes(b"x")
            raise OSError("Lustre hiccup")
        return real(source, dest, name)

    monkeypatch.setattr(stage_ms, "_extract_stream", flaky)
    out = stage_ms.stream_untar(tar, tmp_path / "shm")
    assert len(calls) == 2 and "STAGE_RETRY" in caplog.text
    assert _tree(out) == _tree(ms)                             # the partial file is gone
