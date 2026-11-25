"""pipeline.yaml fpack_deterministic: snapshot FITS compressed with `fpack -qt 4` (quantization level 4 as before,
the dither seed from the first tile's checksum instead of the clock); default off (`fpack -v`, as v5)."""
import dataclasses
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from gpu_subband_imaging.config import ConfigError, PipelineConfig, load_pipeline
from gpu_subband_imaging.stages import compress

from test_v3_batch import FakeDaemon, STAMPS, _bdir, _config, _worker, env  # noqa: F401
from test_v3_engine import needs_v3_imager

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "config" / "pipeline.yaml"
REAL_FPACK = os.environ.get("FPACK_BIN") or "/opt/devel/pipeline/envs/development/bin/fpack"


def test_fpack_args():
    assert compress.fpack_args() == ["-v"]                       # v5: fpack's own default (seed from the clock)
    assert compress.fpack_args(deterministic=True) == ["-qt", "4", "-v"]


def _pipeline(tmp_path, **extra):
    d = yaml.safe_load(EXAMPLE.read_text())
    d.update(extra)
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_pipeline(p)


def test_config_key(tmp_path):
    assert PipelineConfig.__dataclass_fields__["fpack_deterministic"].default is False
    assert _pipeline(tmp_path).fpack_deterministic is False
    assert _pipeline(tmp_path, fpack_deterministic=True).fpack_deterministic is True
    with pytest.raises(ConfigError, match="fpack_deterministic must be true or false"):
        _pipeline(tmp_path, fpack_deterministic="false")


@pytest.mark.parametrize("deterministic", [False, True])
def test_fpack_command(tmp_path, monkeypatch, deterministic):
    """The options reach fpack; the source FITS is replaced by the .fz."""
    log = tmp_path / "args.txt"
    fake = tmp_path / "fpack"
    fake.write_text(f'#!/bin/sh\necho "$@" > "{log}"\nfor f; do :; done\ncp "$f" "$f.fz"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FPACK_BIN", str(fake))
    compress._fpack_cmd.cache_clear()
    try:
        f = tmp_path / "a.fits"
        f.write_bytes(b"x")
        fz = compress.fpack(f, deterministic=deterministic) if deterministic else compress.fpack(f)
    finally:
        compress._fpack_cmd.cache_clear()
    assert fz == tmp_path / "a.fits.fz" and fz.exists() and not f.exists()
    assert log.read_text().split() == (["-qt", "4"] if deterministic else []) + ["-v", str(f)]


@pytest.mark.skipif(not (os.access(REAL_FPACK, os.X_OK) and shutil.which("funpack", path=str(Path(REAL_FPACK).parent))),
                    reason="needs CFITSIO fpack / funpack")
def test_real_fpack_qt4_gives_the_same_pixels_every_time(tmp_path, monkeypatch):
    """`-qt 4`: two compressions of one image give the same compressed data (DATASUM) and decompressed pixels; the
    default `-v` dithers with a clock seed (different ZDITHER0 / data between runs a second apart)."""
    fits = pytest.importorskip("astropy.io.fits")
    monkeypatch.setenv("FPACK_BIN", REAL_FPACK)
    compress._fpack_cmd.cache_clear()
    img = np.random.default_rng(1).normal(size=(1, 1, 256, 256)).astype(np.float32)
    out = {}
    try:
        for det in (True, False):
            for i in range(2):
                f = tmp_path / f"{det}_{i}.fits"
                fits.PrimaryHDU(img).writeto(f)
                fz = compress.fpack(f, deterministic=det)
                with fits.open(fz, disable_image_compression=True) as h:
                    hdr = h[1].header
                    out[det, i] = (hdr["DATASUM"], hdr["ZDITHER0"], hdr["ZQUANTIZ"])
                un = tmp_path / f"{det}_{i}_un.fits"
                subprocess.run([str(Path(REAL_FPACK).parent / "funpack"), "-O", str(un), str(fz)], check=True)
                out[det, i] += (fits.getdata(un).tobytes(),)
                if not det and i == 0:
                    import time
                    time.sleep(1.1)                      # the clock seed changes every second
    finally:
        compress._fpack_cmd.cache_clear()
    assert out[True, 0] == out[True, 1]
    assert out[False, 0][0] != out[False, 1][0] and out[False, 0][1] != out[False, 1][1]
    assert out[True, 0][2] == out[False, 0][2]                   # the same quantization method, level 4
    dec = np.frombuffer(out[True, 0][3], dtype=">f4").reshape(img.shape)
    assert np.abs(dec - img).max() < 0.5                          # lossy at level 4, as the default


@needs_v3_imager
@pytest.mark.parametrize("on", [False, True])
def test_worker_passes_the_option(env, monkeypatch, on):  # noqa: F811
    calls = []

    def fpack(f, **kw):
        calls.append(kw)
        fz = Path(f"{f}.fz")
        shutil.copy(f, fz)
        return fz
    monkeypatch.setattr(compress, "fpack", fpack)
    cfg = _config(env.tmp, delete_checkpoint=False)
    cfg = dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, fpack_deterministic=on))
    _worker(cfg, FakeDaemon()).run_batch(env.tars)
    assert len(calls) == 2 * len(STAMPS)                         # CLEAN I + pilot V of each snapshot
    assert calls == [({"deterministic": True} if on else {})] * len(calls)
    assert len(list((_bdir(env.tmp) / "snapshots_clean").glob("*.fz"))) == len(STAMPS)
