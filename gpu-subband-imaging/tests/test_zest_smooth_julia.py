"""workers/zest_smooth.jl (per-snapshot smoothing exemption list) through julia; skipped without julia."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TEST_JL = Path(__file__).resolve().parent / "julia" / "test_zest_smooth.jl"


def _julia():
    jb = os.environ.get("JULIA_BIN")
    j = os.path.join(jb, "julia") if jb else shutil.which("julia")
    return j if j and os.access(j, os.X_OK) else None


@pytest.mark.skipif(_julia() is None, reason="needs julia")
def test_smooth_exempt_by_elevation():
    r = subprocess.run([_julia(), "--startup-file=no", str(TEST_JL)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
