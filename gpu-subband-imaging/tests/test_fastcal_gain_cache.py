"""GSI_PREP_GAIN_CACHE (v3.prep_gain_cache): calibrate_arrays with the cached per-row gain products gives the
same DATA / FLAG bytes as the per-snapshot computation, also for a second snapshot and a new row order."""
import numpy as np

from gpu_subband_imaging.stages import fastcal


def _snap(rng, nant=12, nchan=16):
    a1, a2 = np.triu_indices(nant)
    a1, a2 = a1.astype(np.int32), a2.astype(np.int32)
    n = a1.size
    data = (rng.standard_normal((n, nchan, 4)) + 1j * rng.standard_normal((n, nchan, 4))).astype(np.complex64)
    return data, np.zeros((n, nchan, 4), bool), np.zeros(n, bool), a1, a2


def test_gain_cache_bitwise(monkeypatch):
    rng = np.random.default_rng(3)
    nant, nchan = 12, 16
    J = rng.standard_normal((nant, nchan, 2)) + 1j * rng.standard_normal((nant, nchan, 2))
    dead = np.zeros((nant, 2), bool); dead[3, 1] = True
    freq = np.linspace(50e6, 60e6, nchan)
    for seed in (1, 2):
        for perm in (False, True):
            data, flag, frow, a1, a2 = _snap(np.random.default_rng(seed), nant, nchan)
            if perm:
                p = np.random.default_rng(9).permutation(a1.size)
                data, flag, frow, a1, a2 = data[p], flag[p], frow[p], a1[p], a2[p]
            out = []
            for v in ("0", "1"):
                monkeypatch.setenv("GSI_PREP_GAIN_CACHE", v)
                d, f, r = data.copy(), flag.copy(), frow.copy()
                fastcal.calibrate_arrays(d, f, r, a1, a2, freq, J, dead, "2,5")
                out.append((d, f, r))
            assert out[0][0].tobytes() == out[1][0].tobytes()
            assert out[0][1].tobytes() == out[1][1].tobytes() and out[0][2].tobytes() == out[1][2].tobytes()
