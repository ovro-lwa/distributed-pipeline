"""The real Julia zest daemon on a synthetic slot: read_slot / write_slot! against numpy.

ZEST_TEST_IO skips the peel, so this checks exactly the slot I/O of workers/zest_daemon.jl
(cross-row gather with lo/hi antennas, flag OR over pols, ComplexF32 <-> F64, the scatter,
autos untouched, state PEELED), the id-tagged replies, the PREPPED check and RESET_WARM.

Needs julia (JULIA_BIN/julia or on PATH), TTCALX_DIR (TTCalX with PyCall + python-casacore,
as workers/run_worker.sh) and a sources file (GSI_TEST_ZEST_SOURCES, default the gsi-v2
7-source model); skipped otherwise.  Starting the daemon takes about a minute.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from gpu_subband_imaging.stages.peel import ZestDaemon
from gpu_subband_imaging.v3 import slots
from gpu_subband_imaging.v3.snap import SnapArrays

DAEMON_JL = Path(__file__).resolve().parents[1] / "workers" / "zest_daemon.jl"
SOURCES = os.environ.get("GSI_TEST_ZEST_SOURCES",
                         "/opt/devel/pipeline/gsi-v2/models/sources_7src_tauA.json")


def _julia():
    jb = os.environ.get("JULIA_BIN")
    j = os.path.join(jb, "julia") if jb else shutil.which("julia")
    return j if j and os.access(j, os.X_OK) else None


pytestmark = pytest.mark.skipif(
    not (_julia() and os.path.isdir(os.environ.get("TTCALX_DIR", "/nonexistent"))
         and os.path.exists(SOURCES)),
    reason="needs julia, TTCALX_DIR and a sources file")


def synthetic(nant=9, nchan=6, seed=1):
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(nant)
    a1, a2 = a1.astype(np.int32), a2.astype(np.int32)
    swap = (np.arange(a1.size) % 3 == 1) & (a1 != a2)     # some rows as (hi, lo)
    a1[swap], a2[swap] = a2[swap].copy(), a1[swap].copy()
    perm = rng.permutation(a1.size)                       # not in baseline order
    a1, a2 = a1[perm], a2[perm]
    nrow = a1.size
    data = (rng.standard_normal((nrow, nchan, 4)) * 1e3 +
            1j * rng.standard_normal((nrow, nchan, 4)) * 1e3).astype(np.complex64)
    t = 5.2e9 + np.zeros(nrow)
    pos = np.array([-2409150.0, -4478573.0, 3838617.0]) + rng.standard_normal((nant, 3)) * 100
    return SnapArrays(
        stem="20261031_101007_23MHz", antenna1=a1, antenna2=a2,
        uvw=rng.standard_normal((nrow, 3)) * 100, data=data,
        flag=rng.random((nrow, nchan, 4)) < 0.2, weight=np.ones((nrow, 4), np.float32),
        sigma=np.ones((nrow, 4), np.float32), flag_row=np.zeros(nrow, bool), time=t,
        time_centroid=t, exposure=np.full(nrow, 10.0), interval=np.full(nrow, 10.0),
        meta=dict(freqs=np.linspace(22.6e6, 27.1e6, nchan), chan_width=np.full(nchan, 95703.0),
                  phase_dir=np.array([1.2, 0.649]), ant_pos=pos))


def reference(s, perturb):
    """write_ms_fast!(read_ms_fast(...)) in numpy, with the ZEST_TEST_IO change."""
    data, flag = s.data.copy(), s.flag.copy()
    cross = np.nonzero(s.antenna1 != s.antenna2)[0]      # baseline α = α-th cross row
    v = s.data[cross].astype(np.complex128)
    f = s.flag[cross].any(axis=2)
    if perturb:
        v = v * (1.0001 * complex(np.cos(0.3), np.sin(0.3)))
        f[::7] = ~f[::7]
    data[cross] = v.astype(np.complex64)
    flag[cross] = f[:, :, None]
    return data, flag


def _start(mode):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return ZestDaemon(_julia(), os.environ["TTCALX_DIR"], str(DAEMON_JL), SOURCES, env=env,
                      zest_env={"ZEST_DEVICE": "cpu", "ZEST_MODEL": "exact", "ZEST_COLUMN": "DATA",
                                "ZEST_TEST_IO": mode, "JULIA_NUM_THREADS": "2"},
                      start_timeout=900)


@pytest.mark.parametrize("mode", ["identity", "perturb"])
def test_daemon_slot_io_matches_numpy(mode, tmp_path):
    s = synthetic()
    slot = tmp_path / "gsi3_slot"
    slots.write_slot(slot, s)
    d = _start(mode)
    try:
        d.zest_slot(slot, timeout=300)
        assert slots.get_state(slot) == slots.PEELED
        v = slots.view_slot(slot, writable=False)
        ref_data, ref_flag = reference(s, mode == "perturb")
        assert v.flag.tobytes() == ref_flag.tobytes()
        autos = s.antenna1 == s.antenna2
        assert v.data[autos].tobytes() == s.data[autos].tobytes()      # autos untouched
        if mode == "identity":
            assert v.data.tobytes() == ref_data.tobytes()
        else:   # Julia vs numpy complex multiply: equal to the last float32 ulp at most
            rel = np.abs(v.data - ref_data) / np.maximum(np.abs(ref_data), 1e-30)
            assert rel.max() <= 2.5e-7 and (v.data == ref_data).mean() > 0.99
            assert not np.array_equal(v.data, s.data)
        for f in ("uvw", "weight", "time", "antenna1"):                 # nothing else changes
            assert getattr(v, f).tobytes() == getattr(s, f).tobytes()
        # a slot that is not PREPPED is refused; RESET_WARM answers
        slots.set_state(slot, slots.FREE)
        with pytest.raises(RuntimeError, match="expected PREPPED"):
            d.zest_slot(slot, timeout=120)
        d.reset_warm(timeout=60)
        assert d.restarts == 0
    finally:
        d.close()


@pytest.mark.parametrize("key", ["ZEST_SMOOTH_BELOW_MHZ", "ZEST_SMOOTH_EXEMPT_FLUX",
                                 "ZEST_RFI_SMOOTH_FLUX", "ZEST_SMOOTH_EXEMPT_BELOW_EL_DEG"])
def test_daemon_rejects_negative_smoothing_switches(key):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with pytest.raises(RuntimeError, match=f"{key} must be >= 0"):
        ZestDaemon(_julia(), os.environ["TTCALX_DIR"], str(DAEMON_JL), SOURCES, env=env,
                   zest_env={"ZEST_DEVICE": "cpu", "ZEST_MODEL": "exact", "ZEST_COLUMN": "DATA",
                             key: "-1", "JULIA_NUM_THREADS": "2"},
                   start_timeout=900)


PROD_PEEL = {"ZEST_SOLVER": "rk4", "ZEST_MAXITER": "5", "ZEST_TOLERANCE": "0.0001", "ZEST_MINUVW": "0.0",
             "ZEST_PEELITER": "2", "ZEST_MIN_SOURCE_ELEVATION_DEG": "-90", "ZEST_REQUIRE_CONVERGENCE": "false",
             "ZEST_UVW": "geometric", "ZEST_FLAG_MODEL": "true", "ZEST_RESUBTRACT": "true", "ZEST_FAST_IO": "true",
             "ZEST_FLAG_ZERO": "true", "ZEST_RESIDUAL_GUARD": "true"}


@pytest.mark.parametrize("engine", ["daemon", "sky_rfi"])
def test_save_solutions_both_engines(engine, tmp_path):
    """peel.save_solutions with a real (CPU) peel of a synthetic slot: the daemon writes
    <slot>.peelsol/sky.{bin,tsv}, every model source (elevation threshold -90) with its 0-based
    index, the gains in the layout stages/peel_solutions.convert reads."""
    import json
    from gpu_subband_imaging.config import model_source_names
    from gpu_subband_imaging.stages import peel_solutions
    s = synthetic(seed=4)
    slot = tmp_path / "gsi3_slot"
    slots.write_slot(slot, s)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    zenv = dict(PROD_PEEL, ZEST_DEVICE="cpu", ZEST_MODEL="exact", ZEST_COLUMN="DATA", JULIA_NUM_THREADS="2",
                ZEST_ENGINE=engine, ZEST_SAVE_SOLUTIONS="sky,rfi")
    d = ZestDaemon(_julia(), os.environ["TTCALX_DIR"], str(DAEMON_JL), SOURCES, env=env, zest_env=zenv,
                   start_timeout=900)
    try:
        d.zest_slot(slot, timeout=600)
    finally:
        d.close()
    sol = peel_solutions.sol_dir(slot)
    assert sorted(p.name for p in sol.iterdir()) == ["sky.bin", "sky.tsv"]      # no RFI model: no rfi pass
    files = peel_solutions.convert(sol, tmp_path / "out", {"engine": engine}, peel_solutions.slot_axes(slot),
                                   ms_name="x_avg.ms")
    names = model_source_names(SOURCES)
    with np.load(files[0]) as z:
        assert z["source_names"].tolist() == names
        assert z["source_indices"].tolist() == list(range(len(names)))
        assert z["gains"].shape == (len(names), 4, 9, 6, 1) and z["gains"].dtype == np.complex64
        assert np.isfinite(z["gains"]).all() and (z["niters"] >= 1).all()
        assert json.loads(str(z["metadata_json"]))["stage"] == "sky"
        # an accepted source's gains moved off the identity start
        acc = z["accepted"]
        if acc.any():
            g = z["gains"][acc]
            assert not np.allclose(g[:, 0], 1) or not np.allclose(g[:, 1], 0)
