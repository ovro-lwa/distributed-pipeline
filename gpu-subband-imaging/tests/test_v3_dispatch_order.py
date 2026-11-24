"""Scheduling switches of gsi v5 (both opt-in): v3.retry_on_checkpoint_node (a band-hour's first retry waits for the
node holding its local checkpoint) and pipeline.dispatch_band_order (band order inside an LST hour); and the tmpfs
raw-MS count of _plan_shm.  Ported from exp/gsi-v5-sched (676640f) without the lanes / gpu_share parts."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_subband_imaging import config as config_mod
from gpu_subband_imaging.config import Slot, V3Params
from gpu_subband_imaging.ledger import FAILED, QUEUED, Job
from gpu_subband_imaging.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "examples" / "config"

AOF = dict(prep_inmem="on", peel_io="shm", snapshot_io="shm", hour_source="memory", write_kept_ms=False,
           aoflag="hour")


# -- config ---------------------------------------------------------------------------------------------------
def test_defaults_are_off():
    assert V3Params().retry_on_checkpoint_node is False
    pl = config_mod.load_pipeline(CONFIG / "pipeline.yaml")
    assert pl.dispatch_band_order == []


def test_dispatch_band_order_is_read(tmp_path):
    c = tmp_path / "cfg"
    shutil.copytree(CONFIG, c)
    d = yaml.safe_load((c / "pipeline.yaml").read_text())
    d["dispatch_band_order"] = [73, "55", 82]
    (c / "pipeline.yaml").write_text(yaml.safe_dump(d))
    assert config_mod.load_pipeline(c / "pipeline.yaml").dispatch_band_order == [73, 55, 82]


# -- orchestrator -----------------------------------------------------------------------------------------------
def _orch(order=(), v3=None):
    o = Orchestrator.__new__(Orchestrator)
    pl = SimpleNamespace(dispatch_band_order=list(order),
                         v3=V3Params(**(v3 or {**AOF, "checkpoint": "local", "retry_on_checkpoint_node": True})))
    o.cfg = SimpleNamespace(pipeline=pl)
    o.slots = [Slot("n1", 0), Slot("n2", 0)]
    o.group_of = {(b, 0): "00h" for b in (18, 55, 82)}
    o.group_of.update({(b, 1): "01h" for b in (18, 55, 82)})
    return o


def _st(jobs, node_of=None):
    node_of = node_of or {}
    return {(j.band, "00h" if j.batch == 0 else "01h"): {"batches": [j], "hour": None, "node": node_of.get(j.band)}
            for j in jobs}


def test_failed_band_hour_waits_for_its_checkpoint_node():
    o = _orch()
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    st = _st([failed, queued], {18: "n1"})
    order = {"00h": 0, "01h": 1}
    assert o._choose(Slot("n2", 0), [failed, queued], st, False, order) is queued     # not the redo elsewhere
    assert o._choose(Slot("n1", 1), [failed, queued], st, False, order) is failed     # its node, any GPU
    second = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=2)              # 2nd retry: anywhere
    assert o._choose(Slot("n2", 0), [second, queued], _st([second, queued], {18: "n1"}), False, order) is second
    gone = Job(18, 0, 358, status=FAILED, slot="n9:gpu0", attempts=1)                # node left the cluster
    assert o._choose(Slot("n2", 0), [gone, queued], _st([gone, queued], {18: "n9"}), False, order) is gone


@pytest.mark.parametrize("v3", [dict(AOF, checkpoint="local"),                       # default: off, as before
                                dict(AOF, checkpoint="local", retry_on_checkpoint_node=True, archive="iv",
                                     archive_root="/tmp/a")])
def test_no_pinning_by_default_or_without_local_checkpoints(v3):
    o = _orch(v3=v3)
    failed = Job(18, 0, 358, status=FAILED, slot="n1:gpu0", attempts=1)
    queued = Job(55, 1, 358, status=QUEUED)
    assert o._choose(Slot("n2", 0), [failed, queued], _st([failed, queued], {18: "n1"}), False,
                     {"00h": 0, "01h": 1}) is failed


def test_dispatch_band_order_inside_an_lst_hour():
    jobs = [Job(b, h, 358) for h in (0, 1) for b in (18, 55, 82)]
    order = {"00h": 0, "01h": 1}

    def seq(o):
        left, out = list(jobs), []
        while left:
            j = o._choose(Slot("n1", 0), left, _st(left), False, order)
            out.append((j.band, j.batch))
            left.remove(j)
        return out
    assert seq(_orch()) == [(18, 0), (55, 0), (82, 0), (18, 1), (55, 1), (82, 1)]
    assert seq(_orch(order=[82, 55])) == [(82, 0), (55, 0), (18, 0), (82, 1), (55, 1), (18, 1)]


# -- _plan_shm ----------------------------------------------------------------------------------------------------
class _W:
    UNTAR_SPACE_MARGIN = 1.0

    def __init__(self, root):
        self.root = root
        self._untar_disabled = False

    def _untar_root(self):
        return self.root


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="needs /dev/shm")
@pytest.mark.parametrize("lookahead,feeders,raw", [(8, 3, 3), (2, 8, 3), (4, 5, 5)])
def test_plan_shm_counts_one_raw_ms_per_running_feeder(tmp_path, monkeypatch, lookahead, feeders, raw):
    """tmpfs holds min(lookahead + 1, max_feeders) raw MSes at once (not lookahead): untar stays in tmpfs when that
    fits next to the slots, and moves to work_root when it does not."""
    from gpu_subband_imaging.v3 import worker_v3
    tar_b = 1 << 30
    tar = tmp_path / "a.tar"
    with open(tar, "wb") as f:                     # sparse: only the size matters
        f.truncate(tar_b)
    need = (lookahead + 2) * 130e6 * 1.2 + raw * tar_b

    def run(free):
        monkeypatch.setattr(worker_v3.shutil, "disk_usage", lambda p: SimpleNamespace(free=free))
        b = worker_v3.V3Batch.__new__(worker_v3.V3Batch)
        b.w = _W(Path("/dev/shm") / "gsi-test-not-created" / "untar")      # parent missing: /dev/shm is used
        b.tars = [str(tar)]
        b.lookahead, b.write_kept, b.band, b.batch = lookahead, False, 55, 0
        b.cfg = SimpleNamespace(cluster=SimpleNamespace(max_feeders=feeders))
        b._plan_shm()
        return b.w._untar_disabled
    assert run(need + 1e6) is False
    assert run(need - 1e6) is True
