"""stages/gpu_image.SnapshotDaemon against a fake workers/ttcalx_snapshot_daemon.py (no GPU).

Default = 13405ce: one request at a time, requests without "id", no reply deadline unless
GSI_SNAP_REPLY_TIMEOUT_S is set (then the daemon is killed).  GSI_SNAP_PIPELINE=1: ids, replies
matched by id only; an id-less reply or a timeout kills the daemon and fails what is outstanding.
"""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_subband_imaging.stages import gpu_image

FAKE = r'''
import json, os, sys, time
mode = os.environ.get("FAKE_MODE", "serial")
log = open(os.environ["FAKE_LOG"], "a")
ready = {"status": "ready"}
if mode.startswith("pipe"):
    ready["pipeline"] = True
print("GSI_SNAP " + json.dumps(ready), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    log.write(json.dumps(dict(req, pid=os.getpid())) + "\n"); log.flush()
    if mode == "hang" or (mode == "pipe_hang" and req["prefix"].endswith("slow")):
        time.sleep(30)
    out = {"status": "error", "error": "echo " + req["prefix"]}     # the client raises with it
    if "ok" in req["prefix"]:
        out = {"status": "ok", "files": [], "timing": {"TOTAL": 0.5}}
    if mode.startswith("pipe") and mode != "pipe_noid":
        out["id"] = req["id"]
    print("noise line", flush=True)
    print("GSI_SNAP " + json.dumps(out), flush=True)
'''

GEOM = SimpleNamespace(pixels=64, scale=1.0)
PAR = SimpleNamespace(clean_niter=10)


def _daemon(tmp_path, mode, **env):
    script = tmp_path / "fake_daemon.py"
    script.write_text(FAKE)
    log = tmp_path / "requests.jsonl"
    e = dict(FAKE_MODE=mode, FAKE_LOG=str(log), **env)
    return gpu_image.SnapshotDaemon(sys.executable, str(script), e), log


def _requests(log):
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def test_default_is_serial_without_ids_or_deadline(tmp_path):
    d, log = _daemon(tmp_path, "serial")
    assert not d.pipelined and d.reply_timeout is None
    t = {}
    assert d.image(Path("a.ms"), Path("/x/ok_a"), GEOM, PAR, timings=t) == \
        [Path("/x/ok_a-I-image.fits"), Path("/x/ok_a-V-image.fits")]
    assert t == {"image_gpu": 0.5}
    errs = {}

    def run(name):
        try:
            d.image(Path(f"{name}.ms"), Path(f"/x/{name}"), GEOM, PAR)
        except RuntimeError as e:
            errs[name] = str(e)
    th = [threading.Thread(target=run, args=(n,)) for n in ("b", "c", "e")]
    [x.start() for x in th]
    [x.join() for x in th]
    d.close()
    assert {k: v.endswith(f"echo /x/{k}") for k, v in errs.items()} == {"b": True, "c": True, "e": True}
    reqs = _requests(log)
    # the 13405ce request: exactly these keys, no id
    assert all(set(r) == {"ms", "prefix", "pixels", "scale", "niter", "pid"} for r in reqs)
    assert len({r["pid"] for r in reqs}) == 1


def test_default_deadline_only_when_set_kills_and_restarts(tmp_path):
    d, log = _daemon(tmp_path, "hang", GSI_SNAP_REPLY_TIMEOUT_S="1")
    assert d.reply_timeout == 1.0
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="no reply within 1 s"):
        d.image(Path("a.ms"), Path("/x/a"), GEOM, PAR)
    assert time.monotonic() - t0 < 20
    with pytest.raises(RuntimeError, match="no reply within 1 s"):
        d.image(Path("b.ms"), Path("/x/b"), GEOM, PAR)
    d.close()
    assert len({r["pid"] for r in _requests(log)}) == 2          # killed, then a new daemon


def test_pipelined_matches_replies_by_id(tmp_path):
    d, log = _daemon(tmp_path, "pipe", GSI_SNAP_PIPELINE="1")
    assert d.pipelined and d.reply_timeout == 900.0
    errs = {}

    def run(name):
        try:
            d.image(Path(f"{name}.ms"), Path(f"/x/{name}"), GEOM, PAR)
            errs[name] = "ok"
        except RuntimeError as e:
            errs[name] = str(e)
    th = [threading.Thread(target=run, args=(n,)) for n in ("p1", "p2", "ok_3", "p4")]
    [x.start() for x in th]
    [x.join() for x in th]
    d.close()
    assert errs["ok_3"] == "ok"
    assert all(errs[k].endswith(f"echo /x/{k}") for k in ("p1", "p2", "p4"))
    assert sorted(r["id"] for r in _requests(log)) == [1, 2, 3, 4]


def test_pipelined_never_guesses_an_idless_reply(tmp_path):
    d, _ = _daemon(tmp_path, "pipe_noid", GSI_SNAP_PIPELINE="1")
    with pytest.raises(RuntimeError, match="daemon died: reply without a pending id"):
        d.image(Path("a.ms"), Path("/x/ok_a"), GEOM, PAR)
    d.close()


def test_pipelined_needs_a_pipelined_daemon(tmp_path):
    d, _ = _daemon(tmp_path, "serial", GSI_SNAP_PIPELINE="1")
    with pytest.raises(RuntimeError, match="not pipelined"):
        d.image(Path("a.ms"), Path("/x/ok_a"), GEOM, PAR)
    d.close()


def test_pipelined_timeout_kills_and_fails_outstanding(tmp_path):
    d, log = _daemon(tmp_path, "pipe_hang", GSI_SNAP_PIPELINE="1", GSI_SNAP_REPLY_TIMEOUT_S="1")
    errs = {}

    def run(name):
        try:
            d.image(Path(f"{name}.ms"), Path(f"/x/{name}"), GEOM, PAR)
            errs[name] = "ok"
        except RuntimeError as e:
            errs[name] = str(e)
    a = threading.Thread(target=run, args=("slow",))
    a.start()
    time.sleep(0.3)
    b = threading.Thread(target=run, args=("ok_after",))
    b.start()
    a.join()
    b.join()
    # the second request was queued behind the hung one in the same daemon: failed, never given another reply
    assert "no reply within 1 s" in errs["slow"] and errs["ok_after"] != "ok"
    assert d.image(Path("c.ms"), Path("/x/ok_c"), GEOM, PAR)       # a new daemon
    d.close()
    assert len({r["pid"] for r in _requests(log)}) == 2
