#!/usr/bin/env python
"""gsi v3 step 3: peel through the MS (v2) vs through a /dev/shm slot (v3.peel_io=shm).

Subcommands (pipeline python, PYTHONPATH=<code>/src):

  prep     averaged, calibrated, UVW-fixed MSes from raw tars with the v2 prep functions
           (stage_ms.stage, fastcal.flag_calibrate + fastcal.average, uvw.fix_uvw as
           Worker._fix_uvw), one <stamp>.ms per tar in --out.
  peel     copy every <stamp>.ms of --inp to --out and peel the copies in one daemon, in
           stamp order: --mode ms (ZestDaemon.zest, the v2 path) or --mode shm
           (v3.peel_shm.peel_ms_via_slot).  The ZEST_* settings come from the peel: section
           of --cfg (PeelParams.daemon_env), with --device / --test-io overrides.  Writes
           --out/results.json (per stamp: daemon times, ZEST_TIMING, hand-off times, logs).
  compare  bitwise DATA / FLAG (and every other main column) of two peel outputs.

Julia / TTCalX come from the environment as in workers/run_worker.sh: JULIA_BIN (dir),
TTCALX_DIR, JULIA_DEPOT_PATH, CUDA_VISIBLE_DEVICES, JULIA_NUM_THREADS (device=cpu).

Recipe (as run on lwacalim10 on CPU, 2026-11-01; for the GPU drop --device cpu and set
CUDA_VISIBLE_DEVICES=<gpu>):

  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
  export PYTHONPATH=$CODE/src CASARCFILES=/opt/devel/pipeline/gsi-v2/casarc \
         PYTHONDONTWRITEBYTECODE=1 JULIA_CUDA_USE_COMPAT=false \
         JULIA_BIN=/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin \
         TTCALX_DIR=/lustre/nkosogor/peel_exp/ttcalx_sponge3 \
         JULIA_DEPOT_PATH=$W/depot:/opt/devel/nkosogor/nkosogor/julia_depot
  E=/lustre/nkosogor/peel_exp/e2e
  python v3_shm_peel_equiv.py prep --out $W/ms/23 --work $W/work --bp <B> --xy <Xf> \
      --badants <csv> /lustre/nkosogor/peel_exp/v3_inputs/23MHz/<first 6>.ms.tar
  for m in ms ms shm; do                  # MS twice: the run-to-run floor
    python v3_shm_peel_equiv.py peel --mode $m --inp $W/ms/23 --out $W/runs/$m.$RANDOM/23 \
        --cfg $E/cfg_new_23/pipeline.yaml --daemon-jl $CODE/workers/zest_daemon.jl
  done
  python v3_shm_peel_equiv.py compare $W/runs/<ms run>/23 $W/runs/<shm run>/23
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def cmd_prep(a) -> int:
    from gpu_subband_imaging.stages import fastcal, stage_ms, uvw
    badants = Path(a.badants).read_text().strip() if a.badants else ""
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    for tar in a.tars:
        stamp = Path(tar).name[:-len(".ms.tar")]
        dst = out / f"{stamp}.ms"
        if dst.exists():
            continue
        d = work / stamp
        t0 = time.perf_counter()
        ms = stage_ms.stage(tar, d)
        t1 = time.perf_counter()
        fastcal.flag_calibrate(ms, a.bp, a.xy, badants)
        t2 = time.perf_counter()
        avg = fastcal.average(ms, d / f"{stamp}_avg.ms", a.chanbin)
        t3 = time.perf_counter()
        note = ""
        if a.fix_uvw != "off":
            try:
                st = uvw.fix_uvw(avg, epoch=a.fix_uvw)
                note = f"rot_deg={st.rot_deg:.4f} rephased={st.rephased}"
            except Exception as e:               # as Worker._fix_uvw: keep the stored UVW
                note = f"fix_uvw failed, stored UVW kept: {e}"
        shutil.rmtree(ms)
        shutil.move(str(avg), str(dst))
        shutil.rmtree(d, ignore_errors=True)
        print(f"PREP {stamp} stage={t1 - t0:.1f}s flag_cal={t2 - t1:.1f}s average={t3 - t2:.1f}s "
              f"fix_uvw={time.perf_counter() - t3:.1f}s {note}", flush=True)
    return 0


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _daemon(a):
    import yaml
    from gpu_subband_imaging.config import PeelParams
    from gpu_subband_imaging.stages.peel import ZestDaemon
    d = yaml.safe_load(Path(a.cfg).read_text())
    peel = dict(d.get("peel", {}))
    if a.device:
        peel["device"] = a.device
    for kv in a.set:                              # e.g. --set fast_io=false
        k, v = kv.split("=", 1)
        peel[k] = yaml.safe_load(v)
    zenv = {"ZEST_COLUMN": "DATA",
            **PeelParams(**peel).daemon_env(d["cal"].get("rfi_sources"))}
    if a.test_io:
        zenv["ZEST_TEST_IO"] = a.test_io
    julia = os.path.join(os.environ["JULIA_BIN"], "julia")
    t0 = time.perf_counter()
    dm = ZestDaemon(julia=julia, project=os.environ["TTCALX_DIR"], daemon_jl=a.daemon_jl,
                    sources=d["cal"]["sources"], env=dict(os.environ), zest_env=zenv)
    return dm, zenv, time.perf_counter() - t0


def cmd_peel(a) -> int:
    from gpu_subband_imaging.v3 import peel_shm, slots
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name for p in inp.glob("*.ms"))[:a.n or None]
    for s in stems:                              # copies first: the peel loop does only peeling
        shutil.rmtree(out / s, ignore_errors=True)
        shutil.copytree(str(inp / s), str(out / s))
    col = _Collect()
    logging.getLogger("gsi.peel").addHandler(col)
    logging.getLogger("gsi.peel").setLevel(logging.INFO)
    logging.getLogger("gsi.peel").propagate = False
    dm, zenv, t_start = _daemon(a)
    ring = slots.SlotRing(1) if a.mode == "shm" else None
    res = dict(mode=a.mode, cfg=a.cfg, daemon_jl=a.daemon_jl, zest_env=zenv,
               julia_threads=os.environ.get("JULIA_NUM_THREADS"), daemon_start_s=t_start,
               snapshots=[])
    try:
        for s in stems:
            ms = out / s
            col.lines.clear()
            steps: dict = {}
            t0 = time.perf_counter()
            try:
                if a.mode == "ms":
                    el = dm.zest(ms)
                else:
                    el = peel_shm.peel_ms_via_slot(dm, ms, ring, timeout=a.timeout, timings=steps)
                err = None
            except Exception as e:
                el, err = None, str(e)
            wall = time.perf_counter() - t0
            zt = next((ln for ln in col.lines if ln.startswith("ZEST_TIMING")), "")
            zt_d = {k: float(v) for k, v in (kv.split("=") for kv in zt.split()[1:])} if zt else {}
            rec = dict(stem=s, daemon_s=el, wall_s=wall, error=err, zest_timing=zt_d,
                       handoff=steps, log=list(col.lines))
            res["snapshots"].append(rec)
            print(f"PEEL {a.mode} {s} wall={wall:.2f}s daemon={el} {zt} "
                  f"{' '.join(f'{k}={v:.3f}' for k, v in steps.items())} err={err}", flush=True)
    finally:
        dm.close()
        if ring is not None:
            ring.close()
    (out / "results.json").write_text(json.dumps(res, indent=1))
    return 0 if all(r["error"] is None for r in res["snapshots"]) else 1


_COLS = ("DATA", "FLAG", "UVW", "WEIGHT", "SIGMA", "FLAG_ROW", "ANTENNA1", "ANTENNA2", "TIME")


def _compare_ms(x: Path, y: Path) -> dict:
    from casacore.tables import table
    r = {}
    with table(str(x), ack=False) as tx, table(str(y), ack=False) as ty:
        for c in _COLS:
            u, v = tx.getcol(c), ty.getcol(c)
            same = u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
            e = dict(bitwise=bool(same))
            if not same and u.shape == v.shape:
                if c == "DATA":
                    d = np.abs(u.astype(np.complex128) - v.astype(np.complex128))
                    e.update(max_abs=float(np.nanmax(d)), n_diff=int((u != v).sum()),
                             rel_rms=float(np.sqrt(np.nanmean(d ** 2) /
                                                   np.nanmean(np.abs(u.astype(np.complex128)) ** 2))))
                else:
                    e.update(n_diff=int((u != v).sum()))
            r[c] = e
    return r


def cmd_compare(a) -> int:
    x, y = Path(a.a), Path(a.b)
    stems = sorted(p.name for p in x.glob("*.ms") if (y / p.name).exists())
    out = {}
    ok = bool(stems)
    for s in stems:
        r = _compare_ms(x / s, y / s)
        out[s] = r
        bad = {c: e for c, e in r.items() if not e["bitwise"]}
        ok &= not any(c in ("DATA", "FLAG") for c in bad)
        print(f"COMPARE {s} DATA={r['DATA']['bitwise']} FLAG={r['FLAG']['bitwise']} "
              f"other_diff={sorted(bad) if bad else '-'} "
              f"{r['DATA'] if not r['DATA']['bitwise'] else ''}", flush=True)
    print(f"COMPARE_SUMMARY {x} vs {y}: n={len(stems)} bitwise_DATA_FLAG={ok}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--bp", required=True)
    p.add_argument("--xy", required=True)
    p.add_argument("--badants", default="")
    p.add_argument("--chanbin", type=int, default=4)
    p.add_argument("--fix-uvw", default="centroid")
    p.add_argument("tars", nargs="+")
    p = sub.add_parser("peel")
    p.add_argument("--mode", choices=("ms", "shm"), required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cfg", required=True, help="pipeline.yaml with the peel: and cal: sections")
    p.add_argument("--daemon-jl", required=True)
    p.add_argument("--device", default="", help="override peel.device (gpu|cpu)")
    p.add_argument("--test-io", default="", choices=("", "identity", "perturb"))
    p.add_argument("--set", action="append", default=[], help="peel.<key>=<yaml value> override")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("-n", type=int, default=0, help="first n MSes only")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--json", default="")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    return {"prep": cmd_prep, "peel": cmd_peel, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
