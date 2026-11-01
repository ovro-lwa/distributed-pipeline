#!/usr/bin/env python
"""gsi v3 steps 1-2 on real snapshots: v2 prep vs in-memory prep, equivalence and CPU time.

For every tar, in one of up to --procs worker processes:

- v2, exactly the worker's prep=numpy path: stage_ms.stage (copy the tar to scratch,
  extract), fastcal.flag_calibrate, fastcal.average (CASA mstransform subprocess),
  uvw.fix_uvw, quality.visibility_stats;
- v3: stage_ms.stage(untar_to=tmpfs) (streamed), fastprep.prep_arrays,
  quality.visibility_stats_arrays, fastprep.write_ms_like (template: mstransform of a
  tar outside the test set);
- compare: the two staged raw MSes file by file (before v2 rewrites its copy), the
  SnapArrays against v2's averaged MS (compare_snap_ms), the written MS against v2's
  (mscompare.table_diff: structure, keywords, storage managers, all subtable values),
  the written MS read back against the SnapArrays, and the quality stats.

Wall and CPU seconds per stage; CPU includes child processes (the CASA subprocess).
The order of the v2 and v3 runs alternates between files (the first run of a tar
reads it cold from Lustre).  One JSON line per tar in --out.

    v3_prep_compare.py --bp B --xy X --badants-file f.csv --out eq.jsonl \\
        --work /tmp/v3cmp --untar-to /dev/shm/v3cmp --procs 4 --template-tar T  TAR...
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import resource
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _usage():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.perf_counter(), s.ru_utime + s.ru_stime, c.ru_utime + c.ru_stime


class Clock:
    def __init__(self):
        self.t = {}

    def run(self, name, fn, *a, **k):
        w0, s0, c0 = _usage()
        out = fn(*a, **k)
        w1, s1, c1 = _usage()
        self.t[name] = {"wall": w1 - w0, "cpu": (s1 - s0) + (c1 - c0), "cpu_children": c1 - c0}
        return out


def _same_tree(a: Path, b: Path) -> bool:
    fa = sorted(str(p.relative_to(a)) for p in a.rglob("*"))
    fb = sorted(str(p.relative_to(b)) for p in b.rglob("*"))
    if fa != fb:
        return False
    return all(filecmp.cmp(a / f, b / f, shallow=False) for f in fa if (a / f).is_file())


def one(tar: str, args, template: str, v2_first: bool) -> dict:
    from gpu_subband_imaging.stages import fastcal, quality, stage_ms, uvw
    from gpu_subband_imaging.v3 import fastprep
    from gpu_subband_imaging.v3.mscompare import table_diff

    stamp = Path(tar).name[:-len(".ms.tar")]
    work = Path(args.work) / stamp
    udir = Path(args.untar_to) / stamp
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(udir, ignore_errors=True)
    rec = {"stamp": stamp, "tar": tar, "v2_first": v2_first, "pid": os.getpid()}
    v2c, v3c = Clock(), Clock()
    try:
        # staging, in alternating order (cold Lustre read goes to the first one)
        def stage_v2():
            return v2c.run("stage", stage_ms.stage, tar, work / "v2")

        def stage_v3():
            return v3c.run("stage", stage_ms.stage, tar, work / "v3", untar_to=udir)

        if v2_first:
            raw2, raw3 = stage_v2(), stage_v3()
        else:
            raw3, raw2 = stage_v3(), stage_v2()
        rec["staged_identical"] = _same_tree(raw2, raw3)

        # v3 (raw MS read only)
        snap = v3c.run("prep", fastprep.prep_arrays, str(raw3), args.bp, args.xy, args.badants,
                       args.chanbin, args.fix_uvw)
        for k, v in snap.meta["timings"].items():
            v3c.t[k] = v
        st3 = v3c.run("quality", quality.visibility_stats_arrays, snap.data, snap.flag,
                      args.sample_rows, args.min_amplitude)
        out3 = work / "v3" / f"{stamp}_avg.ms"
        v3c.run("write", fastprep.write_ms_like, template, str(raw3), snap, str(out3))
        rec["fix_uvw_v3"] = snap.meta["fix_uvw"]
        rec["fix_uvw_error_v3"] = snap.meta["fix_uvw_error"]

        # v2 (the worker's prep=numpy steps)
        v2c.run("cal", fastcal.flag_calibrate, raw2, args.bp, args.xy, args.badants)
        avg2 = v2c.run("average", fastcal.average, raw2, work / "v2" / f"{stamp}_avg.ms",
                       args.chanbin)
        fix2 = None
        if args.fix_uvw != "off":
            try:
                fix2 = v2c.run("uvw", uvw.fix_uvw, avg2, epoch=args.fix_uvw).__dict__
            except Exception as e:          # the worker keeps the stored UVW
                rec["fix_uvw_error_v2"] = str(e)
        rec["fix_uvw_v2"] = fix2
        st2 = v2c.run("quality", quality.visibility_stats, avg2, "DATA", args.sample_rows,
                      args.min_amplitude)

        # comparisons
        rec["cmp_v2"] = fastprep.compare_snap_ms(snap, str(avg2))
        rec["table_diff"] = table_diff(str(avg2), str(out3))
        rec["cmp_readback"] = fastprep.compare_snap_ms(snap, str(out3))
        rec["stats_equal"] = st2 == st3
        rec["stats_v2"] = st2.__dict__
        rec["flag_frac"] = float(snap.flag.mean())
    except Exception:
        rec["error"] = traceback.format_exc()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(udir, ignore_errors=True)
    rec["t_v2"], rec["t_v3"] = v2c.t, v3c.t
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tars", nargs="+")
    ap.add_argument("--bp", required=True)
    ap.add_argument("--xy", required=True)
    ap.add_argument("--badants-file", required=True)
    ap.add_argument("--template-tar", required=True, help="a tar of the same band, not in the set")
    ap.add_argument("--chanbin", type=int, default=4)
    ap.add_argument("--fix-uvw", default="centroid")
    ap.add_argument("--sample-rows", type=int, default=1024)
    ap.add_argument("--min-amplitude", type=float, default=1e-12)
    ap.add_argument("--work", required=True)
    ap.add_argument("--untar-to", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    args.badants = Path(args.badants_file).read_text().strip()

    from gpu_subband_imaging.stages import stage_ms
    from gpu_subband_imaging.v3 import fastprep
    tdir = Path(args.work) / "_template"
    shutil.rmtree(tdir, ignore_errors=True)
    raw = stage_ms.stage(args.template_tar, tdir)
    template = str(fastprep.make_template(str(raw), str(tdir / "template_avg.ms"), args.chanbin))
    shutil.rmtree(raw)

    import multiprocessing
    n = 0
    with open(args.out, "a") as fo, ProcessPoolExecutor(
            max_workers=args.procs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = {pool.submit(one, t, args, template, i % 2 == 0): t for i, t in enumerate(args.tars)}
        for f in as_completed(futs):
            rec = f.result()
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
            c = rec.get("cmp_v2", {})
            print(f"[{n}/{len(futs)}] {rec['stamp']} identical={c.get('identical')} "
                  f"table_diff={len(rec.get('table_diff', ['?']))} "
                  f"readback={rec.get('cmp_readback', {}).get('identical')} "
                  f"staged={rec.get('staged_identical')} error={'error' in rec}", flush=True)
    shutil.rmtree(tdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
