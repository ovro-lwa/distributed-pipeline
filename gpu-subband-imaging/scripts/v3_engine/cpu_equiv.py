#!/usr/bin/env python
"""gsi v3 steps 4/6, CPU parts on real snapshots: prep -> slot -> record / kept MS vs v2.

  prep     (pipeline python) for every v2 averaged MS <inp>/<stamp>.ms (v3_shm_peel_equiv.py prep:
           the v2 prep=numpy path) and its raw tar: stream the tar into tmpfs, prep it straight
           into a shm slot (fastprep.prep_slot_task, no MS), then
             - kept MS from the slot (fastprep.kept_from_slot_task = write_ms_like) and
               mscompare.table_diff against the v2 MS (structure, keywords, every value);
             - the slot's record through the real engine (workers/gsi_v3_engine.py, fake imaging
               backend: the records, R and the store are the real code) into a GSIV file, the
               LST-hour centre as the record centre (record only, slot unpeeled).
  records  (imager venv) every record of the GSIV file against hourstore.records_from_ms of the
           v2 MS (= msio.load_samples, the v2 hour job's input) and of the kept-from-slot MS:
           bitwise per array (I, V, uvw, natw, ok, freqs, chan_width, a1, a2, ant_pos) and meta,
           R bitwise or max |dR|.

    cpu_equiv.py prep --inp $W/ms/23 --tars <dir with the tars> --out $W/eq/23 --bp B --xy X \
        --badants csv --phase-centre "05h30m00s 37d12m57.057s" --engine-python $VPY \
        --engine-pythonpath $IMSRC
    cpu_equiv.py records --out $W/eq/23 --phase-centre "05h30m00s 37d12m57.057s"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def cmd_prep(a) -> int:
    from gpu_subband_imaging.stages import stage_ms
    from gpu_subband_imaging.v3 import engine_client, fastprep, mscompare, slots
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    badants = Path(a.badants).read_text().strip() if a.badants else ""
    env = dict(os.environ, GSI_V3_ENGINE_BACKEND="fake", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), a.engine_pythonpath]))
    env.pop("PYTHONHOME", None)
    eng = engine_client.EngineClient(a.engine_python, str(REPO / "workers" / "gsi_v3_engine.py"), env,
                                     start_timeout=300)
    gsiv = out / "records.gsiv"
    for f in (gsiv, Path(f"{gsiv}.idx.jsonl")):
        if f.exists():
            f.unlink()
    eng.request("store_open", path=str(gsiv), backend="ram", header=dict(mode="iv", band=a.band),
                phase_centre=a.phase_centre)
    ring = slots.SlotRing(1)
    tpl = None
    results = []
    try:
        for v2 in sorted(Path(a.inp).glob("*.ms")):
            stamp = v2.name[:-3]
            tar = Path(a.tars) / f"{stamp}.ms.tar"
            r = dict(stamp=stamp)
            d = work / stamp
            shutil.rmtree(d, ignore_errors=True)
            t0 = time.perf_counter()
            raw = stage_ms.stage(str(tar), d, untar_to=Path(a.untar_to) / stamp)
            r["stage_s"] = time.perf_counter() - t0
            if tpl is None:
                tpl = work / "_template_avg.ms"
                fastprep.make_template(str(raw), str(tpl), a.chanbin)
            slot = ring.acquire()
            try:
                t0 = time.perf_counter()
                res = fastprep.prep_slot_task(str(raw), a.bp, a.xy, badants, a.chanbin, a.fix_uvw, 1024, 1e-12,
                                              str(slot), 0.01)
                r["prep_slot_s"] = time.perf_counter() - t0
                r["slot_written"] = res["slot_written"]
                r["prep_timings"] = res["meta"]["timings"]
                kept = out / f"{stamp}_kept.ms"
                shutil.rmtree(kept, ignore_errors=True)
                t0 = time.perf_counter()
                fastprep.kept_from_slot_task(str(tpl), str(raw), str(slot), str(kept))
                r["kept_write_s"] = time.perf_counter() - t0
                diffs = mscompare.table_diff(str(v2), str(kept))
                r["kept_vs_v2_identical"] = not diffs
                r["kept_vs_v2_diffs"] = diffs[:20]
                t0 = time.perf_counter()
                rep = eng.request("image_slot", timeout=600, slot=str(slot), prefix=str(d / "img"), pixels=16,
                                  scale=1.0, niter=0, phase_centre=None, record_phase_centre=a.phase_centre,
                                  image=False, record=True, stem=stamp, path=str(v2), require_state=slots.PREPPED)
                r["engine_record_s"] = time.perf_counter() - t0
                r["record"] = rep.get("record")
                r["rotation_s"] = rep.get("timing", {}).get("rotation")
            finally:
                ring.release(slot)
                shutil.rmtree(d, ignore_errors=True)
                shutil.rmtree(Path(a.untar_to) / stamp, ignore_errors=True)
            print(json.dumps({k: v for k, v in r.items() if k != "prep_timings"}, default=str), flush=True)
            results.append(r)
    finally:
        eng.close()
        ring.close()
        if tpl is not None:
            shutil.rmtree(tpl, ignore_errors=True)
    (out / "prep.json").write_text(json.dumps(results, indent=1, default=str))
    return 0 if all(r.get("kept_vs_v2_identical") for r in results) else 1


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == \
        np.ascontiguousarray(b).tobytes()


def cmd_records(a) -> int:
    from gpu_subband_imaging.v3 import hourstore
    from ttcalx_imager.rotate import parse_phase_centre
    out = Path(a.out)
    ra, dec = parse_phase_centre(a.phase_centre)
    hm, ha, recs = hourstore.read_records(out / "records.gsiv")
    res = []
    keys = ("I", "V", "uvw", "natw", "ok_bits", "freqs", "chan_width", "a1", "a2", "ant_pos")
    for rec in recs:
        stem = rec.stem
        r = dict(stem=stem, n_kept=rec.n)
        for name, ms in (("v2_ms", Path(rec.meta["path"])), ("kept_from_slot", out / f"{stem}_kept.ms")):
            ref = hourstore.records_from_ms(ms, ra, dec, stem=stem, path=rec.meta["path"])
            same = {k: _same(rec.arrays[k], ref.arrays[k]) for k in keys}
            dR = float(np.abs(rec.arrays["R"] - ref.arrays["R"]).max())
            m1 = {k: v for k, v in rec.meta.items() if k not in ("written_ns",)}
            m2 = {k: v for k, v in ref.meta.items() if k not in ("written_ns",)}
            r[name] = dict(arrays_bitwise=all(same.values()), not_equal=[k for k, v in same.items() if not v],
                           R_bitwise=_same(rec.arrays["R"], ref.arrays["R"]), R_max_abs_diff=dR,
                           meta_equal=m1 == m2, meta_diff=[k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k)])
        print(json.dumps(r), flush=True)
        res.append(r)
    (out / "records.json").write_text(json.dumps(res, indent=1))
    ok = all(x[n]["arrays_bitwise"] and x[n]["meta_equal"] and x[n]["R_max_abs_diff"] < 1e-9
             for x in res for n in ("v2_ms", "kept_from_slot"))
    print(f"RECORDS {len(res)} ok={ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--inp", required=True, help="dir of v2 averaged MSes <stamp>.ms")
    p.add_argument("--tars", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--untar-to", required=True)
    p.add_argument("--bp", required=True)
    p.add_argument("--xy", required=True)
    p.add_argument("--badants", default="")
    p.add_argument("--band", type=int, required=True)
    p.add_argument("--chanbin", type=int, default=4)
    p.add_argument("--fix-uvw", default="centroid")
    p.add_argument("--phase-centre", required=True)
    p.add_argument("--engine-python", required=True)
    p.add_argument("--engine-pythonpath", required=True)
    p = sub.add_parser("records")
    p.add_argument("--out", required=True)
    p.add_argument("--phase-centre", required=True)
    a = ap.parse_args(argv)
    return {"prep": cmd_prep, "records": cmd_records}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
