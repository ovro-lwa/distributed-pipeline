#!/usr/bin/env python
"""gsi v3 step 4 on the GPU: snapshot images from the shm slot vs v2 (MS copy + chgcentre + MS read).

The same peeled MSes (<inp>/<stamp>.ms, e.g. from v3_shm_peel_equiv.py peel --mode ms) are imaged

  A  v2: copy, chgcentre to the LST-hour centre (stages.image.chgcentre), workers/ttcalx_snapshot_daemon.py
     on the copy (stages.gpu_image.SnapshotDaemon) -- exactly Worker._image_archive;
  B  v2 again (the run-to-run floor: cuFINUFFT atomics);
  S  v3: v3.peel_shm.snap_from_ms -> shm slot (PEELED) -> workers/gsi_v3_engine.py image_slot at the
     LST-hour centre (in-memory chgcentre rotation, no MS), its record into a RAM store;

with the same imager source for all three.  ``compare`` reports per snapshot, for I and V: noise
(MAD sigma in the 0.40 horizon mask), diff rms / max in sigma for A-S and A-B, check-source peaks
S/A, and the FITS header keys that differ (e2e_compare_v2.py functions).

    gpu_snap_equiv.py image --inp $W/peeled/23 --out $W/img/23 --band 23 --lst-hour 5 \
        --imager-python $VPY --imager-pythonpath $IMSRC [-n 24]
    gpu_snap_equiv.py compare --out $W/img/23 --json res.json
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

REPO = Path(__file__).resolve().parents[2]
E2E = "/lustre/nkosogor/peel_exp/e2e"


def _geom(band):
    from gpu_subband_imaging.config import load_subbands
    return load_subbands(REPO / "examples" / "config" / "subbands.yaml").geom(band)


def cmd_image(a) -> int:
    from gpu_subband_imaging import lst
    from gpu_subband_imaging.config import ImagingParams
    from gpu_subband_imaging.stages import gpu_image, image
    from gpu_subband_imaging.v3 import engine_client, peel_shm, slots
    out = Path(a.out)
    for k in ("A", "B", "S"):
        (out / k).mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    pc = lst.phase_centre(a.lst_hour)
    geom = _geom(a.band)
    ip = ImagingParams(mode="clean", imager="ttcalx")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONPATH=a.imager_pythonpath, NUMPY_MADVISE_HUGEPAGE="0", OPENBLAS_NUM_THREADS="1")
    mss = sorted(Path(a.inp).glob("*.ms"))[: a.n or None]
    timing = {"A": [], "B": [], "S": []}
    # v2 daemon (A and B): pipelined like the worker with 2 archive threads would be is not needed here;
    # one snapshot at a time (the image values do not depend on it)
    for k in ("A", "B"):
        d = gpu_image.SnapshotDaemon(a.imager_python, str(REPO / "workers" / "ttcalx_snapshot_daemon.py"), env)
        try:
            for ms in mss:
                stamp = ms.name[:-3]
                t0 = time.perf_counter()
                src = work / f"{stamp}_lst.ms"
                shutil.rmtree(src, ignore_errors=True)
                shutil.copytree(ms, src)
                image.chgcentre(src, pc)
                t1 = time.perf_counter()
                steps = {}
                d.image(src, out / k / f"img_{stamp}", geom, ip, timings=steps)
                shutil.rmtree(src, ignore_errors=True)
                timing[k].append(dict(stamp=stamp, copy_chgcentre=t1 - t0, image=time.perf_counter() - t1,
                                      image_gpu=steps.get("image_gpu")))
                print(k, stamp, json.dumps(timing[k][-1]), flush=True)
        finally:
            d.close()
    # v3 engine (S)
    eenv = dict(env, PYTHONPATH=os.pathsep.join([str(REPO / "src"), a.imager_pythonpath]))
    eng = engine_client.EngineClient(a.imager_python, str(REPO / "workers" / "gsi_v3_engine.py"), eenv,
                                     start_timeout=600)
    ring = slots.SlotRing(2)
    try:
        eng.request("store_open", path=None, backend="ram")
        for ms in mss:
            stamp = ms.name[:-3]
            slot = ring.acquire()
            try:
                t0 = time.perf_counter()
                slots.write_slot(slot, peel_shm.snap_from_ms(ms))
                slots.set_state(slot, slots.PEELED)
                t1 = time.perf_counter()
                r = eng.request("image_slot", timeout=900, slot=str(slot), prefix=str(out / "S" / f"img_{stamp}"),
                                pixels=geom.pixels, scale=geom.scale, niter=ip.clean_niter, phase_centre=pc,
                                record_phase_centre=pc, image=True, record=True, stem=stamp, path=str(ms))
                timing["S"].append(dict(stamp=stamp, ms_to_slot=t1 - t0, image=time.perf_counter() - t1,
                                        image_gpu=r["image"]["timing"].get("TOTAL"),
                                        engine=r["image"]["timing"], rotation=r["timing"].get("rotation"),
                                        record=r.get("record")))
                print("S", stamp, json.dumps({k: v for k, v in timing["S"][-1].items() if k != "engine"},
                                             default=float), flush=True)
            finally:
                ring.release(slot)
        info = eng.request("info")
    finally:
        eng.close()
        ring.close()
    shutil.rmtree(work, ignore_errors=True)
    (out / "timing.json").write_text(json.dumps(dict(timing=timing, engine_info=info), indent=1, default=float))
    return 0


def cmd_compare(a) -> int:
    sys.path.insert(0, E2E)
    import e2e_compare_v2 as ec
    from astropy.io import fits
    out = Path(a.out)
    rows = []
    for fa in sorted((out / "A").glob("img_*-I-image.fits")):
        stamp = fa.name[len("img_"):-len("-I-image.fits")]
        row = dict(stamp=stamp)
        for pol in ("I", "V"):
            p = {k: out / k / f"img_{stamp}-{pol}-image.fits" for k in "ABS"}
            if not all(x.exists() for x in p.values()):
                row[pol] = dict(missing=[k for k, x in p.items() if not x.exists()])
                continue
            im = {k: ec.read_image(v, str(out)) for k, v in p.items()}
            data = {k: v[0] for k, v in im.items()}
            mask = ec.horizon_mask(data["A"].shape)
            beam_px = 2 if data["A"].shape[0] <= 2048 else 3
            r = dict(AS=ec.compare_pair(data["A"], data["S"], mask), AB=ec.compare_pair(data["A"], data["B"], mask))
            if pol == "I":
                sa = ec.src_table(data["A"], im["A"][1], ec.mad_std(data["A"][mask]), beam_px)
                ss = ec.src_table(data["S"], im["S"][1], ec.mad_std(data["S"][mask]), beam_px)
                sb = ec.src_table(data["B"], im["B"][1], ec.mad_std(data["B"][mask]), beam_px)
                r["src"] = {n: dict(A=sa[n], S=ss[n], B=sb[n], S_over_A=ss[n] / sa[n] if sa[n] else None,
                                    snr_A=sa[n] / sa["sigma"]) for n in ec.CHECK if np.isfinite(sa[n])}
            ha, hs = fits.getheader(p["A"]), fits.getheader(p["S"])
            keys = [k for k in set(ha) | set(hs) if k not in ("HISTORY", "COMMENT", "")]
            r["header_diff"] = {k: [str(ha.get(k)), str(hs.get(k))] for k in sorted(keys) if ha.get(k) != hs.get(k)}
            row[pol] = r
        rows.append(row)
        print(json.dumps(dict(stamp=stamp, I=row.get("I", {}).get("AS"), I_floor=row.get("I", {}).get("AB"),
                              hdr=row.get("I", {}).get("header_diff")), default=float), flush=True)

    def agg(pol, pair, key):
        v = [r[pol][pair][key] for r in rows if pol in r and pair in r[pol]]
        return dict(n=len(v), max=float(np.max(v)) if v else None, median=float(np.median(v)) if v else None)
    summ = {pol: {pair: {k: agg(pol, pair, k) for k in ("diff_rms_sigma", "diff_max_sigma", "noise_ratio")}
                  for pair in ("AS", "AB")} for pol in ("I", "V")}
    ratios = [v["S_over_A"] for r in rows if "I" in r for v in r["I"].get("src", {}).values()
              if v["S_over_A"] is not None and v["snr_A"] > 10]
    summ["src_S_over_A_snr>10"] = dict(n=len(ratios), min=min(ratios) if ratios else None,
                                       max=max(ratios) if ratios else None)
    summ["header_diff_keys"] = sorted({k for r in rows for pol in ("I", "V") if pol in r
                                       for k in r[pol].get("header_diff", {})})
    print(json.dumps(summ, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1, default=float))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image")
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--band", type=int, required=True)
    p.add_argument("--lst-hour", type=int, required=True)
    p.add_argument("--imager-python", required=True)
    p.add_argument("--imager-pythonpath", required=True)
    p.add_argument("-n", type=int, default=0)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--json", default="")
    a = ap.parse_args(argv)
    return {"image": cmd_image, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
