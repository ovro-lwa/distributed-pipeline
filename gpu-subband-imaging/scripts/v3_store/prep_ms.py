"""Make averaged, calibrated, fix_uvw'd snapshot MSes with the v2 prep functions (no peel).

usage: prep_ms.py <band> <n> <badants_file> <bp> <xy> <out_dir> <work_dir>
"""
import json, logging, shutil, sys, time
from pathlib import Path
from gpu_subband_imaging.stages import stage_ms, fastcal, uvw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
band, n, badf, bp, xy, out, work = sys.argv[1:8]
n = int(n); out = Path(out).resolve(); work = Path(work).resolve(); bp = str(Path(bp).resolve()); xy = str(Path(xy).resolve())
bad = Path(badf).read_text().strip()
tars = sorted(Path(f"/lustre/nkosogor/peel_exp/v3_inputs/{band}MHz").glob("*.ms.tar"))[:n]
out.mkdir(parents=True, exist_ok=True)
for tar in tars:
    stamp = tar.name[:-len(".ms.tar")]
    dst = out / f"{stamp}.ms"
    if dst.exists():
        continue
    d = work / stamp
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.perf_counter()
    ms = stage_ms.stage(str(tar), d)
    t1 = time.perf_counter()
    fastcal.flag_calibrate(ms, bp, xy, bad)
    t2 = time.perf_counter()
    avg = fastcal.average(ms, d / f"{stamp}_avg.ms", 4)
    t3 = time.perf_counter()
    # worker._fix_uvw: fix_uvw(epoch=cfg.fix_uvw='centroid'), keep the MS if it fails
    try:
        st = uvw.fix_uvw(avg, epoch="centroid")
        fx = dict(rot_deg=st.rot_deg, max_dw_m=st.max_dw_m, rephased=st.rephased)
    except Exception as e:
        fx = dict(error=repr(e))
    t4 = time.perf_counter()
    shutil.move(str(avg), str(dst))
    shutil.rmtree(d, ignore_errors=True)
    print(json.dumps(dict(stamp=stamp, stage=t1 - t0, flag_cal=t2 - t1, average=t3 - t2, fix_uvw=t4 - t3, **fx)), flush=True)
