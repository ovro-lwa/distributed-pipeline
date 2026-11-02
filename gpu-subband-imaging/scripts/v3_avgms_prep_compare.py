"""v3 in-memory prep of night-time averaged MSes (chanbin 1) vs the production backlog prep.

v2 side (production backlog code, gsi-v2-test2, run in a subprocess with its PYTHONPATH):
fastcal.flag_calibrate on a copy, rename (chanbin 1), uvw.fix_uvw(centroid), visibility_stats.
v3 side (exp/v3-avgms): fastprep.prep_arrays(chanbin=1) of another copy (read only), then
write_ms_like with a chanbin-1 template.  Compared: every SnapArrays column bit for bit
against the v2 MS, the whole written MS against it (table_diff), and the quality stats.

    python avg_prep_cmp.py --out results.json --bp B --xy X --badants CSV MS...
Inputs are only read (copied to --scratch first).
"""
import argparse, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

V2_SRC = "/opt/devel/pipeline/gsi-v2-test2/distributed-pipeline/gpu-subband-imaging/src"
V3_SRC = str(Path(__file__).resolve().parents[1] / "src")

V2_SIDE = r'''
import json, sys, time
from pathlib import Path
from gpu_subband_imaging.stages import fastcal, quality, uvw
raw, out, bp, xy, bad = sys.argv[1:6]
t0 = time.perf_counter()
fastcal.flag_calibrate(Path(raw), bp, xy, bad)
Path(raw).rename(out)
t1 = time.perf_counter()
fix = uvw.fix_uvw(Path(out), epoch="centroid").__dict__
t2 = time.perf_counter()
st = quality.visibility_stats(Path(out), "DATA", 1024, 1e-12).__dict__
print(json.dumps({"fix": fix, "stats": st, "t_cal": t1 - t0, "t_uvw": t2 - t1}, default=str))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ms", nargs="+")
    ap.add_argument("--bp", required=True)
    ap.add_argument("--xy", required=True)
    ap.add_argument("--badants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/dev/shm/nk_avgms")
    a = ap.parse_args()
    sys.path.insert(0, V3_SRC)
    from gpu_subband_imaging.stages import quality
    from gpu_subband_imaging.v3 import fastprep
    from gpu_subband_imaging.v3.mscompare import table_diff

    env2 = dict(os.environ, PYTHONPATH=V2_SRC)
    recs = []
    for ms in a.ms:
        stamp = Path(ms).name[:-len("_averaged.ms")]
        w = Path(a.scratch) / stamp
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True)
        rec = {"ms": ms}
        try:
            c2, c3 = w / "in2" / Path(ms).name, w / "in3" / Path(ms).name
            for c in (c2, c3):
                c.parent.mkdir()
                subprocess.run(["cp", "-r", ms, str(c)], check=True)
            avg2 = w / f"{stamp}_avg.ms"
            p = subprocess.run([sys.executable, "-c", V2_SIDE, str(c2), str(avg2), a.bp, a.xy, a.badants],
                               env=env2, capture_output=True, text=True, cwd="/tmp")
            if p.returncode:
                raise RuntimeError("v2 side: " + p.stderr[-2000:])
            rec["v2"] = json.loads(p.stdout.strip().splitlines()[-1])
            t0 = time.perf_counter()
            snap = fastprep.prep_arrays(str(c3), a.bp, a.xy, a.badants, 1, "centroid")
            rec["t_v3_prep"] = time.perf_counter() - t0
            st3 = quality.visibility_stats_arrays(snap.data, snap.flag, 1024, 1e-12).__dict__
            rec["stats_equal"] = st3 == rec["v2"]["stats"]
            rec["fix_v3"] = snap.meta["fix_uvw"]
            rec["fix_err_v3"] = snap.meta["fix_uvw_error"]
            rec["cmp_snap_vs_v2"] = fastprep.compare_snap_ms(snap, str(avg2))
            tpl = w / "tpl.ms"
            fastprep.make_template(str(c3), str(tpl), 1)
            out3 = w / "v3_avg.ms"
            fastprep.write_ms_like(str(tpl), str(c3), snap, str(out3))
            rec["table_diff_v2_vs_v3ms"] = table_diff(str(avg2), str(out3))
            rec["flag_frac"] = float(snap.flag.mean())
        except Exception:
            rec["error"] = traceback.format_exc()
        finally:
            shutil.rmtree(w, ignore_errors=True)
        recs.append(rec)
        ok = rec.get("cmp_snap_vs_v2", {}).get("identical")
        print(f"{Path(ms).name}: identical={ok} stats_equal={rec.get('stats_equal')} "
              f"err={'yes' if 'error' in rec else 'no'}", flush=True)
        Path(a.out).write_text(json.dumps(recs, indent=1, default=str))


if __name__ == "__main__":
    main()
