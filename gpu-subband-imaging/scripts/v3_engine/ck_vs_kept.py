"""Records of a gsi v3 store / checkpoint (GSIV) vs hourstore.records_from_ms of the kept MSes of the same run.
usage (imager venv): ck_vs_kept.py <file.gsiv> <kept dir> "<LST>h30m00s 37d12m57.057s" """
import glob, json, sys
import numpy as np
from gpu_subband_imaging.v3 import hourstore
from ttcalx_imager.rotate import parse_phase_centre
ck, kept_dir, pc = sys.argv[1], sys.argv[2], sys.argv[3]
ra, dec = parse_phase_centre(pc)
recs = hourstore.read_records(ck)[2]
out = []
for r in recs:
    ms = f"{kept_dir}/{r.stem}.ms"
    ref = hourstore.records_from_ms(ms, ra, dec, stem=r.stem, path=r.meta["path"])
    ks = [k for k in ("I", "V", "uvw", "natw", "ok_bits", "freqs", "chan_width", "a1", "a2", "ant_pos")
          if r.arrays[k].tobytes() != ref.arrays[k].tobytes()]
    m1 = {k: v for k, v in r.meta.items() if k != "written_ns"}; m2 = {k: v for k, v in ref.meta.items() if k != "written_ns"}
    out.append(dict(stem=r.stem, path=r.meta["path"], n=r.n, arrays_differ=ks, meta_equal=m1 == m2,
                    dR=float(np.abs(r.arrays["R"] - ref.arrays["R"]).max())))
    print(json.dumps(out[-1]))
print("ALL_BITWISE", all(not o["arrays_differ"] and o["meta_equal"] for o in out), "max dR", max(o["dR"] for o in out))
