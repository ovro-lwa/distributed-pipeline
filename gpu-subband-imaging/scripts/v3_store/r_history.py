"""usage: CASARCFILES=... r_history.py <alternative casarc>

R history dependence: snapshot k computed first in a fresh process vs after its predecessors."""
import json, subprocess, sys, os
from pathlib import Path
CODE = r'''
import sys, json, numpy as np
from ttcalx_imager.msio import read_ms
from ttcalx_imager.rotate import snapshot_rotation, parse_phase_centre
ra, dec = parse_phase_centre("04h30m00s 37d12m57.057s")
out = {}
for p in sys.argv[1:]:
    ms = read_ms(p, want_v=False)
    R = snapshot_rotation(ms.uvw, ms.a1, ms.a2, ms.ant_pos, ms.time, ra, dec)
    out[p] = R.tolist()
print("RJSON" + json.dumps(out))
'''
def run(paths, env):
    o = subprocess.run([sys.executable, "-c", CODE] + paths, capture_output=True, text=True, env=env).stdout
    return json.loads([l for l in o.splitlines() if l.startswith("RJSON")][0][5:])
import numpy as np
ms = sorted(str(p) for p in (Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / "23MHz").glob("*.ms"))
for tag, rc in (("prod casarc", os.environ["CASARCFILES"]), ("d_interval=0", sys.argv[1])):
    env = dict(os.environ, CASARCFILES=rc)
    seq = run(ms, env)                                   # 0..11 in one process
    fresh = {p: run([p], env)[p] for p in ms}            # each first in its own process
    rev = run(ms[::-1], env)                             # reverse order
    d1 = [float(np.max(np.abs(np.array(seq[p]) - np.array(fresh[p])))) for p in ms]
    d2 = [float(np.max(np.abs(np.array(seq[p]) - np.array(rev[p])))) for p in ms]
    print(tag, "max|R_seq - R_fresh| per snapshot:", ["%.1e" % x for x in d1])
    print(tag, "max|R_seq - R_reverse| per snapshot:", ["%.1e" % x for x in d2], flush=True)
    if tag == "prod casarc":
        base = seq
    else:
        print("d_interval=0 vs prod (seq):", "%.1e" % max(float(np.max(np.abs(np.array(seq[p]) - np.array(base[p])))) for p in ms))
