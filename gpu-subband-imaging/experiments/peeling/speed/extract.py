"""Extract I=(XX+YY)/2 of cross correlations (row order), flags, uvw, freqs (MS only read)."""
import sys, glob, numpy as np
from casacore.tables import table
P = "/lustre/nkosogor/peel_exp"
SRC = {"prep": P + "/data/{s}/*_prep.ms",
       "cpu_ttcal": P + "/runs/bench_20261023_094323/cpu/{s}.ms",
       "gpu_kernel": P + "/runs/bench_20261023_094323/cm_after/{s}.ms",
       "gpu_exact": P + "/runs/bench2_20261023_203655/gpu_exact_after/{s}.ms",
       "gpu_fast": P + "/runs/fasttest_20261023_204156/slowwrite/{s}.ms"}
for s in sys.argv[1:]:
    for tag, pat in SRC.items():
        ms = glob.glob(pat.format(s=s))[0]
        t = table(ms, ack=False)
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        x = a1 != a2
        d = t.getcol("DATA")[x]; f = t.getcol("FLAG")[x]
        out = dict(I=((d[:, :, 0] + d[:, :, 3]) / 2).astype(np.complex64), flag=f.any(axis=2),
                   a1=a1[x], a2=a2[x])
        if tag == "prep":
            out["uvw"] = t.getcol("UVW")[x]
            out["freq"] = table(ms + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
            out["time"] = t.getcol("TIME")[0]
            out["phase_dir"] = table(ms + "/FIELD", ack=False).getcol("PHASE_DIR")[0, 0]
        np.savez(f"/tmp/nk_peel/out/{tag}_{s}.npz", **out)
        print("EXTRACTED", tag, s, flush=True)
