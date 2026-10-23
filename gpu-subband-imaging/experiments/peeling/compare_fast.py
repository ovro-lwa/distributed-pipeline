"""Optimized TTCalX outputs vs the earlier runs of the same configuration (MS only read).

    python compare_fast.py STUDY [STUDY ...]   -> JSON on stdout
fast_sky vs gpu_cm_exact and fast_joint vs gpu_cm_exact_joint: max |ΔDATA| / rms(DATA) over
unflagged cross-correlations, and whether FLAG is identical.
"""
import glob
import json
import os
import sys

import numpy as np
from casacore.tables import table

PAIRS = (("fast_sky", "gpu_cm_exact"), ("fast_joint", "gpu_cm_exact_joint"))
out = {}
for study in sys.argv[1:]:
    for snapdir in sorted(glob.glob(f"{study}/*MHz_*")):
        snap = os.path.basename(snapdir)
        for new, old in PAIRS:
            a_p, b_p = f"{snapdir}/{new}.ms", f"{snapdir}/{old}.ms"
            if not (os.path.exists(a_p) and os.path.exists(b_p)):
                continue
            a, b = table(a_p, ack=False), table(b_p, ack=False)
            x = a.getcol("ANTENNA1") != a.getcol("ANTENNA2")
            da, db = a.getcol("DATA")[x], b.getcol("DATA")[x]
            fa, fb = a.getcol("FLAG")[x], b.getcol("FLAG")[x]
            ok = ~fa
            rms = np.sqrt(np.mean(np.abs(db[ok]) ** 2))
            out[f"{os.path.basename(study)}/{snap}/{new}"] = dict(
                max_rel=float(np.abs(da[ok] - db[ok]).max() / rms),
                rms_rel=float(np.sqrt(np.mean(np.abs(da[ok] - db[ok]) ** 2)) / rms),
                flags_identical=bool(np.array_equal(fa, fb)))
            a.close(); b.close()
print(json.dumps(out))
