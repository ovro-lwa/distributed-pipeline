"""How well do raw model visibilities (no gains) match the data?

    python model_data_corr.py DATA_MS MODEL_MS [MODEL_MS ...]

Per model: |<d, m>| / (|d||m|) and the best single complex scale fit
residual, over unflagged cross-correlations, XX and YY separately and per
channel median.  The model that is the right sky (not its mirror image)
correlates strongly with the data when that source dominates.
"""
import sys
import numpy as np
from casacore.tables import table


def load(ms):
    with table(ms, ack=False) as t:
        return t.getcol("DATA"), t.getcol("FLAG"), t.getcol("ANTENNA1") != t.getcol("ANTENNA2")


d, fl, cross = load(sys.argv[1])
good = (~fl) & cross[:, None, None]
for m_ms in sys.argv[2:]:
    m, _, _ = load(m_ms)
    out = []
    for p, name in ((0, "XX"), (3, "YY")):
        g = good[:, :, p]
        dd, mm = np.where(g, d[:, :, p], 0), np.where(g, m[:, :, p], 0)
        c = abs(np.vdot(mm.ravel(), dd.ravel())) / (np.linalg.norm(mm) * np.linalg.norm(dd))
        cc = abs(np.vdot(np.conj(mm).ravel(), dd.ravel())) / (np.linalg.norm(mm) * np.linalg.norm(dd))
        chan = [abs(np.vdot(mm[:, k], dd[:, k])) / (np.linalg.norm(mm[:, k]) * np.linalg.norm(dd[:, k]) + 1e-30)
                for k in range(mm.shape[1])]
        out.append(f"{name}: corr={c:.3f} corr(conj model)={cc:.3f} chan_median={np.median(chan):.3f}")
    print(m_ms, " | ".join(out))
