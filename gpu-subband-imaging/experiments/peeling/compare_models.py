"""Compare model visibilities written to MS copies (DATA), CPU vs GPU.

    python compare_models.py REF_MS OTHER_MS [OTHER_MS ...]
"""
import sys
import numpy as np
from casacore.tables import table


def load(ms):
    with table(ms, ack=False) as t:
        return t.getcol("DATA"), t.getcol("ANTENNA1") != t.getcol("ANTENNA2"), t.getcol("UVW")


ref, cross, uvw = load(sys.argv[1])
blen = np.linalg.norm(uvw, axis=1)
for other in sys.argv[2:]:
    d, _, _ = load(other)
    r, o = ref[cross], d[cross]
    rel = np.linalg.norm(o - r) / np.linalg.norm(r)
    ratio = np.abs(o).sum((0, 2)) / np.abs(r).sum((0, 2))  # per channel amplitude ratio
    m = np.abs(r[:, :, 0]) > 1e-6 * np.abs(r[:, :, 0]).max()
    ph = np.angle(o[:, :, 0][m] / r[:, :, 0][m])
    long = blen[cross] > np.percentile(blen[cross], 90)
    mlong = m & long[:, None]
    phl = np.angle(o[:, :, 0][mlong] / r[:, :, 0][mlong])
    print(f"{other}: rel_diff={rel:.3e} amp_ratio[min/med/max]={ratio.min():.4f}/{np.median(ratio):.4f}/{ratio.max():.4f} "
          f"XX phase diff rms={np.sqrt(np.mean(ph**2)):.3e} rad (long bl {np.sqrt(np.mean(phl**2)):.3e}) "
          f"per-pol rel={[round(float(np.linalg.norm(o[...,i]-r[...,i])/max(np.linalg.norm(r[...,i]),1e-30)),5) for i in range(4)]}")
