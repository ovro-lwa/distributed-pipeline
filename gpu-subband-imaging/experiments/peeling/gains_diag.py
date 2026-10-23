"""Diagnostics of saved direction-dependent gains (ZEST_SAVE_GAINS_DIR).

    python gains_diag.py GAINS_DIR MS_FOR_GEOMETRY [--match PATTERN]

For each <tag>.<pass>.<i>.gains.bin (complex128 (Nant, Nfreq, 4), F order):
  shift_deg   the source offset the gains imply: per channel, the plane-wave
              phase gradient exp(i k (Δe x_j + Δn y_j)) across the array that
              best matches the xx gain phases (grid search, |Δ| ≤ 12°);
              median over channels of |Δ|
  resid_rad   circular std of the xx gain phases after removing that gradient
              and the mean phase (non-geometric phase freedom used)
  frac_out<D> fraction of gains whose phase exceeds the prior's flat region
              for a max offset of D degrees (2π sin(D) ν/c ρ_j)
  amp_mad     median absolute deviation of |g_xx| about its median
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from casacore.tables import table

C = 299792458.0


def geometry(ms):
    pos = table(ms + "/ANTENNA", ack=False).getcol("POSITION")
    freq = table(ms + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
    centre = np.median(pos, axis=0)
    up = centre / np.linalg.norm(centre)
    east = np.cross([0, 0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    d = pos - centre
    return np.stack([d @ east, d @ north], 1), freq


def fit_shift(g, xy, k, grid):
    """Best plane-wave offset (deg) for the phases of g (Nant,) at wavenumber k."""
    ph = np.exp(1j * np.angle(g))
    ok = np.isfinite(ph) & (np.abs(g) > 0)
    th = np.radians(grid)
    # separable: exp(-i k (a x + b y)) = exp(-i k a x) exp(-i k b y)
    ex = np.exp(-1j * k * np.outer(np.sin(th), xy[ok, 0]))  # (G, Na)
    ey = np.exp(-1j * k * np.outer(np.sin(th), xy[ok, 1]))
    score = np.abs((ex * ph[ok][None, :]) @ ey.T)  # (Gx, Gy)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    a, b = th[i], th[j]
    model = np.exp(1j * k * (np.sin(a) * xy[:, 0] + np.sin(b) * xy[:, 1]))
    r = ph * np.conj(model)
    r = r[ok]
    r *= np.conj(np.mean(r)) / (abs(np.mean(r)) + 1e-30)
    R = abs(np.mean(r))
    return np.degrees(np.hypot(a, b)), float(np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gains_dir")
    ap.add_argument("ms")
    ap.add_argument("--match", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    xy, freq = geometry(a.ms)
    rho = np.hypot(xy[:, 0], xy[:, 1])
    grid = np.arange(-12, 12.01, 0.1)
    out = {}
    for path in sorted(glob.glob(f"{a.gains_dir}/*{a.match}*.gains.bin")):
        meta = json.load(open(path + ".json"))
        na, nf, _ = meta["shape"]
        g = np.fromfile(path, dtype="<c16").reshape((na, nf, 4), order="F")
        gx = g[:, :, 0]
        if nf != len(freq):
            continue
        shifts, resid = [], []
        for c in range(nf):
            s, r = fit_shift(gx[:, c], xy, 2 * np.pi * freq[c] / C, grid)
            shifts.append(s); resid.append(r)
        # phase relative to the gain-weighted mean phase per channel (gauge)
        ref = np.angle(np.nansum(gx, axis=0))
        phi = np.abs(np.angle(gx * np.exp(-1j * ref)[None, :]))
        fr = {}
        for D in (0.5, 1, 2, 5):
            lim = 2 * np.pi * np.sin(np.radians(D)) * freq[None, :] / C * rho[:, None]
            sel = lim < np.pi
            fr[f"frac_out{D}"] = float(np.mean(phi[sel] > lim[sel])) if sel.any() else float("nan")
        amp = np.abs(gx)
        r = {"name": meta["name"], "shift_deg": float(np.median(shifts)),
             "shift_deg_p90": float(np.percentile(shifts, 90)),
             "resid_rad": float(np.median(resid)),
             "amp_mad": float(np.median(np.abs(amp - np.median(amp)))), **fr}
        tag = os.path.basename(path)[:-len(".gains.bin")]
        out[tag] = r
        print(f"{tag:48s} {r['name']:8s} shift={r['shift_deg']:5.2f}° (p90 {r['shift_deg_p90']:5.2f}) "
              f"resid={r['resid_rad']:.2f} rad  " +
              " ".join(f"{k}={v:.2f}" for k, v in fr.items()) + f" amp_mad={r['amp_mad']:.3f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
