"""Within-band shape check of a bandpass table against a reference table.

usage: bp_ramp.py <table> <ref table>   -> prints "OK ..." or "REJECT ..."
Per band: per-channel median over antennas / pols of |B| / |B_ref| (both unflagged), normalised by its median;
the 10-90 percentile spread over channels must be < 0.12 (catches amplitude ramps inside a band, which the
per-band median check of run_nightly_lst.check_table does not).
"""
import sys
import numpy as np
from casacore.tables import table


def load(p):
    t = table(p, ack=False)
    g, f, s, a = t.getcol("CPARAM"), t.getcol("FLAG"), t.getcol("SPECTRAL_WINDOW_ID"), t.getcol("ANTENNA1")
    fr = table(t.getkeyword("SPECTRAL_WINDOW"), ack=False).getcol("REF_FREQUENCY") / 1e6
    return g, f, s, a, fr


def main():
    g, f, s, a, fr = load(sys.argv[1])
    G, F, S, A, FR = load(sys.argv[2])
    bad, spread = [], {}
    for k, mhz in enumerate(fr):
        kk = int(np.argmin(np.abs(FR - mhz)))
        if abs(FR[kk] - mhz) > 1:
            continue
        x, xf, xa = g[s == k], f[s == k], a[s == k]
        y, yf, ya = G[S == kk], F[S == kk], A[S == kk]
        common = np.intersect1d(xa, ya)
        x, xf = x[np.isin(xa, common)][np.argsort(xa[np.isin(xa, common)])], xf[np.isin(xa, common)][np.argsort(xa[np.isin(xa, common)])]
        y, yf = y[np.isin(ya, common)][np.argsort(ya[np.isin(ya, common)])], yf[np.isin(ya, common)][np.argsort(ya[np.isin(ya, common)])]
        if x.shape != y.shape:
            continue
        r = np.where(xf | yf | (np.abs(y) == 0), np.nan, np.abs(x) / np.where(np.abs(y) == 0, 1, np.abs(y)))
        per_ch = np.nanmedian(r.reshape(r.shape[0], r.shape[1], -1), axis=(0, 2))
        per_ch = per_ch[np.isfinite(per_ch)]
        if per_ch.size < 4:
            continue
        per_ch = per_ch / np.median(per_ch)
        sp = float(np.percentile(per_ch, 90) - np.percentile(per_ch, 10))
        spread[int(round(mhz))] = round(sp, 3)
        if sp >= 0.12:
            bad.append(int(round(mhz)))
    print(("REJECT within-band ramps at %s MHz" % bad if bad else "OK") + " | spread " + str(spread))


if __name__ == "__main__":
    main()
