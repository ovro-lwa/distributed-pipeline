#!/usr/bin/env python
"""Compare a bandpass (B) table with reference tables, per spectral window.

usage: check_bandpass.py NEW.B REF.B [REF2.B ...] --out DIR

For each band: flagged fraction, and over samples unflagged in both tables the
median and robust scatter of |g_new / g_ref| and of the phase difference after
removing each antenna's median phase difference (reference-antenna convention),
plus the number of antennas whose median amplitude ratio is off by > 20 %.
Writes bandpass_check.json and bandpass_check.png (median amplitude spectra,
ratio and phase-difference spectra, flag fractions).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(path):
    from casacore.tables import table
    t = table(path, ack=False)
    g, f = t.getcol("CPARAM"), t.getcol("FLAG")
    spw, ant = t.getcol("SPECTRAL_WINDOW_ID"), t.getcol("ANTENNA1")
    freqs = table(t.getkeyword("SPECTRAL_WINDOW"), ack=False).getcol("CHAN_FREQ") / 1e6
    t.close()
    nspw, nant = spw.max() + 1, ant.max() + 1
    G = np.full((nspw, nant) + g.shape[1:], np.nan + 0j, dtype=complex)
    F = np.ones((nspw, nant) + g.shape[1:], dtype=bool)
    G[spw, ant], F[spw, ant] = g, f
    G[F] = np.nan
    return G, F, freqs


def mad(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def compare(new, ref):
    Gn, Fn, fq = new
    Gr, Fr, _ = ref
    out = []
    for s in range(min(Gn.shape[0], Gr.shape[0])):
        both = ~Fn[s] & ~Fr[s]
        r = np.where(both, Gn[s] / Gr[s], np.nan)
        amp = np.abs(r)
        ph = np.angle(r)
        ph_ant = np.angle(np.nanmean(np.exp(1j * ph), axis=(1, 2), keepdims=True))
        dph = np.degrees(np.angle(np.exp(1j * (ph - ph_ant))))
        ant_ratio = np.nanmedian(amp, axis=(1, 2))
        out.append(dict(
            spw=s, freq_mhz=float(np.nanmean(fq[s])),
            flag_new=float(Fn[s].mean()), flag_ref=float(Fr[s].mean()),
            n_both=int(both.sum()),
            amp_ratio_median=float(np.nanmedian(amp)) if both.any() else None,
            amp_ratio_scatter=mad(np.log(amp[both])) if both.any() else None,
            phase_diff_scatter_deg=mad(dph[both]) if both.any() else None,
            n_ant_amp_off_20pct=int(np.sum(np.abs(ant_ratio - 1) > 0.2)),
            n_ant_compared=int(np.sum(np.isfinite(ant_ratio))),
        ))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    new = load(a.new)
    refs = {r: load(r) for r in a.refs}
    res = {"new": a.new, "comparisons": {r: compare(new, v) for r, v in refs.items()}}
    with open(os.path.join(a.out, "bandpass_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Gn, Fn, fq = new
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for s in range(Gn.shape[0]):
        ax[0, 0].plot(fq[s], np.nanmedian(np.abs(Gn[s, :, :, 0]), axis=0), "-", color="C0", lw=0.8,
                      label="new" if s == 0 else None)
    for k, (r, (Gr, Fr, _)) in enumerate(refs.items()):
        for s in range(Gr.shape[0]):
            ax[0, 0].plot(fq[s], np.nanmedian(np.abs(Gr[s, :, :, 0]), axis=0), "-", color=f"C{k + 1}",
                          lw=0.6, alpha=0.7, label=os.path.basename(r) if s == 0 else None)
        for s in range(min(Gn.shape[0], Gr.shape[0])):
            both = ~Fn[s] & ~Fr[s]
            rr = np.where(both, Gn[s] / Gr[s], np.nan)[:, :, 0]
            ax[0, 1].plot(fq[s], np.nanmedian(np.abs(rr), axis=0), "-", color=f"C{k + 1}", lw=0.8,
                          label=os.path.basename(r) if s == 0 else None)
            ph = np.angle(rr)
            ph = ph - np.angle(np.nanmean(np.exp(1j * ph), axis=1, keepdims=True))
            ax[1, 0].plot(fq[s], np.degrees(np.nanstd(np.angle(np.exp(1j * ph)), axis=0)), "-",
                          color=f"C{k + 1}", lw=0.8, label=os.path.basename(r) if s == 0 else None)
    for k, (r, cmp) in enumerate(res["comparisons"].items()):
        ax[1, 1].plot([c["freq_mhz"] for c in cmp], [c["flag_ref"] for c in cmp], "o-", color=f"C{k + 1}",
                      label=f"flagged, {os.path.basename(r)}")
    cmp0 = next(iter(res["comparisons"].values()))
    ax[1, 1].plot([c["freq_mhz"] for c in cmp0], [c["flag_new"] for c in cmp0], "s-", color="C0", label="flagged, new")
    ax[0, 0].set_ylabel("median |g| (XX)")
    ax[0, 1].set_ylabel("median |g_new / g_ref| (XX)")
    ax[0, 1].set_ylim(0.5, 1.5)
    ax[1, 0].set_ylabel("phase(g_new / g_ref) scatter over antennas (deg)")
    ax[1, 1].set_ylabel("flagged fraction")
    for x in ax.flat:
        x.set_xlabel("MHz")
        x.legend(fontsize=7)
    fig.suptitle(os.path.basename(a.new))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "bandpass_check.png"), dpi=90)

    print(f"{'MHz':>6} {'flag_new':>8} " + " ".join(f"{'ref' + str(i) + ' flag amp scat ph_deg nbad':>34}" for i in range(len(refs))))
    for s, c in enumerate(cmp0):
        cols = []
        for cmp in res["comparisons"].values():
            x = cmp[s]
            cols.append(f"{x['flag_ref']:.2f} {x['amp_ratio_median'] or float('nan'):.3f} "
                        f"{x['amp_ratio_scatter'] or float('nan'):.3f} {x['phase_diff_scatter_deg'] or float('nan'):6.1f} "
                        f"{x['n_ant_amp_off_20pct']:3d}/{x['n_ant_compared']}")
        print(f"{c['freq_mhz']:6.1f} {c['flag_new']:8.2f} " + "   ".join(cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
