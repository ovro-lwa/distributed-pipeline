"""More report figures from the study outputs.

    python report_figs2.py fullsky STUDY OUT SNAP V1,V2,V3       large full-sky panels, labelled sources
    python report_figs2.py onsource STUDY OUT SNAP V1,...         zoom grid on every target
    python report_figs2.py gains GAINS_DIR MS OUT TAG_NOREG TAG_REG SOURCE_INDEX MAX_OFFSET_DEG
                                                                  gain phase vs antenna distance
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load, robust_rms, target_pixels  # noqa: E402
from report_figs import LABEL  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = 299792458.0


def horizon(shape, frac=0.495):
    yy, xx = np.indices(shape)
    return np.hypot(xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2) < frac * shape[0]


def fullsky(study, out, snap, variants):
    imgs = {v: load(f"{study}/{snap}/img/{v}", "I") for v in variants}
    im0, wcs, t = imgs[variants[0]]
    hz = horizon(im0.shape)
    rms = {v: robust_rms(im[horizon(im.shape, 0.47)]) for v, (im, _, _) in imgs.items()}
    vmax = 12 * min(rms.values())
    pix = target_pixels(wcs, t, im0.shape)
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 5.6))
    for ax, v in zip(np.atleast_1d(axes), variants):
        ax.imshow(np.where(hz, imgs[v][0], np.nan), origin="lower", cmap="inferno",
                  vmin=-vmax / 4, vmax=vmax)
        for name, (x, y) in pix.items():
            ax.annotate(name, (x, y), xytext=(x + 60, y + 60), color="cyan", fontsize=8,
                        arrowprops=dict(arrowstyle="-", color="cyan", lw=0.6))
        ax.set_title(f"{LABEL.get(v, v)}\nrobust rms {rms[v]:.2f} Jy/beam", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{snap.replace('_', ' ')} UT  —  full sky, dirty Stokes I, same colour scale",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/fullsky_{snap}.png", dpi=130)


def onsource(study, out, snap, variants):
    imgs = {v: load(f"{study}/{snap}/img/{v}", "I") for v in variants}
    im0, wcs, t = imgs[variants[0]]
    hz = horizon(im0.shape)
    pix = target_pixels(wcs, t, im0.shape)
    targets = [k for k in ("CasA", "CygA", "VirA", "RFI") if k in pix]
    half = int(6.0 / abs(wcs.wcs.cdelt[0]) / 2)
    rmin = min(robust_rms(im[horizon(im.shape, 0.47)]) for im, _, _ in imgs.values())
    fig, axes = plt.subplots(len(targets), len(variants),
                             figsize=(2.3 * len(variants), 2.45 * len(targets)), squeeze=False)
    for i, tn in enumerate(targets):
        x, y = (int(round(c)) for c in pix[tn])
        sl = (slice(max(0, y - half), y + half), slice(max(0, x - half), x + half))
        for j, v in enumerate(variants):
            z = np.where(hz, imgs[v][0], np.nan)[sl]
            ax = axes[i, j]
            ax.imshow(z, origin="lower", cmap="inferno", vmin=-5 * rmin, vmax=25 * rmin)
            ax.set_xticks([]); ax.set_yticks([])
            pk = np.nanmax(np.abs(z)) if np.isfinite(z).any() else np.nan
            ax.text(0.03, 0.04, f"max {pk:.0f}", color="w", fontsize=7, transform=ax.transAxes)
            if i == 0:
                ax.set_title(LABEL.get(v, v), fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{tn} (12°)", fontsize=8)
    fig.suptitle(f"{snap.replace('_', ' ')} UT — on-source residuals (Jy/beam, same scale)", fontsize=10, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/onsource_{snap}.png", dpi=140)


def gains(gdir, ms, out, tag0, tag1, idx, deg):
    from casacore.tables import table
    pos = table(ms + "/ANTENNA", ack=False).getcol("POSITION")
    freq = table(ms + "/SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ")[0]
    c = np.median(pos, 0); up = c / np.linalg.norm(c)
    d = pos - c
    rho = np.linalg.norm(d - np.outer(d @ up, up), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ch = len(freq) // 2
    for ax, tag in zip(axes, (tag0, tag1)):
        path = f"{gdir}/{tag}.{idx}.gains.bin"
        meta = json.load(open(path + ".json"))
        na, nf, _ = meta["shape"]
        g = np.fromfile(path, dtype="<c16").reshape((na, nf, 4), order="F")[:, ch, 0]
        ok = (np.abs(g) > 0) & (np.abs(g - 1) > 1e-9)  # flagged antennas keep g = 1
        ax.scatter(rho[ok], np.degrees(np.angle(g[ok])), s=4, alpha=0.7)
        r = np.linspace(0, rho.max(), 200)
        k = 2 * np.pi * freq[ch] / C * r
        for D, ls in ((deg, "-"), (deg + 1, "--")):
            lim = np.degrees(np.minimum(k * np.sin(np.radians(D)), np.pi))
            ax.plot(r, lim, "r" + ls, lw=1); ax.plot(r, -lim, "r" + ls, lw=1)
        ax.set_xlabel("antenna distance from array centre (m)")
        ax.set_title(f"{meta['name']} — {'no prior' if tag == tag0 else f'Tukey prior {deg}° (+1° taper)'}"
                     f"\n{freq[ch] / 1e6:.1f} MHz, XX gain phase", fontsize=9)
    axes[0].set_ylabel("gain phase (deg)")
    axes[0].set_ylim(-185, 185)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/gains_{tag1}_{idx}.png", dpi=140)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fullsky":
        fullsky(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(","))
    elif mode == "onsource":
        onsource(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(","))
    elif mode == "gains":
        gains(*sys.argv[2:7], int(sys.argv[7]), float(sys.argv[8]))
