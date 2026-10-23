"""Summarise a run_variant.sh study: image and visibility metrics per snapshot.

    python study_summary.py STUDY_DIR [--snaps 23MHz_0800 ...] [--vis] [--png]

Per snapshot and variant (images in <snap>/img/<variant>-I-image.fits):
  rms        robust rms of Stokes I inside the horizon (r < 0.47 of the image)
  <target>   peak |I| and rms in a 4° box at Cyg A / Cas A / Vir A and at the
             RFI emitter (r < 0.495 so the near-horizon RFI is included)
  V_rms      robust rms of Stokes V
With --vis, from <snap>/<variant>.ms against <snap>/prep.ms:
  removed    fraction of cross-correlation power removed by peeling
  corr_<ref> |<removed, removed_ref>| / (|removed| |removed_ref|) with
             ref = cpu_sky and cpu_sky_rfi (the main pipeline)
Writes STUDY_DIR/summary.json and, with --png, <snap>/compare.png with the
full images and zooms on the targets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_img import load, robust_rms, target_pixels  # noqa: E402

ORDER = ["prep", "cpu_sky", "cpu_sky_rfi", "gpu_prod", "gpu_prodfix", "gpu_prodfix_rfi", "gpu_exp", "gpu_geom",
         "gpu_cpumatch", "gpu_cm_exact", "gpu_cm_exact_joint",
         "gpu_geom_rfi", "gpu_geom_rficpu", "gpu_rfi_oldbr",
         "gpu_reg0.5", "gpu_reg1", "gpu_reg2", "gpu_reg5"]
REFS = ("cpu_sky", "cpu_sky_rfi")


def radius_mask(shape, frac):
    yy, xx = np.indices(shape)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    return np.hypot(xx - cx, yy - cy) < frac * shape[0]


def vis_removed(prep_ms, ms):
    from casacore.tables import table
    with table(ms, ack=False) as t:
        d = t.getcol("DATA"); f = t.getcol("FLAG")
        cross = t.getcol("ANTENNA1") != t.getcol("ANTENNA2")
    with table(prep_ms, ack=False) as t:
        p = t.getcol("DATA"); f |= t.getcol("FLAG")
    good = (~f) & cross[:, None, None]
    good[..., 1:3] = False  # parallel hands only
    return np.where(good, p - d, 0), np.where(good, p, 0)


def snapshot(snapdir, do_vis, do_png):
    variants = [v for v in ORDER if os.path.exists(f"{snapdir}/img/{v}-I-image.fits")]
    variants += sorted(os.path.basename(p)[:-len("-I-image.fits")]
                       for p in glob.glob(f"{snapdir}/img/*-I-image.fits")
                       if os.path.basename(p)[:-len("-I-image.fits")] not in ORDER)
    if not variants:
        return None
    out = {}
    imgs = {}
    for v in variants:
        img, wcs, t = load(f"{snapdir}/img/{v}", "I")
        imgs[v] = img
        inner = radius_mask(img.shape, 0.47)
        outer = radius_mask(img.shape, 0.495)
        pix = target_pixels(wcs, t, img.shape)
        half = int(4.0 / abs(wcs.wcs.cdelt[0]) / 2)
        r = {"rms": robust_rms(img[inner]), "targets": {}}
        for tn, (x, y) in pix.items():
            x, y = int(round(x)), int(round(y))
            sl = (slice(max(0, y - half), y + half), slice(max(0, x - half), x + half))
            b = np.where(outer[sl], img[sl], np.nan)
            if np.isfinite(b).any():
                r["targets"][tn] = {"peak": float(np.nanmax(np.abs(b))),
                                    "rms": float(np.sqrt(np.nanmean(b ** 2)))}
        try:
            vimg, _, _ = load(f"{snapdir}/img/{v}", "V")
            r["V_rms"] = robust_rms(vimg[inner])
        except FileNotFoundError:
            pass
        out[v] = r
    if do_vis and os.path.exists(f"{snapdir}/prep.ms"):
        removed = {}
        for v in variants:
            if v == "prep" or not os.path.exists(f"{snapdir}/{v}.ms"):
                continue
            rem, p0 = vis_removed(f"{snapdir}/prep.ms", f"{snapdir}/{v}.ms")
            removed[v] = rem
            out[v]["removed"] = float(np.sum(np.abs(rem) ** 2) / np.sum(np.abs(p0) ** 2))
        for v, rem in removed.items():
            for ref in REFS:
                if ref in removed and ref != v:
                    rr = removed[ref]
                    out[v][f"corr_{ref}"] = float(abs(np.vdot(rr, rem)) /
                                                  (np.linalg.norm(rr) * np.linalg.norm(rem) + 1e-30))
                    out[v][f"reldiff_{ref}"] = float(np.linalg.norm(rem - rr) /
                                                     (np.linalg.norm(rr) + 1e-30))
    if do_png:
        make_png(snapdir, variants, imgs, out)
    return out


def make_png(snapdir, variants, imgs, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, wcs, t = load(f"{snapdir}/img/{variants[0]}", "I")
    shape = imgs[variants[0]].shape
    pix = target_pixels(wcs, t, shape)
    zooms = [k for k in ("RFI", "CasA", "CygA", "VirA") if k in pix]
    half = int(8.0 / abs(wcs.wcs.cdelt[0]) / 2)
    outer = radius_mask(shape, 0.495)
    ref_rms = res["prep"]["rms"] if "prep" in res else res[variants[0]]["rms"]
    vmax = 8 * min(r["rms"] for r in res.values())
    ncol = 1 + len(zooms)
    fig, axes = plt.subplots(len(variants), ncol, figsize=(3.2 * ncol, 3.2 * len(variants)),
                             squeeze=False)
    for i, v in enumerate(variants):
        img = np.where(outer, imgs[v], np.nan)
        ax = axes[i, 0]
        ax.imshow(img, origin="lower", cmap="inferno", vmin=-vmax / 4, vmax=vmax)
        ax.set_title(f"{v}  rms={res[v]['rms']:.2f}", fontsize=8)
        for j, tn in enumerate(zooms):
            x, y = (int(round(c)) for c in pix[tn])
            sl = (slice(max(0, y - half), y + half), slice(max(0, x - half), x + half))
            ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False,
                                       ec="cyan", lw=0.5))
            z = axes[i, 1 + j]
            zv = 3 * vmax
            z.imshow(img[sl], origin="lower", cmap="inferno", vmin=-zv / 4, vmax=zv)
            pk = res[v]["targets"].get(tn, {}).get("peak", np.nan)
            z.set_title(f"{tn} 16° peak={pk:.0f}", fontsize=8)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{os.path.basename(snapdir)}  Stokes I (dirty)  {t.isot}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{snapdir}/compare.png", dpi=80)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--snaps", nargs="*")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    snaps = a.snaps or sorted(os.path.basename(p) for p in glob.glob(f"{a.study}/*MHz_*"))
    path = f"{a.study}/summary.json"
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for s in snaps:
        r = snapshot(f"{a.study}/{s}", a.vis, a.png)
        if r is None:
            continue
        summary[s] = r
        for v, m in r.items():
            tg = " ".join(f"{k}={x['peak']:.0f}" for k, x in m["targets"].items())
            extra = " ".join(f"{k}={m[k]:.3f}" for k in m if k.startswith(("removed", "corr_")))
            print(f"{s:11s} {v:16s} rms={m['rms']:7.2f} V={m.get('V_rms', np.nan):6.2f} {tg} {extra}")
        json.dump(summary, open(path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
