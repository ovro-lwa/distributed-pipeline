"""Image-plane comparison of peeling runs (WSClean dirty Stokes I/V).

    python compare_img.py OUT_PNG NAME=PREFIX [NAME=PREFIX ...]
        [--ref NAME] [--pol I] [--json OUT]

PREFIX is a WSClean -name prefix (PREFIX-I-image.fits).  Reports per image
the robust RMS inside the horizon and the peak / RMS within a box around
each peel target (Cyg A, Cas A, Vir A and the RFI emitter's apparent
position), and, against --ref, the RMS of the difference image.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OVRO = EarthLocation(lat=37.2397808 * u.deg, lon=-118.2816819 * u.deg, height=1183 * u.m)
TARGETS = {
    "CygA": SkyCoord("19h59m28.36s", "+40d44m02.1s"),
    "CasA": SkyCoord("23h23m24s", "+58d48m54s"),
    "VirA": SkyCoord("12h30m49.4s", "+12d23m28s"),
}
RFI_SITE = EarthLocation(lat=37.53159 * u.deg, lon=-118.62673 * u.deg, height=2072 * u.m)
BOX_DEG = 4.0


def rfi_altaz():
    """Apparent az/el of the RFI emitter seen from the array centre."""
    p = (RFI_SITE.get_itrs().cartesian - OVRO.get_itrs().cartesian).xyz.to(u.m).value
    lat, lon = OVRO.lat.rad, OVRO.lon.rad
    e = -np.sin(lon) * p[0] + np.cos(lon) * p[1]
    n = (-np.sin(lat) * np.cos(lon) * p[0] - np.sin(lat) * np.sin(lon) * p[1]
         + np.cos(lat) * p[2])
    up = np.cos(lat) * np.cos(lon) * p[0] + np.cos(lat) * np.sin(lon) * p[1] + np.sin(lat) * p[2]
    return np.degrees(np.arctan2(e, n)) % 360, np.degrees(np.arctan2(up, np.hypot(e, n)))


def load(prefix: str, pol: str):
    with fits.open(f"{prefix}-{pol}-image.fits") as h:
        hdr = h[0].header
        img = np.squeeze(h[0].data).astype(float)
    return img, WCS(hdr).celestial, Time(hdr["DATE-OBS"], scale="utc")


def robust_rms(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def target_pixels(wcs, t, shape):
    frame = AltAz(obstime=t, location=OVRO)
    out = {}
    for name, c in TARGETS.items():
        if c.transform_to(frame).alt.deg > 0:
            out[name] = wcs.world_to_pixel(c)
    az, el = rfi_altaz()
    c = SkyCoord(az=az * u.deg, alt=max(el, 1.0) * u.deg, frame=frame).icrs
    out["RFI"] = wcs.world_to_pixel(c)
    return {k: (float(x), float(y)) for k, (x, y) in out.items()
            if np.isfinite(x) and 0 <= x < shape[1] and 0 <= y < shape[0]}


def horizon_mask(shape, frac=0.47):
    yy, xx = np.indices(shape)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    return np.hypot(xx - cx, yy - cy) < frac * shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_png")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--pol", default="I")
    ap.add_argument("--json")
    a = ap.parse_args()

    runs = dict(s.split("=", 1) for s in a.runs)
    imgs = {k: load(p, a.pol) for k, p in runs.items()}
    first = next(iter(imgs.values()))
    wcs, t = first[1], first[2]
    shape = first[0].shape
    mask = horizon_mask(shape)
    pix = target_pixels(wcs, t, shape)
    scale = abs(wcs.wcs.cdelt[0])
    half = int(BOX_DEG / scale / 2)

    def box(img, xy):
        x, y = int(round(xy[0])), int(round(xy[1]))
        return img[max(0, y - half):y + half, max(0, x - half):x + half]

    res = {"time": t.isot, "targets_px": pix, "images": {}}
    ref = imgs[a.ref][0] if a.ref else None
    for name, (img, _, _) in imgs.items():
        r = {"rms": robust_rms(img[mask]), "targets": {}}
        for tn, xy in pix.items():
            b = box(img, xy)
            r["targets"][tn] = {"peak_abs": float(np.nanmax(np.abs(b))),
                                "rms": float(np.sqrt(np.nanmean(b ** 2)))}
        if ref is not None and name != a.ref:
            d = img - ref
            r["diff_vs_ref_rms"] = robust_rms(d[mask])
            r["diff_vs_ref_std"] = float(np.nanstd(d[mask]))
            r["diff_vs_ref_targets"] = {tn: float(np.sqrt(np.nanmean(box(d, xy) ** 2)))
                                        for tn, xy in pix.items()}
        res["images"][name] = r
        tg = " ".join(f"{k}:pk={v['peak_abs']:.1f}/rms={v['rms']:.2f}"
                      for k, v in r["targets"].items())
        dv = (f" diff_vs_{a.ref}: rms={r['diff_vs_ref_rms']:.3f} std={r['diff_vs_ref_std']:.3f}"
              if "diff_vs_ref_rms" in r else "")
        print(f"IMG {name:24s} rms={r['rms']:.3f} {tg}{dv}")

    n = len(imgs) + (len(imgs) - 1 if ref is not None else 0)
    ncol = len(imgs)
    nrow = 2 if ref is not None else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow), squeeze=False)
    v = 6 * res["images"][next(iter(imgs))]["rms"]
    for i, (name, (img, _, _)) in enumerate(imgs.items()):
        ax = axes[0, i]
        ax.imshow(np.where(mask, img, np.nan), origin="lower", cmap="inferno", vmin=-v, vmax=2 * v)
        ax.set_title(f"{name}\nrms={res['images'][name]['rms']:.2f}", fontsize=9)
        for tn, (x, y) in pix.items():
            ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False,
                                       ec="cyan", lw=0.6))
            ax.text(x, y + half, tn, color="cyan", fontsize=7, ha="center")
        ax.set_xticks([]); ax.set_yticks([])
        if ref is not None:
            ax = axes[1, i]
            if name == a.ref:
                ax.axis("off")
                continue
            d = np.where(mask, img - ref, np.nan)
            dv = 3 * res["images"][a.ref]["rms"]
            ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv)
            ax.set_title(f"{name} - {a.ref}\nrms={res['images'][name]['diff_vs_ref_rms']:.3f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Stokes {a.pol}  {t.isot}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out_png, dpi=90)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
