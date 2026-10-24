"""Analyse the pipeline integration test: worker outputs vs last night's production,
per-phase timings (test runs and production logs), and print JSON."""
import glob, json, os, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/lustre/nkosogor/peel_exp/repo/experiments/peeling")
from compare_img import robust_rms, target_pixels, load  # noqa: E402

T = Path("/lustre/nkosogor/peel_exp/gsi_test")
PRODUCTION = Path("/lustre/pipeline/snapshots_gpu_pipeline")
FUNPACK = "/opt/devel/pipeline/envs/development/bin/funpack"
BANDS = (32, 41, 69)
VARIANTS = ("prod", "prod_branch", "cpumatch", "cpumatch_cpu")
TMP = Path(tempfile.mkdtemp(prefix="nk_gsi_an_"))


def unpack(fz):
    # One directory per source folder: the file names are the same in every output tree.
    sub = TMP / str(abs(hash(str(Path(fz).parent))))
    sub.mkdir(exist_ok=True)
    out = sub / (Path(fz).name[:-3])
    if not out.exists():
        subprocess.run([FUNPACK, "-O", str(out), str(fz)], check=True)
    return out


def image(fz):
    with fits.open(unpack(fz)) as h:
        return np.squeeze(h[0].data).astype(float), h[0].header


def horizon(shape, frac=0.47):
    yy, xx = np.indices(shape)
    return np.hypot(xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2) < frac * shape[0]


def metrics(fz_i, fz_v):
    I, _ = image(fz_i)
    V, _ = image(fz_v)
    prefix = str(unpack(fz_i))[:-len("-I-image.fits")]
    unpack(fz_v)
    _, wcs, t = load(prefix, "I")
    m = horizon(I.shape)
    half = int(2.0 / abs(wcs.wcs.cdelt[0]))
    out = {"rms": robust_rms(I[m]), "V_rms": robust_rms(V[m]), "targets": {}}
    for name, (x, y) in target_pixels(wcs, t, I.shape).items():
        x, y = int(round(x)), int(round(y))
        box = I[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
        out["targets"][name] = float(np.abs(box).max())
    return out, I


def fits_for(root, band, stamp):
    d = root / f"{band}MHz" / "2026-10-23" / "08" / "fits"
    return d / f"img_{stamp}-I-image.fits.fz", d / f"img_{stamp}-V-image.fits.fz"


TIMING = re.compile(r"timing (\d+)MHz batch\d+ stamp=(\S+) phase=(\S+)(?: file=\S+)? elapsed=([\d.]+) s")
BATCH = re.compile(r"batch (\d+)MHz/\d+: total elapsed ([\d.]+) s for (\d+) files")
ZT = re.compile(r"ZEST_TIMING read=([\d.]+) peel=([\d.]+) write=([\d.]+)")


def timings(logs, stamps):
    ph, zt, batch = {}, [], []
    for log in logs:
        for line in open(log, errors="replace"):
            m = TIMING.search(line)
            if m and m.group(2) in stamps:
                ph.setdefault(m.group(3), []).append(float(m.group(4)))
            m = ZT.search(line)
            if m:
                zt.append(tuple(map(float, m.groups())))
            m = BATCH.search(line)
            if m:
                batch.append((float(m.group(2)), int(m.group(3))))
    return ph, zt, batch


res = {"compare": {}, "timing": {}}
for band in BANDS:
    stamps = [Path(l).name[:-len(".ms.tar")] for l in (T / f"manifest_{band}.txt").read_text().split()]
    for stamp in stamps:
        row = {}
        imgs = {}
        p_i, p_v = fits_for(PRODUCTION, band, stamp)
        if p_i.exists():
            row["production"], imgs["production"] = metrics(p_i, p_v)
        for v in VARIANTS:
            a_i, a_v = fits_for(T / "out" / v, band, stamp)
            if a_i.exists():
                row[v], imgs[v] = metrics(a_i, a_v)
        for a, b in (("prod", "production"), ("prod_branch", "prod"), ("cpumatch_cpu", "cpumatch"), ("cpumatch", "prod")):
            if a in imgs and b in imgs:
                d = imgs[a] - imgs[b]
                row[f"{a}-{b}"] = dict(identical=bool(np.array_equal(imgs[a], imgs[b])),
                                       maxdiff_over_rms=float(np.abs(d).max() / row[b]["rms"]),
                                       rmsdiff_over_rms=float(robust_rms(d[horizon(d.shape)]) / row[b]["rms"]))
        res["compare"][f"{band}MHz_{stamp}"] = row
    for v in VARIANTS:
        logs = glob.glob(str(T / "out" / v / f"{band}MHz" / "2026-10-23" / "logs" / "*.log"))
        if logs:
            ph, zt, batch = timings(logs, set(stamps))
            res["timing"][f"{v}/{band}"] = dict(phases={k: [float(np.median(x)), len(x)] for k, x in ph.items()},
                                                zest_timing=[float(np.median([z[i] for z in zt])) for i in range(3)] if zt else None,
                                                batch=batch)
    logs = glob.glob(str(PRODUCTION / f"{band}MHz" / "2026-10-23" / "logs" / "*.log"))
    ph, _, _ = timings(logs, set(stamps))
    res["timing"][f"production/{band}"] = dict(phases={k: [float(np.median(x)), len(x)] for k, x in ph.items()})
print(json.dumps(res))
