"""gsi v3 step 5 GPU check: hour products from --archive vs from the MSes (imager venv).

  1. deterministic GPU stages, bitwise: LazySnap.expand on CuPy == host load_samples arrays;
     HourGrid imaging weights, sum of weights, w-term K and chunking (dicts vs LazySnaps);
     the SampleSet inputs + rotated I / V / PSF visibilities of a chunk in every
     TTCALX_CHUNK_CACHE mode, first call and cache hit;
  2. the products through the production command (ttcalx-image hour, subprocess): MSes twice
     (the cuFINUFFT floor) and --archive twice; FITS compared bitwise, else diff rms / max in
     units of the image noise against the MS-vs-MS floor (design section 7 criteria).

usage: gpu_equiv.py --band 55 --ms-dir DIR --archive F.gsiv --out OUT --lst-hour 4
                    [--products orca] [--reps 2] [--ttcalx-image BIN] [--cpu]
--cpu: dry run on a node without a GPU (tests/hour_cpu numpy stand-in; products in-process).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def deterministic_checks(hour, gsiv, cp, ms, ar, ra, dec, band, log):
    from ttcalx_imager import bands
    res = {}
    dm = {s["stem"]: s for s in ms}
    # a. expansion on the device
    bad = []
    for s in ar:
        for k in ("u", "v", "w", "nw"):
            if not same(cp.asnumpy(s.expand(k, cp)), dm[s["stem"]][k]):
                bad.append((s["stem"], k))
    res["expand_device_bad"] = bad
    # all keys of load_samples (host), bitwise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from equiv_common import diff
    res["load_samples_vs_archive_bad"] = {s["stem"]: d for s in ar for d in [diff(dm[s["stem"]], s)] if d}
    res["stems_equal"] = sorted(dm) == sorted(s["stem"] for s in ar)
    log(f"load_samples vs load_archive: {len(ar)} snapshots, bad {res['load_samples_vs_archive_bad']}, "
        f"stems equal {res['stems_equal']}")
    log(f"expand on device: {len(ar) * 4 - len(bad)}/{len(ar) * 4} bitwise")
    # b. HourGrid weights / setup, both product weightings
    N, sdeg = bands.geometry(band)
    msd = sorted([dict(s) for s in ms], key=lambda s: s["time"])
    ars = sorted(ar, key=lambda s: s["time"])
    for tag, rob, tk in (("taper_r0", 0.0, 30.0), ("notaper_r-0.75", -0.75, None)):
        G1 = hour.HourGrid(msd, N, sdeg, robust=rob, tukey_inner=tk, need_svd=True)
        G2 = hour.HourGrid(ars, N, sdeg, robust=rob, tukey_inner=tk, need_svd=True)
        r = dict(iw_bad=[i for i, (a, b) in enumerate(zip(G1.iw, G2.iw)) if not same(a, b)],
                 sumw_equal=G1.sumw == G2.sumw, K_equal=G1.svd.K == G2.svd.K, chunks_equal=G1.chunks == G2.chunks,
                 npts_equal=G1.npts == G2.npts)
        res[f"hourgrid_{tag}"] = r
        log(f"HourGrid {tag}: {r}")
    # c. chunk inputs in every cache mode (first call and cache hit), vs dicts + auto
    old = os.environ.get("TTCALX_CHUNK_CACHE")
    ref = {}
    try:
        for mode, snaps in (("auto", msd), ("auto", ars), ("host", ars), ("off", ars), ("store", ars), ("device", ars)):
            os.environ["TTCALX_CHUNK_CACHE"] = mode
            G = hour.HourGrid(snaps, N, sdeg, robust=0.0, tukey_inner=30.0, need_svd=True)
            ids = G.chunks[0]
            got = {}
            for rep in range(2):
                for col in ("I", "PSF", "V"):
                    S, vis = G._chunk(ids, col)
                    got[(rep, col)] = [cp.asnumpy(x) for x in (S.x, S.y, S.w, S.wt, S.flip, vis)]
            if not ref:
                ref = got
                continue
            bad = [f"rep{k[0]}:{k[1]}:{n}" for k in ref for n, a, b in zip("xywtfv", ref[k], got[k]) if not same(a, b)]
            res[f"chunk_{mode}_{'ms' if snaps is msd else 'archive'}_bad"] = bad
            log(f"chunk inputs, cache={mode}: {'bitwise' if not bad else bad}")
    finally:
        if old is None:
            os.environ.pop("TTCALX_CHUNK_CACHE", None)
        else:
            os.environ["TTCALX_CHUNK_CACHE"] = old
    return res


def fits_stats(a_path, b_path, radius_frac=0.40):
    from astropy.io import fits
    a = fits.getdata(a_path).astype(np.float64).squeeze()
    b = fits.getdata(b_path).astype(np.float64).squeeze()
    if a.shape != b.shape:
        return dict(shape_mismatch=[a.shape, b.shape])
    bit = fits.getdata(a_path).tobytes() == fits.getdata(b_path).tobytes()
    n = a.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    m = np.hypot(yy - n / 2, xx - n / 2) < radius_frac * n
    ok = m & np.isfinite(a) & np.isfinite(b)
    med = np.median(a[ok])
    sigma = 1.4826 * np.median(np.abs(a[ok] - med))
    d = (a - b)[ok]
    return dict(bitwise=bit, sigma=float(sigma), diff_rms_sigma=float(np.sqrt(np.mean(d ** 2)) / sigma) if sigma else None,
                diff_max_sigma=float(np.max(np.abs(d)) / sigma) if sigma else None, peak_a=float(np.max(a[ok])),
                peak_rel=float((np.max(b[ok]) - np.max(a[ok])) / max(abs(np.max(a[ok])), 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True)
    ap.add_argument("--ms-dir", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lst-hour", type=int, required=True)
    ap.add_argument("--products", default="orca")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ttcalx-image", default=str(Path(sys.executable).parent / "ttcalx-image"))
    ap.add_argument("--cpu", action="store_true", help="dry run without a GPU (numpy stand-in)")
    ap.add_argument("--tests-dir", default="", help="imager tests dir (hour_cpu.py) for --cpu")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    import ttcalx_imager
    log("imager:", ttcalx_imager.__file__)
    ms = sorted(glob.glob(os.path.join(os.path.abspath(a.ms_dir), "*.ms")))     # the archive stores absolute paths
    ctx = None
    if a.cpu:
        sys.path.insert(0, a.tests_dir)
        import hour_cpu
        hour = hour_cpu.import_hour()
        ctx = hour_cpu.cpu_hour(cache="auto", free_gb=1000)
        ctx.__enter__()
        cp = hour.cp
    else:
        import cupy as cp
        from ttcalx_imager import hour
        log("GPU:", cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"])
    from ttcalx_imager import bands, gsiv
    from ttcalx_imager.rotate import parse_phase_centre
    ra, dec = parse_phase_centre(bands.hour_phase_centre(a.lst_hour))
    # nproc=1: the MSes in time order in one process -- the order the archive writer computed R in
    # (msio R depends on the casacore call history at ~1e-10, so a process pool is not bitwise)
    t0 = time.perf_counter(); snaps_ms = hour.load_all(ms, ra, dec, nproc=1); t_ms = time.perf_counter() - t0
    t0 = time.perf_counter(); snaps_ar = gsiv.load_archive(a.archive, ra, dec); t_ar = time.perf_counter() - t0
    res = dict(band=a.band, n_ms=len(ms), n_records=len(snaps_ar), load_ms_s=t_ms, load_archive_s=t_ar)
    res["deterministic"] = deterministic_checks(hour, gsiv, cp, snaps_ms, snaps_ar, ra, dec, a.band, log)
    del snaps_ms, snaps_ar
    if ctx is not None:
        ctx.__exit__(None, None, None)
    # products: the production command, MSes and archive, reps times each
    (out / "ms.txt").write_text("\n".join(ms) + "\n")
    common = ["--band", a.band, "--lst-hour", str(a.lst_hour), "--products", a.products, "--qa", "none",
              "--nproc", "8", "-v"]
    if a.cpu:                       # the w-snapshot reprojection needs a real GPU
        common += ["--mode", "vis"]
    runs = {}
    for src, argv in (("ms", [f"@{out / 'ms.txt'}"]), ("archive", ["--archive", a.archive])):
        for r in range(a.reps):
            tag = f"{src}_r{r}"
            d = out / tag; d.mkdir(exist_ok=True)
            cmd = [a.ttcalx_image, "hour"] + argv + ["--out", str(d / f"{a.band}MHz")] + common
            t0 = time.perf_counter()
            if a.cpu:
                from ttcalx_imager import cli
                with hour_cpu.cpu_hour(cache=os.environ.get("TTCALX_CHUNK_CACHE", "auto")):
                    rc = cli.main(cmd[1:])
                os.environ.pop("TTCALX_CHUNK_CACHE", None)
            else:
                with open(d / "log.txt", "w") as fh:
                    rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            s = json.load(open(d / f"{a.band}MHz-summary.json"))
            runs[tag] = dict(rc=rc, wall_s=time.perf_counter() - t0, timing=s["timing"], rejected=s["rejected"],
                             n_loaded=s["n_loaded"])
            log(f"{tag}: rc {rc}, {runs[tag]['wall_s']:.1f} s, load {s['timing'].get('load + rotation (CPU)', 0):.1f} s")
    res["runs"] = runs
    if a.cpu:                       # the stand-in writes no FITS
        json.dump(res, open(out / "result.json", "w"), indent=1, default=str)
        log(json.dumps(res["deterministic"], default=str))
        return
    # FITS comparison: archive vs ms_r0 against the floor ms_r1 vs ms_r0
    cmpres = {}
    names = sorted(os.path.basename(p) for p in glob.glob(str(out / "ms_r0" / "*.fits")))
    for nm in names:
        ref = out / "ms_r0" / nm
        row = {}
        for tag in runs:
            if tag != "ms_r0" and (out / tag / nm).exists():
                row[tag] = fits_stats(ref, out / tag / nm)
        fl = row.get("ms_r1", {})
        verdict = []
        for tag, st in row.items():
            if not tag.startswith("archive") or "bitwise" not in st:
                continue
            if st["bitwise"]:
                verdict.append(f"{tag}:bitwise")
                continue
            frms = fl.get("diff_rms_sigma") or 0.0
            fmax = fl.get("diff_max_sigma") or 0.0
            ok = st["diff_rms_sigma"] <= max(1.5 * frms, 0.05) and st["diff_max_sigma"] <= max(1.5 * fmax, 0.5)
            verdict.append(f"{tag}:{'PASS' if ok else 'FAIL'}")
        row["verdict"] = verdict
        cmpres[nm] = row
        log(nm, verdict, {t: (st.get("bitwise"), st.get("diff_rms_sigma"), st.get("diff_max_sigma"))
                          for t, st in row.items() if t != "verdict"})
    res["fits"] = cmpres
    res["all_pass"] = all(all(v.endswith(("bitwise", "PASS")) for v in r["verdict"]) for r in cmpres.values()) \
        and all(not v for k, v in res["deterministic"].items() if k.endswith("_bad")) and res["deterministic"]["stems_equal"] \
        and all(not (v["iw_bad"]) and v["sumw_equal"] for k, v in res["deterministic"].items() if k.startswith("hourgrid"))
    json.dump(res, open(out / "result.json", "w"), indent=1, default=str)
    log("ALL PASS" if res["all_pass"] else "NOT ALL PASS -- see result.json")


if __name__ == "__main__":
    main()
