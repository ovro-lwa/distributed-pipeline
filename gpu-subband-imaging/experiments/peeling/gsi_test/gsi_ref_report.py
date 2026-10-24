"""Reference-set table: GPU (production / CPU-matched) vs CPU TTCal on identical inputs."""
import json, re, sys
import numpy as np
S = sys.argv[1]
d = json.load(open(f"{S}/summary.json"))
for s, r in json.load(open(f"{S}/summary_69rfi.json")).items():
    if "gpu_cpumatch_rfi" in r:
        d[s]["gpu_cpumatch_rfi"] = r["gpu_cpumatch_rfi"]
vp = {k.split("/")[1]: v for k, v in json.load(open(f"{S}/vpol.json")).items()}
SKIPPED = {"41MHz_080033", "41MHz_080103", "41MHz_080113"}   # empty input: the worker's quality gate skips them
HALF_EMPTY = {"41MHz_080023"}                                    # 49% nonzero: passes the 1% gate
cases = [(32, "gpu_prod", "cpu_sky_rfi"), (32, "gpu_cpumatch", "cpu_sky_rfi"),
         (41, "gpu_prod", "cpu_sky_rfi"), (41, "gpu_cpumatch", "cpu_sky_rfi"),
         (69, "gpu_prod", "cpu_sky_rfi"), (69, "gpu_cpumatch", "cpu_sky"), (69, "gpu_cpumatch_rfi", "cpu_sky_rfi")]
rows = []
for band, v, ref in cases:
    snaps = [s for s in d if s.startswith(f"{band}MHz") and s not in SKIPPED | HALF_EMPTY]
    q = lambda f: [f(s) for s in snaps]
    r = q(lambda s: d[s][v]["rms"] / d[s][ref]["rms"])
    c = q(lambda s: d[s][v].get("corr_" + ref, np.nan))
    out = dict(band=band, variant=v, ref=ref, n=len(snaps), rms=float(np.median(r)), rms_min=float(min(r)),
               rms_max=float(max(r)), corr=float(np.nanmedian(c)))
    for t in ("CasA", "CygA", "RFI"):
        x = [d[s][v]["targets"][t]["peak"] / d[s][ref]["targets"][t]["peak"] for s in snaps
             if t in d[s][v]["targets"] and t in d[s][ref]["targets"]]
        out[t] = float(np.median(x)) if x else None
    vr = [vp[s][v]["V_rms"] / vp[s][ref]["V_rms"] for s in snaps if v in vp.get(s, {}) and ref in vp.get(s, {})]
    out["V_rms"] = float(np.median(vr)) if vr else None
    rows.append(out)
    print(out)
cpu = {}
for l in open(f"{S}/cpu_timing.log"):
    m = re.search(r"CPU_ZEST kind=(\w+) ms=\S+/(\w+)\.ms elapsed=([\d.]+)", l)
    if m:
        cpu.setdefault(f"{m.group(2)}:{m.group(1)}", []).append(float(m.group(3)))
timing = {"cpu": {k: float(np.median(v)) for k, v in cpu.items()}}
for f in ("gpu_prod", "gpu_cpumatch"):
    t = [tuple(map(float, re.findall(r"=([\d.]+)", l))) for l in open(f"{S}/{f}.log") if "ZEST_TIMING" in l]
    timing[f] = {k: float(np.median([x[i] for x in t[1:]])) for i, k in enumerate(("read", "peel", "write"))}
print(timing)
half = {s: {v: round(d[s][v]["rms"], 2) for v in ("prep", "cpu_sky_rfi", "gpu_prod", "gpu_cpumatch")} for s in HALF_EMPTY | SKIPPED}
print(half)
json.dump({"rows": rows, "timing": timing, "bad_inputs": half}, open(f"{S}/ref_report.json", "w"), indent=1)
