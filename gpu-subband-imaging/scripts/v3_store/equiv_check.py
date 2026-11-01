"""venv (CPU): load_samples(ms) vs load_archive(gsiv) bitwise for every snapshot; sizes and timings.
usage: equiv_check.py <build.json> <out.json> [lustre_copy_dir]"""
import json, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from equiv_common import KEYS, diff, digest
import numpy as np
import ttcalx_imager
from ttcalx_imager.gsiv import load_archive, archive_phase_centre
sys.path.insert(0, os.environ.get('IMAGER_TESTS', '/tmp/nk_v3_store/code/ttcalx_imager/tests'))
import hour_cpu
hour = hour_cpu.import_hour()          # cufinufft stub: this node has no CUDA
load_all = hour.load_all
from ttcalx_imager.msio import load_samples
from ttcalx_imager.rotate import parse_phase_centre

print("imager:", ttcalx_imager.__file__, flush=True)
def main():
    build = json.load(open(sys.argv[1]))
    out = {}
    for band, b in build.items():
        f = b["file"]
        ra, dec = parse_phase_centre(b["phase_centre"])
        ms = sorted(str(p) for p in Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / f"{band}MHz".glob("*.ms"))
        r = dict(n_ms=len(ms), file=f, size=os.path.getsize(f), rec_nbytes=b["rec_nbytes"], n_kept=b["n_kept"])
        # --- timings (warm page cache for both: the MSes and the file were just written / read)
        load_all(ms[:2], ra, dec, nproc=1)                         # warm imports / casacore measures
        t0 = time.perf_counter(); A1 = [load_samples(m, ra, dec) for m in ms]; r["t_ms_serial"] = time.perf_counter() - t0
        t0 = time.perf_counter(); A8 = load_all(ms, ra, dec, nproc=8); r["t_ms_nproc8"] = time.perf_counter() - t0
        del A8
        t0 = time.perf_counter(); L = load_archive(f, ra, dec, backend="ram", verify=True, workers=8); r["t_gsiv_ram_verify"] = time.perf_counter() - t0
        t0 = time.perf_counter(); Lm = load_archive(f, ra, dec, backend="memmap", verify=False); r["t_gsiv_memmap_noverify"] = time.perf_counter() - t0
        t0 = time.perf_counter(); Lv = load_archive(f, ra, dec, backend="memmap", verify=True, workers=8); r["t_gsiv_memmap_verify"] = time.perf_counter() - t0
        del Lv
        # expansion cost of u, v, w, nw (what the hour imager pays per access instead of keeping them)
        t0 = time.perf_counter()
        for s in L:
            for k in ("u", "v", "w", "nw"):
                s[k]
        r["t_expand_uvwnw_all"] = time.perf_counter() - t0
        # --- bitwise: every snapshot, every key of load_samples; both backends
        by = {s["stem"]: s for s in L}; bym = {s["stem"]: s for s in Lm}
        r["stems_equal"] = sorted(by) == sorted(a["stem"] for a in A1) == sorted(bym)
        bad = {}
        for a in A1:
            d = diff(a, by[a["stem"]]) + ["memmap:" + k for k in diff(a, bym[a["stem"]])]
            missing = [k for k in KEYS if k in a and k not in by[a["stem"]]]
            extra_keys = sorted(set(by[a["stem"]].keys()) ^ set(a.keys()))
            if d or missing or extra_keys:
                bad[a["stem"]] = dict(diff=d, missing=missing, key_symdiff=extra_keys)
        r["bitwise_bad"] = bad
        r["n_compared"] = len(A1)
        r["n_values_compared"] = int(sum(a[k].size for a in A1 for k in ("u", "v", "w", "nw", "I", "V")))
        r["time_order_equal"] = [s["stem"] for s in L] == [a["stem"] for a in sorted(A1, key=lambda x: x["time"])]
        # ram footprint: load_samples dicts vs records
        r["bytes_load_samples"] = int(sum(a[k].nbytes for a in A1 for k in ("u", "v", "w", "nw", "I", "V")))
        r["bytes_records"] = int(sum(s.nbytes() for s in L))
        r["middle_pd_equal_ms"] = archive_phase_centre(f) == hour.middle_zenith(ms)
        # --- R recompute path: a different phase centre -> R from casacore, equal to load_samples there
        ra2, dec2 = parse_phase_centre("06h30m00s 37d12m57.057s")
        L2 = load_archive(f, ra2, dec2, backend="memmap", verify=False)
        a2 = load_samples(ms[0], ra2, dec2)
        r["recompute_R_bad"] = diff(a2, [s for s in L2 if s["stem"] == a2["stem"]][0])
        out[band] = r
        print(band, json.dumps({k: v for k, v in r.items() if k not in ("rec_nbytes", "n_kept")}), flush=True)
        # --- Lustre copy (single file): load time from Lustre
        if len(sys.argv) > 3:
            d = Path(sys.argv[3]); d.mkdir(parents=True, exist_ok=True)
            lf = d / Path(f).name
            t0 = time.perf_counter(); shutil.copyfile(f, lf); shutil.copyfile(f + ".idx.jsonl", str(lf) + ".idx.jsonl")
            r["t_copy_to_lustre"] = time.perf_counter() - t0
            t0 = time.perf_counter(); Ll = load_archive(str(lf), ra, dec, backend="ram", verify=True, workers=8)
            r["t_gsiv_lustre_ram_verify"] = time.perf_counter() - t0
            r["lustre_bad"] = sum(1 for a in A1 if diff(a, {s["stem"]: s for s in Ll}[a["stem"]]))
            r["lustre_file"] = str(lf)
            print(band, "lustre", r["t_copy_to_lustre"], r["t_gsiv_lustre_ram_verify"], r["lustre_bad"], flush=True)
            del Ll
        del A1, L, Lm
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
