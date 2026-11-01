"""venv, CPU: the hour job through the real CLI, from MSes and from the GSIV archive, with the GPU
kernels replaced by tests/hour_cpu (deterministic binning gridder that logs its inputs).
usage: equiv_hour_cpu.py <build.json> <imager_tests_dir> <out.json> band..."""
import json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import numpy as np
import hour_cpu
hour = hour_cpu.import_hour()
from ttcalx_imager import cli
print("imager:", hour.__file__, flush=True)

def main():
    build = json.load(open(sys.argv[1]))
    res = {}
    for band in sys.argv[4:]:
        b = build[band]
        ms = sorted(str(p) for p in Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / f"{band}MHz".glob("*.ms"))
        tmp = Path(tempfile.mkdtemp(prefix=f"hcpu{band}_", dir=os.environ.get("SCRATCH", "/tmp")))
        (tmp / "ms.txt").write_text("\n".join(ms) + "\n")
        stems = [Path(m).name[:-3] for m in ms]
        (tmp / "x.txt").write_text(stems[4] + "\n")
        # --nproc 1: MSes read in time order in one process, the order the archive writer computed R
        # in (load_samples' R depends on the casacore call history at ~1e-10; ms_auto_np8 shows it)
        common = ["--band", band, "--lst-hour", str(b["lst_hour"]), "--products", "orca", "--qa", "none",
                  "--mode", "vis"]
        runs = {}

        def go(tag, argv, cache, free_gb, nproc=1):
            t0 = time.perf_counter()
            with hour_cpu.cpu_hour(cache=cache, free_gb=free_gb) as rec:
                rc = cli.main(["hour"] + argv + ["--out", str(tmp / tag / f"{band}MHz"), "--nproc", str(nproc)] + common)
            summ = json.load(open(tmp / tag / f"{band}MHz-summary.json"))
            runs[tag] = (rec, summ, time.perf_counter() - t0, rc)
            print(band, tag, "rc", rc, "images", len(rec.images), "calls", len(rec.calls), "rejected", summ["rejected"],
                  "%.1f s" % (time.perf_counter() - t0), "load %.1f s" % summ["timing"]["load + rotation (CPU)"], flush=True)

        # v2: MSes + index exclude (snapshot 4) / v3: archive + stem exclude, every chunk-cache mode
        go("ms_auto", [f"@{tmp / 'ms.txt'}", "--exclude", "4"], "auto", 1000)
        go("ms_host", [f"@{tmp / 'ms.txt'}", "--exclude", "4"], "auto", 0)
        go("ms_auto_np8", [f"@{tmp / 'ms.txt'}", "--exclude", "4"], "auto", 1000, nproc=8)
        go("gsiv_auto", ["--archive", b["file"], "--exclude-stems", f"@{tmp / 'x.txt'}"], "auto", 1000)
        go("gsiv_host", ["--archive", b["file"], "--exclude-stems", f"@{tmp / 'x.txt'}"], "auto", 0)
        go("gsiv_store_dev", ["--archive", b["file"], "--exclude-stems", f"@{tmp / 'x.txt'}"], "store", 1000)
        go("gsiv_store_nocache", ["--archive", b["file"], "--exclude-stems", f"@{tmp / 'x.txt'}"], "store", 0)
        go("gsiv_memmap_off", ["--archive", b["file"], "--archive-backend", "memmap", "--exclude-stems",
                               f"@{tmp / 'x.txt'}"], "off", 0)
        ref_rec, ref_s, _, _ = runs["ms_auto"]
        def strip(s):           # without timings; the per-run output directory replaced by a placeholder
            t = json.dumps({k: v for k, v in s.items() if k != "timing"}, sort_keys=True)
            for tag in runs:
                t = t.replace(str(tmp / tag) + "/", "<OUT>/")
            return t
        gcalls = lambda r: [c for c in r.calls if c[0] in ("grid", "degrid", "grid_psf")]
        out = {}
        for tag, (rec, s, dt, rc) in runs.items():
            img_bad = [k for k in ref_rec.images if k not in rec.images or
                       ref_rec.images[k].tobytes() != rec.images[k].tobytes() or ref_rec.images[k].dtype != rec.images[k].dtype]
            out[tag] = dict(rc=rc, n_images=len(rec.images), images_bad=img_bad, extra_images=sorted(set(rec.images) - set(ref_rec.images)),
                            summary_equal=strip(s) == strip(ref_s), grid_calls_equal=gcalls(rec) == gcalls(ref_rec),
                            n_grid_calls=len(gcalls(rec)), n_sampleset=sum(1 for c in rec.calls if c[0] == "SampleSet"),
                            wall_s=dt, load_s=s["timing"]["load + rotation (CPU)"], rejected=s["rejected"],
                            n_loaded=s["n_loaded"])
            print(band, tag, json.dumps(out[tag]), flush=True)
        res[band] = out
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    json.dump(res, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
