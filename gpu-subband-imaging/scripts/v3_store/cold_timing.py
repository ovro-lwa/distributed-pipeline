"""venv: cold-cache load times (pages dropped with posix_fadvise DONTNEED before each run).
usage: cold_timing.py <build.json> <equiv.json> <out.json>"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ.get("IMAGER_TESTS", "/tmp/nk_v3_store/code/ttcalx_imager/tests"))


def drop(paths):
    n = 0
    for p in paths:
        for f in ([p] if os.path.isfile(p) else [os.path.join(d, x) for d, _, fs in os.walk(p) for x in fs]):
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
            finally:
                os.close(fd)
    return n


def resident(path):
    """fraction of a file's pages in the page cache (mincore via mmap is not exposed; use fincore if present)"""
    import subprocess
    try:
        o = subprocess.run(["fincore", "-b", "-n", "-o", "RES", path], capture_output=True, text=True).stdout.split()
        return int(o[0]) / max(1, os.path.getsize(path))
    except Exception:
        return None


def main():
    import hour_cpu
    hour = hour_cpu.import_hour()
    from ttcalx_imager.gsiv import load_archive
    from ttcalx_imager.msio import load_samples
    from ttcalx_imager.rotate import parse_phase_centre
    build = json.load(open(sys.argv[1])); eq = json.load(open(sys.argv[2]))
    out = {}
    for band, b in build.items():
        ra, dec = parse_phase_centre(b["phase_centre"])
        ms = sorted(str(p) for p in Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / f"{band}MHz".glob("*.ms"))
        r = {}
        hour.load_all(ms[:1], ra, dec, nproc=1)                       # imports / measures warm
        for rep in range(2):
            drop(ms); t0 = time.perf_counter(); [load_samples(m, ra, dec) for m in ms]
            r.setdefault("ms_serial_cold", []).append(time.perf_counter() - t0)
            drop(ms); t0 = time.perf_counter(); hour.load_all(ms, ra, dec, nproc=8)
            r.setdefault("ms_nproc8_cold", []).append(time.perf_counter() - t0)
            for tag, f in (("gsiv_tmp", b["file"]), ("gsiv_lustre", eq[band].get("lustre_file"))):
                if not f:
                    continue
                drop([f]); res0 = resident(f)
                t0 = time.perf_counter(); load_archive(f, ra, dec, backend="ram", verify=True, workers=8)
                r.setdefault(tag + "_ram_crc_cold", []).append(time.perf_counter() - t0)
                r.setdefault(tag + "_resident_before", []).append(res0)
        out[band] = r
        print(band, json.dumps(r), flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
