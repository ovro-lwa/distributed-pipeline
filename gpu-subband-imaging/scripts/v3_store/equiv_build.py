"""py38 (worker env): build one GSIV per band through the worker hook (hourstore.ArchiveClient ->
'hourstore serve' in the imager venv), with the env construction of Worker._archive_client.

usage: equiv_build.py <out_dir> <imager_src> <venv_python> band...
"""
import json, os, sys, time
from pathlib import Path
from gpu_subband_imaging import lst
from gpu_subband_imaging.v3 import hourstore

out_dir, imager_src, py = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out_dir.mkdir(parents=True, exist_ok=True)
src = str(Path(hourstore.__file__).resolve().parents[2])
env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}   # hour_image.imager_env
env.update(NUMPY_MADVISE_HUGEPAGE="0", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
           PYTHONPATH=src + os.pathsep + imager_src)
res = {}
for band in sys.argv[4:]:
    ms = sorted(str(p) for p in (Path(os.environ.get("MS_ROOT", "/tmp/nk_v3_store/ms")) / f"{band}MHz").glob("*.ms"))
    hours = lst.lst_hours([Path(m).name for m in ms])
    assert len(set(hours)) == 1, hours
    pc = lst.phase_centre(hours[0])
    out = out_dir / f"{band}MHz-2026-10-31-{lst.label(hours[0])}-peeled-b000.gsiv"
    for p in (out, hourstore.idx_path(out)):
        if p.exists():
            p.unlink()
    hdr = hourstore.header_for(int(band), "2026-10-31", lst.label(hours[0]), pc, "Run_v3store_test", batch=0,
                              cal_bp="calibration_2026-10-31_21h.B.flagged", cal_xy="xyphase_delay_pos_3.8643ns.Xf")
    t0 = time.perf_counter()
    c = hourstore.ArchiveClient(py, out, pc, hdr, env=env)
    t_start = time.perf_counter() - t0
    rows = []
    for m in ms:
        t1 = time.perf_counter()
        r = c.add(m)
        r["rtt_s"] = time.perf_counter() - t1
        rows.append(r)
    c.close()
    dup = None      # (a re-added stem supersedes its record -- unit tests; its R would come from
    #                another casacore call history, see hourstore docstring)
    sc = hourstore.scan(out, verify=True)
    res[band] = dict(file=str(out), phase_centre=pc, lst_hour=hours[0], start_s=t_start, rows=rows, dup=dup,
                     size=out.stat().st_size, records=len(sc.entries), clean=sc.clean,
                     rec_nbytes=[e["nbytes"] for e in sc.entries], n_kept=[e["n_kept"] for e in sc.entries])
    print(band, "records", len(sc.entries), "size", out.stat().st_size, "clean", sc.clean, "dup", dup,
          "build_s mean %.3f write_s mean %.3f rtt mean %.3f" % (
              sum(r["build_s"] for r in rows) / len(rows), sum(r["write_s"] for r in rows) / len(rows),
              sum(r["rtt_s"] for r in rows) / len(rows)), flush=True)
json.dump(res, open(out_dir / "build.json", "w"), indent=1)
