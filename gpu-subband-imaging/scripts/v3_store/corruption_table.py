"""py38: the corruption table -- every damaged archive either loads the complete correct set or
fails loudly, and the hour job never deletes a kept MS it did not image.

usage: corruption_table.py <clean 82 MHz gsiv> <other-band gsiv> <imager src> <venv python> <out.json>

Each case copies the clean archive into a pipeline-like tree, damages it, adds the worker's
expected list (all 12 stems) and, with kept=True, 12 kept MS directories, then runs
hour_lst.run_band with a stand-in imager that loads the archive with the real
ttcalx_imager.gsiv.load_archive (strict, band-checked) or lists the MSes, and reports what it
loaded.  Columns: load_archive (strict) on the damaged file, run_band outcome, stems imaged,
kept MSes left.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

from gpu_subband_imaging import hour_lst
from gpu_subband_imaging.config import HourImagingParams, V3Params
from gpu_subband_imaging.v3 import hourstore as hs

clean, other, imager_src, venv_py, out_json = sys.argv[1:6]
FAKE = """#!{py}
import json, os, sys
from ttcalx_imager import gsiv
a = sys.argv
if "--help" in a:
    print("--archive"); sys.exit(0)
out = a[a.index("--out") + 1]
if "--archive" in a:
    files = [l.strip() for l in open(a[a.index("--archive") + 1][1:]) if l.strip()]
    try:
        stems = [s["stem"] for s in gsiv.load_archive(files, band=a[a.index("--band") + 1], verify=True)]
    except gsiv.GSIVError as e:
        print("GSIVError:", e); sys.exit(1)
    src = "archive"
else:
    stems = sorted(os.path.basename(l.strip())[:-3] for l in open(a[2][1:]) if l.strip()); src = "ms"
open({log!r}, "a").write(json.dumps(dict(src=src, n=len(stems))) + "\\n")
open(out + "-I-Deep-Taper-Robust-0-image.fits", "w").write("x")
json.dump(dict(timing={{}}, n_loaded=len(stems), rejected=[], products={{}}, qa=[dict(stem=s) for s in stems]),
          open(out + "-summary.json", "w"))
"""


def flip(f, rec_i, back=500):
    e = hs.scan(f).entries[rec_i]
    with open(f, "r+b") as fh:
        fh.seek(e["offset"] + e["nbytes"] - back)
        b = fh.read(1); fh.seek(-1, 1); fh.write(bytes([b[0] ^ 1]))


def damage(case, f):
    sc = hs.scan(f)
    E = sc.entries
    if case == "truncated_mid":
        with open(f, "r+b") as fh:
            fh.truncate(E[5]["offset"] + E[5]["nbytes"] // 2)
    elif case == "torn_last":
        with open(f, "r+b") as fh:
            fh.truncate(E[-1]["offset"] + E[-1]["nbytes"] - 1000)
    elif case == "bit_flip":
        flip(f, 2)
    elif case == "index_deleted":
        hs.idx_path(f).unlink()
    elif case == "index_cut":
        t = hs.idx_path(f).read_text().splitlines()
        hs.idx_path(f).write_text("\n".join(t[:-1]) + "\n" + t[-1][:40])
    elif case == "foreign_band":            # a valid record of another band, written past the writer
        _, _, recs = hs.read_records(other)
        r = recs[0]
        meta = dict(r.meta, stem=r.meta["stem"].replace("_55MHz", "_82MHz").replace("101007", "101500"))
        hdr, pieces, nbytes, crc = hs.encode_block(hs.MAGIC_REC, len(E) + 1, meta, r.arrays)
        with open(f, "ab") as fh:
            fh.write(hdr); [fh.write(x) for x in pieces]
        with open(hs.idx_path(f), "a") as fh:
            fh.write(json.dumps(hs._entry(meta, r.arrays, len(E) + 1, sc.end, nbytes, crc)) + "\n")
    elif case == "raw_block":               # raw bytes of another file's block appended
        o = hs.scan(other).entries[0]
        with open(other, "rb") as fh:
            fh.seek(o["offset"]); blob = fh.read(o["nbytes"])
        with open(f, "ab") as fh:
            fh.write(blob)
    elif case == "missing":
        f.unlink(); hs.idx_path(f).unlink()


def main():
    _, _, recs = hs.read_records(clean, verify=False)
    stems = sorted(r.stem for r in recs)
    rows = []
    for case in ("clean", "truncated_mid", "torn_last", "bit_flip", "index_deleted", "index_cut",
                 "foreign_band", "raw_block", "missing"):
        for kept in (True, False):
            tmp = Path(tempfile.mkdtemp(prefix=f"ct_{case}_", dir=os.environ.get("SCRATCH", "/tmp")))
            log = tmp / "imager.jsonl"
            fake = tmp / "fake-ttcalx-image"
            fake.write_text(FAKE.format(py=venv_py, log=str(log))); fake.chmod(0o755)
            v = V3Params(archive="iv", archive_root=str(tmp / "arch"), hour_source="archive")
            pl = SimpleNamespace(date="2026-10-31", output_root=str(tmp / "out"), layout="lst", run_label="Run_T",
                                 keep_ms_root=str(tmp / "keep"), keep_ms_local=None, v3=v, sun_cut=True,
                                 snapshot_phase_centre="lst",
                                 hour_imaging=HourImagingParams(ttcalx_image=str(fake), nproc=4, delete_kept_ms=True,
                                                                env={"PYTHONPATH": imager_src}),
                                 movie=SimpleNamespace(enabled=False), cal=SimpleNamespace(bp="b", xy="x", sources="s",
                                                                                           rfi_sources=None),
                                 env=SimpleNamespace(ttcalx=str(tmp)), imaging=SimpleNamespace(),
                                 peel_for_band=lambda b: SimpleNamespace(rfi=False))
            cfg = SimpleNamespace(pipeline=pl, cluster=SimpleNamespace(work_root=str(tmp / "work")))
            f = hs.archive_path(cfg, 82, "04h", 0)
            f.parent.mkdir(parents=True)
            shutil.copyfile(clean, f); shutil.copyfile(hs.idx_path(clean), hs.idx_path(f))
            for s in stems:
                hs.add_expected(f, s, batch=0)
            kd = Path(pl.keep_ms_root) / "82MHz" / "2026-10-31" / "04h"
            if kept:
                for s in stems:
                    (kd / f"{s}.ms").mkdir(parents=True)
            damage(case, f)
            la = "-"
            if f.exists():
                r = subprocess.run([venv_py, "-c", "import sys; from ttcalx_imager import gsiv; "
                                    "print(len(gsiv.load_archive(sys.argv[1], band=82)))", str(f)],
                                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=imager_src))
                msg = (r.stderr.strip().splitlines() or ["?"])[-1].split("GSIVError: ")[-1]
                la = f"ok {r.stdout.strip()}" if r.returncode == 0 else "raises: " + msg.replace(str(f), "<file>")[:110]
            import gpu_subband_imaging.metadata as md
            md.software_versions = lambda cfg: {}
            try:
                rec = hour_lst.run_band(cfg, 82, "04h", gpu=None)
                outcome = f"{rec['status']} via {rec.get('hour_source')}"
            except Exception as e:
                outcome = "FAILED (raises): " + ("archive unusable, no complete kept MSes" if "unusable" in str(e)
                                                 else str(e)[:80])
            calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
            left = len(list(kd.iterdir())) if kd.is_dir() else 0
            row = dict(case=case, kept=kept, load_archive=la, run_band=outcome,
                       imaged=[f"{c['src']}:{c['n']}" for c in calls], kept_left=left)
            rows.append(row)
            print(json.dumps(row), flush=True)
            shutil.rmtree(tmp)
    json.dump(rows, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
