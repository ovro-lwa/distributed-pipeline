"""Make a calibrated, peel-ready copy of one averaged snapshot MS.

The source MS is only read (copied first).  Mirrors both pipelines'
pre-peel steps: bad-antenna flags, bandpass + XY-phase applycal, and a
split of CORRECTED_DATA into DATA so CPU TTCal and TTCalX read identical
input.  The averaged night-time MSes already have 48 channels (chanbin 4).

    python prep_ms.py SRC_MS OUT_DIR --bp BP_TABLE [--xy XY] [--badants CSV|auto]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gpu_subband_imaging.stages.casarun import run_py  # noqa: E402
from gpu_subband_imaging.stages.flag import badants  # noqa: E402

_SNIPPET = """
import numpy as np
from casacore.tables import table
from casatasks import applycal, flagdata, mstransform
from gpu_subband_imaging.subbands import spwmap

ms, bp, xy, out, ba = {ms!r}, {bp!r}, {xy!r}, {out!r}, {ba!r}
if ba:
    flagdata(vis=ms, mode="manual", antenna=ba, flagbackup=False)
bp_map = spwmap(ms, bp)
xy_map = spwmap(ms, xy)
applycal(vis=ms, gaintable=[bp, xy], spwmap=[bp_map, xy_map],
         flagbackup=False, calwt=False)
mstransform(vis=ms, outputvis=out, datacolumn="corrected")
print("SPWMAP", bp_map, xy_map)
"""


def obs_mjd(ms: str) -> float:
    from casacore.tables import table
    with table(ms + "/OBSERVATION", ack=False) as t:
        return float(t.getcol("TIME_RANGE")[0][0]) / 86400.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_ms")
    ap.add_argument("out_dir")
    ap.add_argument("--bp", required=True)
    ap.add_argument("--xy", default=os.environ.get("XY_TABLE"))
    ap.add_argument("--badants", default="auto",
                    help="'auto' (MNC lookup), '' (none) or a correlator CSV")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(a.src_ms.rstrip("/")).name.replace("_averaged.ms", "")
    work = out_dir / f"{name}_raw.ms"
    prep = out_dir / f"{name}_prep.ms"
    if prep.exists():
        print("exists", prep)
        return 0
    if work.exists():
        shutil.rmtree(work)
    subprocess.run(["cp", "-r", a.src_ms, str(work)], check=True)

    ba = a.badants
    if ba == "auto":
        ba = badants(obs_mjd(str(work)), os.environ["DEV_PY"])
    run_py(_SNIPPET.format(ms=str(work), bp=a.bp, xy=a.xy, out=str(prep), ba=ba),
           timeout=1800)
    shutil.rmtree(work)
    (out_dir / f"{name}_prep.json").write_text(json.dumps({
        "src_ms": a.src_ms, "bp": a.bp, "xy": a.xy, "badants": ba,
        "n_badants": len([x for x in ba.split(",") if x]),
    }, indent=1))
    print("prepared", prep, "badants", ba)
    return 0


if __name__ == "__main__":
    sys.exit(main())
