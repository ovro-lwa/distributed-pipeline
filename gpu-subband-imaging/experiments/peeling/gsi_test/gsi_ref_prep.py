"""Prepare the test snapshots exactly as the GPU worker does (do_aoflag: false):
stage, bad antennas from MNC, bandpass + XY-phase applycal, average by chanbin."""
import shutil, sys
from pathlib import Path

from gpu_subband_imaging.config import Config
from gpu_subband_imaging.stages import calibrate, flag, stage_ms
from gpu_subband_imaging.worker import _mjd, _stamp

T = Path("/lustre/nkosogor/peel_exp/gsi_test")
cfg = Config.load(T / "config-cpumatch").pipeline
for band in (32, 41, 69):
    tars = [l for l in (T / f"manifest_{band}.txt").read_text().split() if l]
    bad = flag.badants(_mjd(_stamp(tars[0])), f"{cfg.env.dev_env}/bin/python")
    print("BADANTS", band, len(bad.split(",")) if bad else 0, flush=True)
    for tar in tars:
        stamp = _stamp(tar)
        snap = T / "ref" / f"{band}MHz_{stamp.split('_')[1]}"
        out = snap / "prep.ms"
        if out.exists():
            continue
        work = Path("/tmp/nk_gsi_ref") / stamp
        ms = stage_ms.stage(tar, work)
        snap.mkdir(parents=True, exist_ok=True)
        calibrate.flag_cal_average(ms, out, cfg.cal.bp, cfg.cal.xy, bad, cfg.chanbin)
        shutil.rmtree(work, ignore_errors=True)
        print("PREP", snap.name, flush=True)
print("PREP_DONE")
