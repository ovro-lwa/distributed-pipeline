"""Write the test configs (from last night's production config) and manifests."""
import copy, glob, sys
from pathlib import Path
import yaml

PX = Path("/lustre/nkosogor/peel_exp")
T = PX / "gsi_test"
PROD = Path("/opt/devel/pipeline/gpu-subband-imaging/config-allsubbands-auto")
BANDS, HOUR, N = [32, 41, 69], "08", 10
RFI = "/lustre/gh/calibration/pipeline/reference/sources/rfi_43.2_ver20251101.json"
CPU_MATCHED = dict(solver="rk4", maxiter=5, tolerance=1.0e-4, minuvw=5.0, min_source_elevation_deg=0.0,
                   require_convergence=False, model="exact", uvw="geometric", flag_model=True,
                   resubtract=True, fast_io=True, rfi=True)
base = yaml.safe_load((PROD / "pipeline.yaml").read_text())
cluster = yaml.safe_load((PROD / "cluster.yaml").read_text())
cluster["control"]["state_dir"] = str(T / "_state")
cluster["defaults"]["work_root"] = "/tmp/nk_gsi_work"
variants = {
    "prod": dict(ttcalx="/opt/devel/nkosogor/nkosogor/TTCalX", peel={}),
    "prod_branch": dict(ttcalx=str(PX / "ttcalx_branch"), peel={}),
    "cpumatch": dict(ttcalx=str(PX / "ttcalx_branch"), peel=CPU_MATCHED),
    "cpumatch_cpu": dict(ttcalx=str(PX / "ttcalx_branch"), peel={**CPU_MATCHED, "device": "cpu"}),
}
for name, v in variants.items():
    d = copy.deepcopy(base)
    d["bands"], d["hours"] = BANDS, [HOUR]
    d["output_root"] = str(T / "out" / name)
    d["env"]["ttcalx"] = v["ttcalx"]
    d["cal"]["rfi_sources"] = RFI
    d["peel"].update(v["peel"])
    if v["peel"].get("rfi"):
        # The RFI model has flux only at 14.8-40.9 MHz.
        d["peel_overrides"] = {b: {"rfi": False} for b in BANDS if b > 41}
    cdir = T / f"config-{name}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "pipeline.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    (cdir / "cluster.yaml").write_text(yaml.safe_dump(cluster, sort_keys=False))
    (cdir / "subbands.yaml").write_text((PROD / "subbands.yaml").read_text())
for b in BANDS:
    tars = sorted(glob.glob(f"/lustre/pipeline/slow/{b}MHz/2026-10-23/{HOUR}/*.ms.tar"))[:N]
    (T / f"manifest_{b}.txt").write_text("\n".join(tars) + "\n")
print("ok", sorted(p.name for p in T.iterdir()))
