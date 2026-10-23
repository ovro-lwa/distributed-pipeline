#!/usr/bin/env bash
# Regression: experimental TTCalX (ttcalx_fast) vs the feature branch (ttcalx_branch), same daemon logic, real MSes.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
B=/tmp/nk_reg; mkdir -p $B
sed -e 's/gpu_stefcal_exp!(/solve_gains!(/' -e 's/geometry=rfi_geometry,//' $PX/repo_fast/workers/zest_daemon.jl > $B/zest_daemon_branch.jl
SNAPS="23MHz_0300 41MHz_1200 69MHz_1200"
R="ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1"
CM="ZEST_UVW_CONVENTION=geom ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact ZEST_FAST_READ=1 ZEST_FAST_WRITE=1"
run() { # name ttcalx daemon extra-env
  local n=$1 tx=$2 dm=$3; shift 3
  mkdir -p $B/$n; for s in $SNAPS; do p=( $PX/data/$s/*_prep.ms ); rm -rf $B/$n/$s.ms; cp -r ${p[0]} $B/$n/$s.ms; done
  env $R $CM "$@" GPU=0 $PX_SCRIPTS/gpu_zest.sh $tx $dm $SKY_MODEL $(for s in $SNAPS; do echo $B/$n/$s.ms; done) 2>&1 | grep -E "^ZESTED|FAILED|ERROR" | sed "s/^/REG $n /"
}
run exp $PX/ttcalx_fast $PX/repo_fast/workers/zest_daemon.jl
run branch $PX/ttcalx_branch $B/zest_daemon_branch.jl
run branch_cpu $PX/ttcalx_branch $B/zest_daemon_branch.jl ZEST_DEVICE=cpu JULIA_NUM_THREADS=24
run exp_reg $PX/ttcalx_fast $PX/repo_fast/workers/zest_daemon.jl ZEST_RFI_MODE=joint ZEST_REG_MAX_OFFSET_DEG=1 ZEST_REG_TAPER_DEG=1
run branch_reg $PX/ttcalx_branch $B/zest_daemon_branch.jl ZEST_RFI_MODE=joint ZEST_REG_MAX_OFFSET_DEG=1 ZEST_REG_TAPER_DEG=1
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
python - $B $SNAPS <<'PY'
import sys, numpy as np
from casacore.tables import table
B = sys.argv[1]
for s in sys.argv[2:]:
    for a, b in (("exp", "branch"), ("exp", "branch_cpu"), ("exp_reg", "branch_reg")):
        x, y = table(f"{B}/{a}/{s}.ms", ack=False), table(f"{B}/{b}/{s}.ms", ack=False)
        dx, dy = x.getcol("DATA"), y.getcol("DATA")
        rms = np.sqrt(np.mean(np.abs(dx) ** 2))
        print("REG", s, a, "vs", b, "identical:", np.array_equal(dx, dy), "max|diff|/rms: %.2e" % (np.abs(dx - dy).max() / rms),
              "FLAG identical:", np.array_equal(x.getcol("FLAG"), y.getcol("FLAG")))
PY
rm -rf $B
echo REG_DONE
