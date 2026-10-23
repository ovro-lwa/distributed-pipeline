#!/usr/bin/env bash
# End-to-end daemon runs on node-local /tmp copies: fully optimized GPU, and TTCalX on CPU (24 threads),
# both with fast read/write; outputs compared with the unoptimized-I/O GPU run in /tmp/nk_peel/wt/slow.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
B=/tmp/nk_peel/e2e; mkdir -p $B
SNAPS="23MHz_0300 41MHz_1200 69MHz_1200"
R="ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1"
CM="ZEST_UVW_CONVENTION=geom ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact"
for n in gpu cpu24; do
  mkdir -p $B/$n; for s in $SNAPS; do p=( $PX/data/$s/*_prep.ms ); cp -r ${p[0]} $B/$n/$s.ms; done
  extra=""; [ $n = cpu24 ] && extra="ZEST_DEVICE=cpu JULIA_NUM_THREADS=24"
  t0=$(date +%s.%N)
  env $R $CM ZEST_FAST_WRITE=1 ZEST_FAST_READ=1 $extra GPU=0 $PX_SCRIPTS/gpu_zest.sh $PX/ttcalx_fast $PX/repo_fast/workers/zest_daemon.jl $SKY_MODEL $(for s in $SNAPS; do echo $B/$n/$s.ms; done) 2>&1 | grep -E "^ZESTED|^ZEST_TIMING|FAILED|DAEMON_READY" | sed "s/^/E2E $n /"
  echo "E2E $n wall=$(echo "$(date +%s.%N)-$t0" | bc)"
done
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
python - $B $SNAPS <<'PY'
import sys, numpy as np
from casacore.tables import table
B=sys.argv[1]
for s in sys.argv[2:]:
    ref=table(f"/tmp/nk_peel/wt/slow/{s}.ms",ack=False)
    for n in ("gpu","cpu24"):
        b=table(f"{B}/{n}/{s}.ms",ack=False)
        print("E2E", s, n, "DATA identical to unoptimized-I/O GPU run:", np.array_equal(ref.getcol("DATA"), b.getcol("DATA")),
              "FLAG identical:", np.array_equal(ref.getcol("FLAG"), b.getcol("FLAG")),
              "max|diff|/rms:", float(np.abs(ref.getcol("DATA")-b.getcol("DATA")).max()/np.sqrt(np.mean(np.abs(ref.getcol("DATA"))**2))))
PY
echo E2E_DONE
