#!/usr/bin/env bash
# write_gpu_to_ms! vs write_ms_fast!: same peel, both write paths, on node-local /tmp copies (3 MS).
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
B=/tmp/nk_peel/wt; mkdir -p $B
SNAPS="23MHz_0300 41MHz_1200 69MHz_1200"
R="ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1"
CM="ZEST_UVW_CONVENTION=geom ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact"
for n in slow fast; do
  mkdir -p $B/$n; for s in $SNAPS; do p=( $PX/data/$s/*_prep.ms ); cp -r ${p[0]} $B/$n/$s.ms; done
  fw=0; [ $n = fast ] && fw=1
  env $R $CM ZEST_FAST_WRITE=$fw GPU=0 $PX_SCRIPTS/gpu_zest.sh $PX/ttcalx_fast $PX/repo_fast/workers/zest_daemon.jl $SKY_MODEL $(for s in $SNAPS; do echo $B/$n/$s.ms; done) 2>&1 | grep -E "^ZESTED|^ZEST_TIMING|FAILED" | sed "s/^/WT $n /"
done
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate $PY_ENV
python - $B $SNAPS <<'PY'
import sys, numpy as np
from casacore.tables import table
B=sys.argv[1]
for s in sys.argv[2:]:
    a=table(f"{B}/slow/{s}.ms",ack=False); b=table(f"{B}/fast/{s}.ms",ack=False)
    print("WT", s, "DATA identical:", np.array_equal(a.getcol("DATA"), b.getcol("DATA")),
          "FLAG identical:", np.array_equal(a.getcol("FLAG"), b.getcol("FLAG")),
          "other columns identical:", all(np.array_equal(a.getcol(c), b.getcol(c)) for c in ("UVW", "ANTENNA1", "ANTENNA2", "TIME")))
PY
echo WT_DONE
