#!/usr/bin/env bash
# Optimized TTCalX (ttcalx_fast: exact-model kernel, GPU short-baseline flags,
# coalesced StefCal step, numpy write-back) on the same 8 snapshots.
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
B=$PX/runs/bench3_$(date +%Y%m%d_%H%M%S); mkdir -p $B
SNAPS="13MHz_0800 23MHz_0300 32MHz_0800 41MHz_1200 50MHz_0300 59MHz_0800 69MHz_1200 82MHz_0300"
cp_set() { mkdir -p $B/$1; for s in $2; do p=( $PX/data/$s/*_prep.ms ); cp -r ${p[0]} $B/$1/$s.ms; done; }
run() { # name snaps env...
  local n=$1 snaps=$2; shift 2; cp_set $n "$snaps"
  t0=$(date +%s.%N)
  env "$@" GPU=0 $PX_SCRIPTS/gpu_zest.sh $PX/ttcalx_fast $PX/repo_fast/workers/zest_daemon.jl $SKY_MODEL \
      $(for s in $snaps; do echo $B/$n/$s.ms; done) > $B/$n.log 2>&1
  t1=$(date +%s.%N)
  awk -v n=$n '/^ZESTED/{split($2,a,"/"); ms=a[length(a)]; t=$3} /^ZEST_TIMING/{print "BENCH3", n, ms, t, $2, $3, $4}' $B/$n.log
  grep FAILED $B/$n.log
  echo "BENCH3 $n wall_total=$(echo "$t1-$t0"|bc)"
}
R="ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1"
CM="ZEST_UVW_CONVENTION=geom ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact"
run fast_after "$SNAPS" $R $CM ZEST_FAST_WRITE=1
run fast_joint "$SNAPS" $R $CM ZEST_RFI_MODE=joint ZEST_FAST_WRITE=1
run fastcpu48_after "$SNAPS" $R $CM ZEST_FAST_WRITE=1 ZEST_DEVICE=cpu JULIA_NUM_THREADS=48
echo "BENCH3_DIR $B"; echo BENCH3_DONE
