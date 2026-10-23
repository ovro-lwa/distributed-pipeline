#!/usr/bin/env bash
# Timing of the CPU-exact configurations: GPU (new exact-model kernel) and
# TTCalX on the host (ZEST_DEVICE=cpu) at several thread counts; outputs are
# kept for equality checks and images.  Inputs are copied; nothing is deleted.
#   bench2.sh [SNAPS]
source "$(dirname "$0")/env.sh"
B=$PX/runs/bench2_$(date +%Y%m%d_%H%M%S); mkdir -p $B
SNAPS=${1:-"13MHz_0800 23MHz_0300 32MHz_0800 41MHz_1200 50MHz_0300 59MHz_0800 69MHz_1200 82MHz_0300"}
cp_set() { mkdir -p $B/$1; for s in $2; do p=( $PX/data/$s/*_prep.ms ); cp -r ${p[0]} $B/$1/$s.ms; done; }
run() { # name snaps env...
  local n=$1 snaps=$2; shift 2; cp_set $n "$snaps"
  t0=$(date +%s.%N)
  env "$@" GPU=0 $PX_SCRIPTS/gpu_zest.sh $TTCALX_EXP $PX_REPO/workers/zest_daemon.jl $SKY_MODEL \
      $(for s in $snaps; do echo $B/$n/$s.ms; done) > $B/$n.log 2>&1
  t1=$(date +%s.%N)
  awk -v n=$n '/^ZESTED/{split($2,a,"/"); ms=a[length(a)]; t=$3} /^ZEST_TIMING/{print "BENCH2", n, ms, t, $2, $3, $4}' $B/$n.log
  echo "BENCH2 $n wall_total=$(echo "$t1-$t0"|bc)"
}
R="ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1"
CM="ZEST_UVW_CONVENTION=geom ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact"
run gpu_exact_after "$SNAPS" $R $CM
run gpu_exact_joint "$SNAPS" $R $CM ZEST_RFI_MODE=joint
run gpu_kernel_after "$SNAPS" $R ${CM% ZEST_GENVIS=exact}
run cpu48_exact_after "$SNAPS" $R $CM ZEST_DEVICE=cpu JULIA_NUM_THREADS=48
run cpu48_exact_joint "$SNAPS" $R $CM ZEST_RFI_MODE=joint ZEST_DEVICE=cpu JULIA_NUM_THREADS=48
for t in 1 4 12 24; do
  run cpu${t}_exact_after "41MHz_1200 69MHz_1200" $R $CM ZEST_DEVICE=cpu JULIA_NUM_THREADS=$t
done
echo "BENCH2_DIR $B"
echo BENCH2_DONE
