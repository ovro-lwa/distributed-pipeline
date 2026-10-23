#!/usr/bin/env bash
# fast96.sh DATE_TAG   -- optimized TTCalX over all 48 snapshots of one date (one daemon on GPU 0, so timings are clean)
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
case $1 in 1220) D=$PX/data;; 1227) D=$PX/data_1227;; esac
ST=$PX/runs/full_$1
S=$(ls $D | grep MHz_ | sort -t M -k1,1n)
for v in fast_sky fast_after fast_joint; do
  GPU=0 STUDY=$ST DATA=$D WSC_THREADS=12 $PX_SCRIPTS/run_variant.sh $v $S > $ST/logs/${v}.log 2>&1
  echo "FAST96 $1 $v $(grep -c ^ZESTED $ST/logs/${v}.log) zested, $(grep -c FAILED $ST/logs/${v}.log) failed, $(date)"
done
echo FAST96_DONE $1
