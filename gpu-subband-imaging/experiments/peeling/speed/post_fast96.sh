#!/usr/bin/env bash
# summaries of the fast_* variants (8 parallel chunks per date) + comparison with the earlier exact runs
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
V="fast_sky fast_after fast_joint"
for d in full_1220 full_1227; do
  mkdir -p $PX/runs/$d/parts
  set -- $(ls $PX/runs/$d | grep MHz_ | sort -t M -k1,1n); n=$#; per=$(( (n + 7) / 8 )); i=0
  while [ $# -gt 0 ]; do chunk=("${@:1:per}"); shift $(( $# < per ? $# : per ))
    px_py $PX_SCRIPTS/study_summary.py $PX/runs/$d --vis --only $V --snaps "${chunk[@]}" --out $PX/runs/$d/parts/fast_part$i.json > $PX/runs/$d/parts/fast_part$i.log 2>&1 &
    i=$((i+1)); done
done
px_py $PX_SCRIPTS/compare_fast.py $PX/runs/full_1220 $PX/runs/full_1227 > $PX/runs/compare_fast.json 2> $PX/runs/compare_fast.err &
wait; echo POST_FAST96_DONE
