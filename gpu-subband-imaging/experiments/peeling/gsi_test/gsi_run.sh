#!/usr/bin/env bash
# Worker runs of each test config on GPU 0 of this node, one at a time.
T=/lustre/nkosogor/peel_exp/gsi_test
export GSI_ALLOWED_LUSTRE_WRITE_ROOT=$T
W=/lustre/nkosogor/peel_exp/gsi_branch/workers/run_worker.sh
run() { # variant band
  local t0=$(date +%s.%N)
  bash $W 0 $T/config-$1 $2 0 $T/manifest_$2.txt > $T/run_$1_$2.log 2>&1
  echo "GSIRUN $1 $2 rc=$? wall=$(echo "$(date +%s.%N)-$t0" | bc)"
}
for v in prod prod_branch cpumatch; do for b in 32 41 69; do run $v $b; done; done
run cpumatch_cpu 32
echo GSIRUN_DONE
