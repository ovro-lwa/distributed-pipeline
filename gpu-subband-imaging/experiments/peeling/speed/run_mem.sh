#!/usr/bin/env bash
# run_mem.sh TAG THREADS DEVICE SNAPS...   (outputs in /tmp/nk_peel/out)
source /lustre/nkosogor/peel_exp/repo/experiments/peeling/env.sh
TAG=$1; T=$2; DEV=$3; shift 3
MS=$(for s in "$@"; do p=( $PX/data/$s/*_prep.ms ); echo -n "${p[0]},"; done); MS=${MS%,}
export ZEST_RFI_SOURCES=$RFI_MODEL ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1 ZEST_UVW_CONVENTION=geom
export ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5 ZEST_MIN_SOURCE_ELEVATION_DEG=0
export ZEST_REQUIRE_CONVERGENCE=false ZEST_GENVIS=exact ZEST_DEVICE=$DEV ZEST_PEELITER=3 ZEST_MAX_GAIN_AMPLITUDE=100
export ZEST_COLUMN=DATA CUDA_VISIBLE_DEVICES=${GPU:-0}
export TTCALX_DIR=${TTCALX:-$PX/ttcalx_fast} DAEMON=$PX/repo_fast/workers/zest_daemon.jl
export MEM_MS=$MS MEM_OUT=/tmp/nk_peel/out MEM_TAG=$TAG
$JULIA_BIN/julia -t $T --project=$TTCALX_DIR /tmp/nk_peel/mem_zest.jl $SKY_MODEL < /dev/null 2>&1 | grep -E "MEMTIME|MEM_DONE|ERROR|Error|FAILED" 
