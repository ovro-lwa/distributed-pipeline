#!/usr/bin/env bash
# Run a TTCalX zest daemon over MS paths (in place, DATA column).
#   GPU=0 gpu_zest.sh TTCALX_DIR DAEMON_JL SOURCES MS [MS ...]
# Peel settings come from ZEST_* env vars, defaulting to the production
# GPU pipeline values (config-allsubbands-auto/pipeline.yaml).
set -eo pipefail
TTCALX=$1; DAEMON=$2; SOURCES=$3; shift 3
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TTCALX_DIR=$TTCALX
export ZEST_COLUMN=${ZEST_COLUMN:-DATA}
export ZEST_MAXITER=${ZEST_MAXITER:-30}
export ZEST_TOLERANCE=${ZEST_TOLERANCE:-1e-2}
export ZEST_MINUVW=${ZEST_MINUVW:-10.0}
export ZEST_PEELITER=${ZEST_PEELITER:-3}
export ZEST_MIN_SOURCE_ELEVATION_DEG=${ZEST_MIN_SOURCE_ELEVATION_DEG:-15.0}
export ZEST_REQUIRE_CONVERGENCE=${ZEST_REQUIRE_CONVERGENCE:-true}
export ZEST_MAX_GAIN_AMPLITUDE=${ZEST_MAX_GAIN_AMPLITUDE:-100.0}
env | grep '^ZEST_' | sort | tr '\n' ' '; echo
printf '%s\n' "$@" QUIT | "$JULIA_BIN/julia" --project="$TTCALX" "$DAEMON" "$SOURCES"
