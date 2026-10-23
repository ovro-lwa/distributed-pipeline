#!/usr/bin/env bash
# Prepare the shared test set (2024-12-20) under $PX/data/<band>MHz_<HHMM>/.
#   prep_set.sh [JOBS]
set -eo pipefail
source "$(dirname "$0")/env.sh"
JOBS=${1:-4}
BP=/lustre/pipeline/calibration/results/2024-12-20/02h/successful/20251224_094002/tables/calibration_2024-12-20_02h.B.flagged
ROOT=/lustre/pipeline/night-time/averaged
export BP
# band hour: first snapshot of the hour
SET="18:03 23:03 27:03 32:03 36:03 41:03 55:03 73:03
     18:08 23:08 27:08 32:08 36:08 41:08 55:08 73:08
     18:12 23:12 27:12 32:12 36:12 41:12 55:12 73:12"
for item in $SET; do
    band=${item%:*}; hour=${item#*:}
    all=($ROOT/${band}MHz/2024-12-20/$hour/*_averaged.ms); src=${all[0]}
    stamp=$(basename "$src" | cut -d_ -f2)
    echo "$src $PX/data/${band}MHz_${stamp:0:4}"
done | xargs -r -P "$JOBS" -n 2 "$PX_SCRIPTS/prep_one.sh"
