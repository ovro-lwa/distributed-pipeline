#!/usr/bin/env bash
# Prepare snapshots of any night under DATA_DIR/<band>MHz_<HHMM>/.
#   prep_date.sh DATE BP_TABLE DATA_DIR "BANDS" "HOURS" [JOBS]
# For each band and hour the first averaged snapshot is used; a missing hour
# falls back to the latest earlier hour available for that band.
set -eo pipefail
source "$(dirname "$0")/env.sh"
DATE=$1; BP=$2; DATA_DIR=$3; BANDS=$4; HOURS=$5; JOBS=${6:-4}
ROOT=/lustre/pipeline/night-time/averaged
export BP PX_DATA=$DATA_DIR
for band in $BANDS; do
    for hour in $HOURS; do
        h=$hour
        while [ ! -d "$ROOT/${band}MHz/$DATE/$h" ] && [ "$h" -gt 0 ]; do h=$(printf %02d $((10#$h - 1))); done
        all=( "$ROOT/${band}MHz/$DATE/$h"/*_averaged.ms )
        [ -e "${all[0]}" ] || continue
        stamp=$(basename "${all[0]}" | cut -d_ -f2)
        echo "${all[0]} $DATA_DIR/${band}MHz_${stamp:0:4}"
    done
done | xargs -r -P "$JOBS" -n 2 "$PX_SCRIPTS/prep_one.sh"
