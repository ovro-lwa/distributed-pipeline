#!/bin/bash
# Daily nights on a gsi deploy from <first date>: each night through night_guarded.sh (main table if it passes against
# the last accepted table, else the backup, else the night is skipped); the accepted table becomes the next reference.
# usage: [G=/opt/devel/pipeline/gsi-v5] daily_guarded.sh <first YYYY-MM-DD>
# gsi-v4's /opt/devel/pipeline/gsi-v4/daily_guarded_v4.sh with the deploy as a variable (see night_guarded.sh).
# Only ONE daily loop may run (v4 or v5): they share state_dir and output_root.
export G=${G:-/opt/devel/pipeline/gsi-v5}
TAG=$(basename $G)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$1
REFF=$G/last_good_bp.txt
[ -s $REFF ] || echo "WARNING: no $REFF (copy the previous deploy's): the first night checks against the built-in reference"
while true; do
    [ -s $REFF ] && export REF_TABLE=$(cat $REFF)
    bash $HERE/night_guarded.sh $D
    BPT=$(grep -o "starting night $D on $TAG, run label [^ ]*, bandpass [^ ]*" $G/night_guard_${D}.log | tail -1 | awk '{print $NF}')
    [ -n "$BPT" ] && echo $BPT > $REFF
    D=$(date -u -d "$D + 1 day" +%Y-%m-%d)
    # the next date's calibration finishes around 07-08 UTC; night_guarded.sh waits for it
done
