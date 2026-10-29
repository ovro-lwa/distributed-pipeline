#!/usr/bin/env bash
# Every INTERVAL s, re-run nightavg_consumers.sh so a worker whose /fast fills up during a run
# (other users' data) is taken off the shared nightavg queue before it fails every task, and is
# put back once there is room again.
#
# Usage (as pipeline, inside screen):
#   screen -dmS nightavg-watchdog pipeline/nightavg_watchdog.sh
# Optional env: CONSUMERS (default calim06 07 09)  MIN_FREE_GB=50  INTERVAL=300
# Log: deploy/logs/nightavg_watchdog.log (only changes are logged)
set -uo pipefail
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
CONSUMERS=${CONSUMERS:-"calim06@lwacalim06,calim07@lwacalim07,calim09@lwacalim09"}
export MIN_FREE_GB=${MIN_FREE_GB:-50}
INTERVAL=${INTERVAL:-300}
LOG=${REPO}/deploy/logs/nightavg_watchdog.log
cd "$REPO" || exit 1

prev=""
while :; do
    out=$(pipeline/nightavg_consumers.sh "$CONSUMERS" 2>&1)
    state=$(echo "$out" | grep -o "left out.*" )
    if [[ "$state" != "$prev" ]]; then
        echo "$out" | head -1 >> "$LOG"
        prev=$state
    fi
    sleep "$INTERVAL"
done
