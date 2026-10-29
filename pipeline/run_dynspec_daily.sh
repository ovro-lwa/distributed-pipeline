#!/usr/bin/env bash
# Run run_dynspec_slow_when_ready.sh for NDAYS consecutive UTC dates starting at START_DATE.
#
# Usage (as pipeline, inside screen):
#   NDAYS=30 screen -dmS dynspec-daily pipeline/run_dynspec_daily.sh 2026-10-21
#
# Optional env: NDAYS=30, plus everything run_dynspec_slow_when_ready.sh takes.  By default the
# map tasks go to the shared nightavg queue (calim06 07 09 balance it among themselves;
# the night-time averaging only uses it from 14:30 UTC) and the reducer to calim06.
# Logs: deploy/logs/dynspec_slow_<DATE>.log
set -uo pipefail

START_DATE=${1:?usage: $0 YYYY-MM-DD}
NDAYS=${NDAYS:-30}
export QUEUES=${QUEUES:-"nightavg"}
export REDUCE_QUEUE=${REDUCE_QUEUE:-"calim06"}
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python
CONSUMERS=${CONSUMERS:-"calim06@lwacalim06,calim07@lwacalim07,calim09@lwacalim09"}
cd "$REPO" || exit 1

for (( n = 0; n < NDAYS; n++ )); do
    date_str=$(date -u -d "${START_DATE} + ${n} day" +%F)
    # Re-add the shared queue on workers with room on /fast (see nightavg_consumers.sh).
    if [[ $QUEUES == nightavg* ]]; then
        sleep $(( $(date -u -d "${date_str} 10:00:00 UTC" +%s) - $(date -u +%s) )) 2>/dev/null
        pipeline/nightavg_consumers.sh "$CONSUMERS" >> "${REPO}/deploy/logs/dynspec_slow_${date_str}.log" 2>&1
    fi
    "${REPO}/pipeline/run_dynspec_slow_when_ready.sh" "$date_str"
done
