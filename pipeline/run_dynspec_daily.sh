#!/usr/bin/env bash
# Run run_dynspec_slow_when_ready.sh for NDAYS consecutive UTC dates starting at START_DATE.
#
# Usage (as pipeline, inside screen):
#   NDAYS=30 screen -dmS dynspec-daily pipeline/run_dynspec_daily.sh 2026-10-21
#
# Optional env: NDAYS=30, plus everything run_dynspec_slow_when_ready.sh takes.  By default the
# map tasks go to the shared nightavg queue (calim00 06 07 09 10 balance it among themselves;
# the night-time averaging only uses it from 14:30 UTC) and the reducer to calim00.
# Logs: deploy/logs/dynspec_slow_<DATE>.log
set -uo pipefail

START_DATE=${1:?usage: $0 YYYY-MM-DD}
NDAYS=${NDAYS:-30}
export QUEUES=${QUEUES:-"nightavg"}
export REDUCE_QUEUE=${REDUCE_QUEUE:-"calim00"}
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python
CONSUMERS=${CONSUMERS:-"calim00@lwacalim00,calim06@lwacalim06,calim07@lwacalim07,calim09@lwacalim09,calim10@lwacalim10"}
cd "$REPO" || exit 1

for (( n = 0; n < NDAYS; n++ )); do
    date_str=$(date -u -d "${START_DATE} + ${n} day" +%F)
    # Workers restarted outside manage-workers.sh lose the shared queue; re-add it (no-op otherwise).
    if [[ $QUEUES == nightavg* ]]; then
        sleep $(( $(date -u -d "${date_str} 10:00:00 UTC" +%s) - $(date -u +%s) )) 2>/dev/null
        PYTHONPATH=$REPO $PY -m celery -A orca.celery control add_consumer nightavg -d "$CONSUMERS" \
            >> "${REPO}/deploy/logs/dynspec_slow_${date_str}.log" 2>&1
    fi
    "${REPO}/pipeline/run_dynspec_slow_when_ready.sh" "$date_str"
done
