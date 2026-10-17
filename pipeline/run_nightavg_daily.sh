#!/usr/bin/env bash
# Average every night's slow-buffer data into night-time/averaged, one UTC date after another.
#
# For each date (starting at START_DATE) wait until READY_UTC on that date, by which time the
# whole-LST-hour night window (ends <= ~13:10 UTC in Oct, later in winter) is on Lustre, then run
# average_nighttime_slow.py.  A second pass re-submits anything that failed (done files are
# skipped).  The slow buffer only keeps ~24 h, so the oldest data must finish by ~02-03 UTC the
# next day.
#
# Usage (as pipeline, inside screen):
#   screen -dmS nightavg-daily pipeline/run_nightavg_daily.sh 2026-10-18
#
# Optional env: QUEUES="nightavg" (or per-node "calim09:20 ...")  READY_UTC="14:30"  NDAYS=0 (0 = forever)
#               CONSUMERS="calim00@lwacalim00,..."  workers that must consume the nightavg queue
#
# Before each date the previous date's log must say "Outputs missing after run: 0"; otherwise the
# loop stops so a broken night is looked at before more are submitted.
# Logs: deploy/logs/nightavg_<DATE>.log
set -uo pipefail

START_DATE=${1:?usage: $0 YYYY-MM-DD}
QUEUES=${QUEUES:-"nightavg"}   # shared queue on calim00 06 07 09 10 (see deploy/manage-workers.sh)
READY_UTC=${READY_UTC:-"14:30"}
NDAYS=${NDAYS:-0}
CONSUMERS=${CONSUMERS:-"calim00@lwacalim00,calim06@lwacalim06,calim07@lwacalim07,calim09@lwacalim09,calim10@lwacalim10"}
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python

cd "$REPO" || exit 1
export PYTHONPATH=$REPO OPENBLAS_NUM_THREADS=1

date_str=$START_DATE
n=0
while (( NDAYS == 0 || n < NDAYS )); do
    LOG=${REPO}/deploy/logs/nightavg_${date_str}.log
    log() { echo "[$(date -u +%F' '%T)] $*" | tee -a "$LOG"; }

    ready_at=$(date -u -d "${date_str} ${READY_UTC}:00 UTC" +%s)
    log "Waiting for ${date_str} (not before $(date -u -d @${ready_at} +%F' '%T) UTC)"
    while (( $(date -u +%s) < ready_at )); do sleep 300; done

    prev=$(date -u -d "${date_str} - 1 day" +%F)
    if ! grep -q "Outputs missing after run: 0" "${REPO}/deploy/logs/nightavg_${prev}.log" 2>/dev/null; then
        log "Previous night ${prev} not complete (see nightavg_${prev}.log); stopping"
        exit 1
    fi

    # Workers restarted outside manage-workers.sh (e.g. calim10) lose the shared queue; re-add it.
    $PY -m celery -A orca.celery control add_consumer nightavg -d "$CONSUMERS" >> "$LOG" 2>&1

    for pass in 1 2; do
        log "Pass ${pass}: average_nighttime_slow.py --queues ${QUEUES}"
        $PY -W ignore -u pipeline/average_nighttime_slow.py --date "$date_str" --queues $QUEUES >> "$LOG" 2>&1
        rc=$?
        log "Pass ${pass} exited with ${rc}"
        (( rc == 0 )) && break
    done

    date_str=$(date -u -d "${date_str} + 1 day" +%F)
    n=$((n + 1))
done
