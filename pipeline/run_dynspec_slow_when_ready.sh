#!/usr/bin/env bash
# Wait until a night's slow-buffer data is on Lustre, then run produce_dynspec_slow.py.
#
# Usage (as pipeline, inside screen so it survives logout):
#   screen -dmS dynspec-1014 pipeline/run_dynspec_slow_when_ready.sh 2026-10-14
#
# Optional env: HOURS="08 09"  QUEUES="calim00:40 calim06:8"  MIN_FILES=350
#               REDUCE_QUEUE (default: first QUEUES entry)  WAIT_H=1
# If some subband/hour is still short WAIT_H hours after the data should be complete (e.g. a
# subband that was not recorded), run anyway with what is there; missing parts stay zero.
# Log: deploy/logs/dynspec_slow_<DATE>.log
set -uo pipefail

DATE=${1:?usage: $0 YYYY-MM-DD}
HOURS=${HOURS:-"08 09"}
QUEUES=${QUEUES:-"calim00:40 calim06:8"}
MIN_FILES=${MIN_FILES:-350}          # ~359 integrations per hour per subband
ROOT=/lustre/pipeline/slow
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python
FREQS="13 18 23 27 32 36 41 46 50 55 59 64 69 73 78 82"
LOG=${REPO}/deploy/logs/dynspec_slow_${DATE}.log

cd "$REPO" || exit 1
log() { echo "[$(date -u +%F' '%T)] $*" | tee -a "$LOG"; }

last_hour=$(echo $HOURS | awk '{print $NF}')
ready_at=$(( $(date -u -d "${DATE} ${last_hour}:00:00 UTC" +%s) + 3600 + 600 ))
WAIT_H=${WAIT_H:-1}
deadline=$((ready_at + WAIT_H * 3600))
log "Waiting for ${DATE} hours ${HOURS} (not before $(date -u -d @${ready_at} +%F' '%T) UTC)"

while :; do
    now=$(date -u +%s)
    if (( now >= ready_at )); then
        short=""
        for f in $FREQS; do
            for h in $HOURS; do
                n=$(ls ${ROOT}/${f}MHz/${DATE}/${h} 2>/dev/null | grep -c '\.ms\.tar$')
                (( n < MIN_FILES )) && short+=" ${f}MHz/${h}:${n}"
            done
        done
        if [[ -z "$short" ]]; then
            break
        fi
        if (( now >= deadline )); then
            log "Still short after ${WAIT_H} h; running with what is there:${short}"
            break
        fi
        log "Not complete yet:${short}"
    fi
    sleep 300
done

log "Data complete; starting produce_dynspec_slow.py --queues ${QUEUES}"
export PYTHONPATH=$REPO OPENBLAS_NUM_THREADS=1
$PY -u pipeline/produce_dynspec_slow.py --date "$DATE" --hours $HOURS --queues $QUEUES \
    ${REDUCE_QUEUE:+--reduce-queue $REDUCE_QUEUE} >> "$LOG" 2>&1
log "produce_dynspec_slow.py exited with $?"
