#!/usr/bin/env bash
# Average every night's slow-buffer data into night-time/averaged, one UTC date after another.
#
# For each date (starting at START_DATE): at EARLY_UTC submit the LST hours that ended by
# EARLY_UNTIL (already on Lustre), at READY_UTC, by which time the whole night (ends ~13:30 UTC in
# Nov) is on Lustre, submit the rest, then a full pass re-submits anything that failed (done files
# are skipped).  The slow buffer only keeps ~24 h; starting at 14:30 with everything was too late
# on 2026-11-09 (23 h run, last ~2 h deleted before they were averaged).
#
# Usage (as pipeline, inside screen):
#   screen -dmS nightavg-daily pipeline/run_nightavg_daily.sh 2026-10-18
#
# Optional env: QUEUES="nightavg" (or per-node "calim09:20 ...")  READY_UTC="14:30"  NDAYS=0 (0 = forever)  MAX_MISSING=100
#               EARLY_UTC="08:30" EARLY_UNTIL="08:15"  (EARLY_UTC="" disables the early pass)
#               ALT_ROOTS="/lustre/pipeline/cosmology"  other roots to look in when files are missing
#               CONSUMERS="calim06@lwacalim06,..."  workers that must consume the nightavg queue
#
# Before each date the previous date's log should report at most MAX_MISSING (default 100) missing
# outputs, or nightavg_<prev>.ok must exist (accepted by hand); otherwise nightavg_<prev>.incomplete
# is written and the loop goes on (the slow buffer does not wait).
# Logs: deploy/logs/nightavg_<DATE>.log
set -uo pipefail

START_DATE=${1:?usage: $0 YYYY-MM-DD}
QUEUES=${QUEUES:-"nightavg"}   # shared queue on calim06 07 09 (see deploy/manage-workers.sh)
READY_UTC=${READY_UTC:-"14:30"}
EARLY_UTC=${EARLY_UTC-"08:30"}
EARLY_UNTIL=${EARLY_UNTIL:-"08:15"}
NDAYS=${NDAYS:-0}
MAX_MISSING=${MAX_MISSING:-100}
ALT_ROOTS=${ALT_ROOTS:-"/lustre/pipeline/cosmology"}   # where slow hours have been moved to   # previous night may miss up to this many outputs
CONSUMERS=${CONSUMERS:-"calim06@lwacalim06,calim07@lwacalim07,calim09@lwacalim09"}
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
    start_at=$ready_at
    [[ -n "$EARLY_UTC" ]] && start_at=$(date -u -d "${date_str} ${EARLY_UTC}:00 UTC" +%s)
    log "Waiting for ${date_str} (not before $(date -u -d @${start_at} +%F' '%T) UTC)"
    while (( $(date -u +%s) < start_at )); do sleep 300; done

    prev=$(date -u -d "${date_str} - 1 day" +%F)
    prev_log=${REPO}/deploy/logs/nightavg_${prev}.log
    prev_missing=$(grep -o "Outputs missing after run: [0-9]*" "$prev_log" 2>/dev/null | tail -1 | awk '{print $NF}')
    # A night with no slow data at all (recorders down, seen 2026-11-01) has nothing missing.
    if [[ -z "$prev_missing" ]] && grep -q ": 0 files in window" "$prev_log" 2>/dev/null; then
        prev_missing=0
    fi
    if [[ -e "${prev_log%.log}.ok" ]]; then
        log "Previous night ${prev} accepted by hand (${prev_log%.log}.ok)"
    elif [[ -z "$prev_missing" ]] || (( prev_missing > MAX_MISSING )); then
        # Stopping here lost 2026-11-10 and 11-11 entirely; flag the night and carry on.
        log "WARNING: previous night ${prev} not complete (missing: ${prev_missing:-unknown}, see nightavg_${prev}.log); continuing"
        echo "missing: ${prev_missing:-unknown}" > "${prev_log%.log}.incomplete"
    else
        log "Previous night ${prev}: ${prev_missing} outputs missing (<= ${MAX_MISSING}), continuing"
    fi

    # Re-add the shared queue (workers restarted outside manage-workers.sh lose it), but only on
    # workers with room on /fast.
    until pipeline/nightavg_consumers.sh "$CONSUMERS" >> "$LOG" 2>&1; do
        log "No worker has enough free /fast; retrying in 10 min"
        sleep 600
    done

    early_pid=""
    after=()
    if [[ -n "$EARLY_UTC" ]]; then
        log "Early pass (LST hours ending by ${EARLY_UNTIL} UTC), log ${LOG%.log}_early.log"
        $PY -W ignore -u pipeline/average_nighttime_slow.py --date "$date_str" --until "$EARLY_UNTIL" \
            --queues $QUEUES >> "${LOG%.log}_early.log" 2>&1 &
        early_pid=$!
        after=(--after "$EARLY_UNTIL")
    fi
    while (( $(date -u +%s) < ready_at )); do sleep 300; done
    log "Main pass: average_nighttime_slow.py ${after[*]} --queues ${QUEUES}"
    $PY -W ignore -u pipeline/average_nighttime_slow.py --date "$date_str" "${after[@]}" --queues $QUEUES >> "$LOG" 2>&1
    log "Main pass exited with $?"
    if [[ -n "$early_pid" ]]; then
        wait "$early_pid"
        log "Early pass exited with $?"
    fi
    # Full window: re-submits whatever failed and reports what is still missing (read by the gate).
    for pass in 1 2; do
        log "Full pass ${pass}: average_nighttime_slow.py --queues ${QUEUES}"
        $PY -W ignore -u pipeline/average_nighttime_slow.py --date "$date_str" --queues $QUEUES >> "$LOG" 2>&1
        rc=$?
        log "Full pass ${pass} exited with ${rc}"
        (( rc == 0 )) && break
    done
    # Hours that someone moved out of the slow buffer mid-run (seen 2026-10-22: hours 11-13 moved
    # to /lustre/pipeline/cosmology) are averaged from where they went.
    if (( rc != 0 )); then
        for alt in $ALT_ROOTS; do
            log "Pass alt: average_nighttime_slow.py --root ${alt}"
            $PY -W ignore -u pipeline/average_nighttime_slow.py --date "$date_str" --root "$alt" \
                --queues $QUEUES >> "$LOG" 2>&1
            log "Pass alt exited with $?"
        done
    fi

    date_str=$(date -u -d "${date_str} + 1 day" +%F)
    n=$((n + 1))
done
