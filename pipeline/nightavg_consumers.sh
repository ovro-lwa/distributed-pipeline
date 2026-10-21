#!/usr/bin/env bash
# Point the shared nightavg queue at the workers in $1 (comma-separated celery names like
# calim00@lwacalim00) that have at least MIN_FREE_GB free on /fast; take it away from the others.
# A worker with a full /fast fails every task in seconds and would swallow the whole queue.
set -uo pipefail
CONSUMERS=${1:?usage: $0 calim00@lwacalim00,...}
MIN_FREE_GB=${MIN_FREE_GB:-100}
REPO=/opt/devel/nkosogor/nkosogor/distributed-pipeline
PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python

ok=() low=()
for w in ${CONSUMERS//,/ }; do
    host=${w#*@}
    free=$(timeout 60 ssh -x -o LogLevel=ERROR -o BatchMode=yes "$host" "df -BG --output=avail /fast | tail -1" 2>/dev/null | tr -dc 0-9)
    if [[ -n "$free" ]] && (( free >= MIN_FREE_GB )); then ok+=("$w"); else low+=("$w:${free:-?}G"); fi
done
echo "[$(date -u +%F' '%T)] nightavg consumers: ${ok[*]:-none}; left out (< ${MIN_FREE_GB}G free on /fast): ${low[*]:-none}"
cd "$REPO" && export PYTHONPATH=$REPO
(( ${#ok[@]} )) && $PY -m celery -A orca.celery control add_consumer nightavg -d "$(IFS=,; echo "${ok[*]}")"
for w in "${low[@]}"; do
    $PY -m celery -A orca.celery control cancel_consumer nightavg -d "${w%:*}"
done
(( ${#ok[@]} ))
