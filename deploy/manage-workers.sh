#!/usr/bin/env bash
# deploy/manage-workers.sh — Manage Celery subband workers across calim nodes.
# No sudo required — uses nohup + PID files instead of systemd.
#
# Usage:
#   ./manage-workers.sh status                  # show all worker statuses
#   ./manage-workers.sh start                   # start workers on all available nodes
#   ./manage-workers.sh stop                    # graceful stop all workers
#   ./manage-workers.sh restart                 # stop + start (e.g. after git pull)
#   ./manage-workers.sh deploy                  # git pull + restart on all nodes
#   ./manage-workers.sh logs calim08            # tail logs for one node
#   ./manage-workers.sh start calim08           # start just one node
#
set -euo pipefail

# --- Configuration ---
# Nodes that are currently available (edit this list as needed)
AVAILABLE_NODES=(calim01 calim05 calim06 calim07 calim08 calim09 calim10)

REPO_DIR="/opt/devel/nkosogor/nkosogor/distributed-pipeline"
CONDA_ACTIVATE="/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/activate"
CONCURRENCY=45
LOGLEVEL="INFO"

# PID and log file locations (no sudo needed — user-writable dirs)
PID_DIR="${REPO_DIR}/deploy/pids"
LOG_DIR="${REPO_DIR}/deploy/logs"

# Map short name → full hostname
declare -A HOST_MAP
for node in "${AVAILABLE_NODES[@]}"; do
    HOST_MAP[$node]="lwa${node}"
done

# --- Helpers ---
ssh_cmd() {
    local node=$1; shift
    ssh -o ConnectTimeout=5 -o BatchMode=yes "${HOST_MAP[$node]}" "$@"
}

get_nodes() {
    if [[ -n "${2:-}" ]]; then
        echo "$2"
    else
        echo "${AVAILABLE_NODES[@]}"
    fi
}

# --- Commands ---
cmd_status() {
    printf "%-12s %-10s %-8s\n" "NODE" "STATUS" "PID"
    printf "%-12s %-10s %-8s\n" "----" "------" "---"
    for node in $(get_nodes "$@"); do
        local pidfile="${PID_DIR}/${node}.pid"
        local status="stopped"
        local pid="-"
        if ssh_cmd "$node" "test -f ${pidfile}" 2>/dev/null; then
            pid=$(ssh_cmd "$node" "cat ${pidfile}" 2>/dev/null || echo "-")
            if ssh_cmd "$node" "kill -0 ${pid} 2>/dev/null"; then
                status="running"
            else
                status="dead"
            fi
        fi
        printf "%-12s %-10s %-8s\n" "$node" "$status" "$pid"
    done
}

cmd_start() {
    for node in $(get_nodes "$@"); do
        # Check if already running
        local pidfile="${PID_DIR}/${node}.pid"
        if ssh_cmd "$node" "test -f ${pidfile}" 2>/dev/null; then
            local pid
            pid=$(ssh_cmd "$node" "cat ${pidfile}" 2>/dev/null || echo "")
            if [[ -n "$pid" ]] && ssh_cmd "$node" "kill -0 ${pid} 2>/dev/null"; then
                echo "  ⚠ ${node} already running (PID ${pid}), skipping"
                continue
            fi
        fi

        echo "Starting worker on ${node}..."
        ssh_cmd "$node" "mkdir -p ${PID_DIR} ${LOG_DIR}"
        ssh_cmd "$node" bash <<EOF
            source ${CONDA_ACTIVATE}
            cd ${REPO_DIR}
            export OPENBLAS_NUM_THREADS=1
            nohup celery -A orca.celery worker \\
                -Q ${node} \\
                --hostname=${node}@\$(hostname) \\
                -c ${CONCURRENCY} \\
                --loglevel=${LOGLEVEL} \\
                --without-heartbeat \\
                --without-mingle \\
                --pidfile=${PID_DIR}/${node}.pid \\
                >> ${LOG_DIR}/${node}.log 2>&1 &
            disown
EOF
        sleep 1
        echo "  ✓ ${node} started"
    done
}

cmd_stop() {
    for node in $(get_nodes "$@"); do
        local pidfile="${PID_DIR}/${node}.pid"
        if ! ssh_cmd "$node" "test -f ${pidfile}" 2>/dev/null; then
            echo "  ${node}: not running (no PID file)"
            continue
        fi

        local pid
        pid=$(ssh_cmd "$node" "cat ${pidfile}" 2>/dev/null || echo "")
        if [[ -z "$pid" ]]; then
            echo "  ${node}: empty PID file"
            continue
        fi

        echo "Stopping worker on ${node} (PID ${pid})..."
        # Send SIGTERM for graceful shutdown (finish current task)
        ssh_cmd "$node" "kill -TERM ${pid} 2>/dev/null" || true

        # Wait up to 60s for graceful stop, then force
        local waited=0
        while ssh_cmd "$node" "kill -0 ${pid} 2>/dev/null" && [[ $waited -lt 60 ]]; do
            sleep 2
            waited=$((waited + 2))
            printf "  waiting... (%ds)\r" "$waited"
        done

        if ssh_cmd "$node" "kill -0 ${pid} 2>/dev/null"; then
            echo "  ⚠ ${node}: still alive after 60s, sending SIGKILL"
            ssh_cmd "$node" "kill -9 ${pid} 2>/dev/null" || true
        fi

        ssh_cmd "$node" "rm -f ${pidfile}" || true
        echo "  ✓ ${node} stopped"
    done
}

cmd_restart() {
    cmd_stop "$@"
    sleep 2
    cmd_start "$@"
}

cmd_deploy() {
    echo "=== Deploying code + restarting workers ==="
    for node in $(get_nodes "$@"); do
        echo ""
        echo "--- ${node} ---"
        echo "  Pulling latest code..."
        ssh_cmd "$node" "cd ${REPO_DIR} && git pull"
    done

    echo ""
    echo "=== Restarting workers ==="
    cmd_restart "$@"

    echo ""
    echo "=== Deploy complete ==="
    cmd_status
}

cmd_logs() {
    local node="${2:?Usage: $0 logs <node>}"
    local logfile="${LOG_DIR}/${node}.log"
    echo "=== Logs for ${node} (Ctrl+C to stop) ==="
    ssh_cmd "$node" "tail -f -n 200 ${logfile}"
}

cmd_clean_logs() {
    for node in $(get_nodes "$@"); do
        ssh_cmd "$node" "truncate -s 0 ${LOG_DIR}/${node}.log 2>/dev/null" || true
        echo "  ✓ ${node} logs cleared"
    done
}

# --- Main ---
case "${1:-help}" in
    status)     cmd_status "$@" ;;
    start)      cmd_start "$@" ;;
    stop)       cmd_stop "$@" ;;
    restart)    cmd_restart "$@" ;;
    deploy)     cmd_deploy "$@" ;;
    logs)       cmd_logs "$@" ;;
    clean-logs) cmd_clean_logs "$@" ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|deploy|logs|clean-logs} [node]"
        echo ""
        echo "Commands:"
        echo "  status     [node]  — Show worker status on all/one node(s)"
        echo "  start      [node]  — Start worker(s) in background"
        echo "  stop       [node]  — Gracefully stop worker(s)"
        echo "  restart    [node]  — Stop + start worker(s)"
        echo "  deploy     [node]  — git pull + restart on all/one node(s)"
        echo "  logs       <node>  — Tail log file for one node"
        echo "  clean-logs [node]  — Truncate log files"
        echo ""
        echo "Available nodes: ${AVAILABLE_NODES[*]}"
        echo ""
        echo "PID files: ${PID_DIR}/"
        echo "Log files: ${LOG_DIR}/"
        exit 1
        ;;
esac
