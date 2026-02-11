#!/usr/bin/env bash
# deploy/manage-workers.sh — Manage Celery subband workers across calim nodes.
#
# Usage:
#   ./manage-workers.sh status                  # show all worker statuses
#   ./manage-workers.sh start                   # start workers on all available nodes
#   ./manage-workers.sh stop                    # graceful stop all workers
#   ./manage-workers.sh restart                 # stop + start (e.g. after git pull)
#   ./manage-workers.sh deploy                  # git pull + restart on all nodes
#   ./manage-workers.sh install                 # install systemd unit on all nodes
#   ./manage-workers.sh logs calim08            # tail logs for one node
#   ./manage-workers.sh start calim08           # start just one node
#
set -euo pipefail

# --- Configuration ---
# Nodes that are currently available (edit this list as needed)
AVAILABLE_NODES=(calim01 calim05 calim06 calim07 calim08 calim09 calim10)

SERVICE_PREFIX="celery-subband-worker"
REPO_DIR="/opt/devel/nkosogor/nkosogor/distributed-pipeline"
SERVICE_SRC="${REPO_DIR}/deploy/celery-subband-worker@.service"
SERVICE_DEST="/etc/systemd/system/celery-subband-worker@.service"

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
        status=$(ssh_cmd "$node" "systemctl is-active ${SERVICE_PREFIX}@${node}.service 2>/dev/null" || echo "not-found")
        pid=$(ssh_cmd "$node" "systemctl show -p MainPID --value ${SERVICE_PREFIX}@${node}.service 2>/dev/null" || echo "-")
        printf "%-12s %-10s %-8s\n" "$node" "$status" "$pid"
    done
}

cmd_start() {
    for node in $(get_nodes "$@"); do
        echo "Starting worker on ${node}..."
        ssh_cmd "$node" "sudo systemctl start ${SERVICE_PREFIX}@${node}.service"
        echo "  ✓ ${node} started"
    done
}

cmd_stop() {
    for node in $(get_nodes "$@"); do
        echo "Stopping worker on ${node}..."
        ssh_cmd "$node" "sudo systemctl stop ${SERVICE_PREFIX}@${node}.service" || true
        echo "  ✓ ${node} stopped"
    done
}

cmd_restart() {
    for node in $(get_nodes "$@"); do
        echo "Restarting worker on ${node}..."
        ssh_cmd "$node" "sudo systemctl restart ${SERVICE_PREFIX}@${node}.service"
        echo "  ✓ ${node} restarted"
    done
}

cmd_deploy() {
    echo "=== Deploying code + restarting workers ==="
    for node in $(get_nodes "$@"); do
        echo ""
        echo "--- ${node} ---"
        echo "  Pulling latest code..."
        ssh_cmd "$node" "cd ${REPO_DIR} && git pull"
        echo "  Restarting worker..."
        ssh_cmd "$node" "sudo systemctl restart ${SERVICE_PREFIX}@${node}.service"
        echo "  ✓ ${node} deployed"
    done
    echo ""
    echo "=== Deploy complete ==="
    cmd_status
}

cmd_install() {
    echo "=== Installing systemd unit on nodes ==="
    for node in $(get_nodes "$@"); do
        echo "  ${node}: copying service file..."
        ssh_cmd "$node" "sudo cp ${SERVICE_SRC} ${SERVICE_DEST} && sudo systemctl daemon-reload"
        ssh_cmd "$node" "sudo systemctl enable ${SERVICE_PREFIX}@${node}.service"
        echo "  ✓ ${node} installed + enabled"
    done
}

cmd_logs() {
    local node="${2:?Usage: $0 logs <node>}"
    echo "=== Logs for ${node} (Ctrl+C to stop) ==="
    ssh_cmd "$node" "sudo journalctl -u ${SERVICE_PREFIX}@${node}.service -f --no-pager -n 100"
}

# --- Main ---
case "${1:-help}" in
    status)  cmd_status "$@" ;;
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    restart) cmd_restart "$@" ;;
    deploy)  cmd_deploy "$@" ;;
    install) cmd_install "$@" ;;
    logs)    cmd_logs "$@" ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|deploy|install|logs} [node]"
        echo ""
        echo "Commands:"
        echo "  status  [node]  — Show worker status on all/one node(s)"
        echo "  start   [node]  — Start worker(s)"
        echo "  stop    [node]  — Gracefully stop worker(s)"
        echo "  restart [node]  — Stop + start worker(s)"
        echo "  deploy  [node]  — git pull + restart on all/one node(s)"
        echo "  install [node]  — Install systemd unit file on node(s)"
        echo "  logs    <node>  — Tail journald logs for one node"
        echo ""
        echo "Available nodes: ${AVAILABLE_NODES[*]}"
        exit 1
        ;;
esac
