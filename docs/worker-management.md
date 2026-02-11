# Worker Management Guide

## Quick Reference

```bash
# All commands run from the repo on any calim node (e.g. calim10):
cd /opt/devel/nkosogor/nkosogor/distributed-pipeline
```

| Action | Command |
|---|---|
| Start all workers | `./deploy/manage-workers.sh start` |
| Stop all workers | `./deploy/manage-workers.sh stop` |
| Check status | `./deploy/manage-workers.sh status` |
| **Code change → deploy** | `./deploy/manage-workers.sh deploy` |
| Start/stop one node | `./deploy/manage-workers.sh start calim08` |
| Tail logs | `./deploy/manage-workers.sh logs calim08` |
| Clear log files | `./deploy/manage-workers.sh clean-logs` |

## After Code Changes

```bash
# 1. Push what you've developed
git push

# 2. SSH to any calim node and run ONE command:
./deploy/manage-workers.sh deploy
```

This does `git pull` + restart on all 7 nodes automatically.

## Adding/Removing Nodes

Edit `AVAILABLE_NODES` in `deploy/manage-workers.sh`:

```bash
AVAILABLE_NODES=(calim01 calim05 calim06 calim07 calim08 calim09 calim10)
```

## Submitting Jobs

```bash
# Dry run (verify file discovery):
python pipeline/subband_celery.py \
  --range 04-05 --date 2026-01-31 \
  --bp_table /path/to/bandpass.B.flagged \
  --xy_table /path/to/xyphase.Xf \
  --subbands 73MHz 78MHz \
  --peel_sky --peel_rfi --dry_run

# Real run (with NVMe cleanup after archiving):
python pipeline/subband_celery.py \
  --range 04-05 --date 2026-01-31 \
  --bp_table /path/to/bandpass.B.flagged \
  --xy_table /path/to/xyphase.Xf \
  --subbands 73MHz 78MHz \
  --peel_sky --peel_rfi --cleanup_nvme

# Remap subbands to different nodes:
--remap 18MHz=calim01 23MHz=calim05
```

## Monitoring

- **Flower**: `http://localhost:5555` (SSH tunnel: `ssh -L 5555:localhost:5555 lwacalim10`)
- **Worker logs**: `./deploy/manage-workers.sh logs calim08`
- **Log files on disk**: `deploy/logs/calim08.log`
