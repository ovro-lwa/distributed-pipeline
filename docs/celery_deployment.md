# ORCA Celery Deployment Guide

This document covers the distributed task queue infrastructure for the ORCA pipeline.

## Table of Contents
- [Installation](#installation)
- [Cluster Architecture](#cluster-architecture)
- [Components Overview](#components-overview)
- [Configuration](#configuration)
- [Managing Workers](#managing-workers)
- [Monitoring](#monitoring)
- [Adding New Users](#adding-new-users)
- [Troubleshooting](#troubleshooting)

---

## Installation

See the main [README](../README.md) for environment setup and configuration.

**Verify Celery connectivity** (requires valid `~/orca-conf.yml`):

```bash
# Test broker connection
python -c "from orca.celery import app; print(app.control.ping(timeout=2))"
```

---

## Cluster Architecture

```
                              ┌─────────────────────────────────────────┐
                              │         Shared Storage (NFS)            │
                              │  /opt/devel/pipeline/envs/              │
                              │  /home/pipeline/                        │
                              │  /lustre/pipeline/                      │
                              └─────────────────────────────────────────┘
                                    │       │       │       │
          ┌─────────────────────────┼───────┼───────┼───────┼─────────────────────────┐
          │                         │       │       │       │                         │
          ▼                         ▼       ▼       ▼       ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────────────────────┐    ┌──────────────────┐
│   lwacalimhead   │    │              Worker Nodes                    │    │   lwacalim10     │
│   (10.41.0.74)   │    │  lwacalim00-09 (10.41.0.75-84)               │    │  (10.41.0.85)    │
│                  │    │                                              │    │                  │
│  ┌────────────┐  │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │  ┌────────────┐  │
│  │  RabbitMQ  │◀─┼────┼──│ Worker  │ │ Worker  │ │ Worker  │ ...    │────┼─▶│   Redis    │  │
│  │   :5672    │  │    │  │ default │ │ imaging │ │ bandpass│        │    │  │   :6379    │  │
│  └────────────┘  │    │  └─────────┘ └─────────┘ └─────────┘        │    │  └────────────┘  │
│                  │    │                                              │    │                  │
│                  │    │  Each node runs Celery workers that:        │    │  Stores:         │
│                  │    │  • Pull tasks from RabbitMQ                 │    │  • Task results  │
│                  │    │  • Execute pipeline functions               │    │  • Spectrum cache│
│                  │    │  • Push results to Redis                    │    │                  │
│                  │    │                                              │    │  ┌────────────┐  │
│                  │    │                                              │    │  │  Flower    │  │
│                  │    │                                              │    │  │   :5555    │  │
│                  │    │                                              │    │  └────────────┘  │
└──────────────────┘    └──────────────────────────────────────────────┘    └──────────────────┘
```

### Node Summary

| Hostname | IP | Role |
|----------|-----|------|
| lwacalimhead | 10.41.0.74 | RabbitMQ broker |
| lwacalim00-09 | 10.41.0.75-84 | Worker nodes |
| lwacalim10 | 10.41.0.85 | Redis backend, Flower, Worker node |

---

## Components Overview

### Message Flow

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Pipeline Script │      │    RabbitMQ     │      │  Celery Worker  │      │      Redis      │
│   (your code)   │      │ (message queue) │      │ (task executor) │      │ (result store)  │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │                        │
         │  1. task.delay()       │                        │                        │
         │  ──────────────────▶   │                        │                        │
         │                        │                        │                        │
         │                        │  2. deliver task       │                        │
         │                        │  ──────────────────▶   │                        │
         │                        │                        │                        │
         │                        │                        │  3. execute function   │
         │                        │                        │  ────────────────────  │
         │                        │                        │                        │
         │                        │                        │  4. store result       │
         │                        │                        │  ──────────────────▶   │
         │                        │                        │                        │
         │  5. result.get()       │                        │                        │
         │  ◀──────────────────────────────────────────────────────────────────────│
         │                        │                        │                        │
```

**Simplified view:**
```
Your Script ──▶ RabbitMQ ──▶ Worker ──▶ Redis ──▶ Your Script
              (queue task)  (execute)  (result)  (retrieve)
```

### Component Details

| Component | Location | Port | Purpose |
|-----------|----------|------|---------|
| **RabbitMQ** | lwacalimhead | 5672 | Message broker - queues tasks |
| **Redis** | lwacalim10 | 6379 | Result backend - stores task results |
| **Celery Workers** | All nodes | - | Execute tasks from queues |
| **Flower** | lwacalim10 | 5555 | Web UI for monitoring |

### Queues

Defined in `orca/celery.py`:

| Queue | Purpose | Routed Tasks |
|-------|---------|--------------|
| `default` | General processing | Most tasks |
| `imaging` | Imaging pipeline | `imaging_pipeline_task`, `imaging_shared_pipeline_task` |
| `bandpass` | Bandpass calibration | `bandpass_nvme_task` |
| `cosmology` | Cosmology processing | `split_2pol_task` |

---

## Configuration

### User Configuration File

Each user needs `~/orca-conf.yml`:

```yaml
queue:
  prefix: default
  broker_uri: pyamqp://<username>:<password>@rabbitmq.calim.mcs.pvt:5672/<vhost>
  result_backend_uri: redis://10.41.0.85:6379/0

telescope:
  n_ant: 352
  n_subband: 16
  n_chan: 192
  outriggers: [...]  # See default-orca-conf.yml

execs:
  wsclean: /opt/bin/wsclean
  aoflagger: /opt/bin/aoflagger

cluster: calim
```

### Shared Environment

The conda environment is shared via NFS:
```bash
# Activate the shared environment
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor

# Or add to your .condarc:
envs_dirs:
  - /opt/devel/<username>/envs
  - /opt/devel/pipeline/envs
```

---

## Managing Workers

### Starting Workers

```bash
# Basic worker on default queue
celery -A orca.celery worker \
    --hostname=default@$(hostname) \
    --loglevel=INFO \
    --concurrency=40 \
    -Q default

# With CPU pinning (recommended for production)
taskset -c 0-39 celery -A orca.celery worker \
    --hostname=default@$(hostname) \
    --loglevel=INFO \
    --concurrency=40 \
    -Q default

# Worker for specific queue
celery -A orca.celery worker \
    --hostname=imaging@$(hostname) \
    --loglevel=INFO \
    --concurrency=40 \
    -Q imaging
```

### Stopping Workers

```bash
# Graceful shutdown (waits for current tasks)
pkill -15 -f 'celery.*worker'

# Force kill (immediate)
pkill -9 -f 'celery.*worker'

# Across all nodes with pdsh
pdsh -w lwacalim[00-10] "pkill -15 -f 'celery.*worker'"
```

### Checking Worker Status

```bash
# From Python
python3 << 'EOF'
from orca.celery import app

# Ping all workers
print("Workers:", app.control.ping(timeout=2))

# Active queues
i = app.control.inspect()
print("Queues:", i.active_queues())

# Current tasks
print("Active:", i.active())
EOF
```

### Code Updates

When you update orca code, workers need to be restarted:

```bash
# 1. Stop workers gracefully
pdsh -w lwacalim[00-10] "pkill -15 -f 'celery.*worker'"

# 2. Wait for tasks to finish (or check Flower)
sleep 30

# 3. Pull code updates (if using git)
pdsh -w lwacalim[00-10] "cd /opt/devel/pipeline/distributed-pipeline && git pull"

# 4. Restart workers (in screen sessions on each node)
# Or use systemd if configured
```

---

## Monitoring

### Flower Web UI

Flower provides real-time monitoring of workers and tasks.

```bash
# Start Flower (on lwacalim10)
celery -A orca.celery flower --port=5555

# Access via SSH tunnel
ssh -L 5555:localhost:5555 <user>@lwacalim10
# Then open: http://localhost:5555
```

### Command-Line Monitoring

```bash
# Check Redis connectivity
python3 -c "
import redis
r = redis.Redis.from_url('redis://10.41.0.85:6379/0')
print('Redis PING:', r.ping())
print('Keys in DB:', r.dbsize())
"

# Check RabbitMQ queues (on lwacalimhead with sudo)
sudo rabbitmqctl list_queues -p <vhost>

# Check worker processes
pdsh -w lwacalim[00-10] "ps aux | grep 'celery.*worker' | grep -v grep | wc -l"
```

### Inspecting Tasks

```python
from orca.celery import app

# Get inspector
i = app.control.inspect()

# Active tasks (currently running)
i.active()

# Reserved tasks (fetched but not started)
i.reserved()

# Scheduled tasks (eta/countdown)
i.scheduled()

# Registered tasks
i.registered()

# Worker stats
i.stats()
```

---

## Adding New Users

### 1. RabbitMQ Setup

On lwacalimhead (requires sudo):

```bash
# Create vhost for the user
sudo rabbitmqctl add_vhost <username>

# Create user
sudo rabbitmqctl add_user <username> <password>

# Grant permissions on their vhost
sudo rabbitmqctl set_permissions -p <username> <username> ".*" ".*" ".*"

# Verify
sudo rabbitmqctl list_users
sudo rabbitmqctl list_vhosts
```

### 2. User Configuration

The new user creates `~/orca-conf.yml`:

```yaml
queue:
  prefix: default
  broker_uri: pyamqp://<username>:<password>@rabbitmq.calim.mcs.pvt:5672/<username>
  result_backend_uri: redis://10.41.0.85:6379/0  # Shared Redis is fine

telescope:
  n_ant: 352
  n_subband: 16
  n_chan: 192
  outriggers: [...]  # Copy from default-orca-conf.yml

execs:
  wsclean: /opt/bin/wsclean
  aoflagger: /opt/bin/aoflagger

cluster: calim
```

### 3. Environment Setup

```bash
# Create user's conda config (~/.condarc)
pkgs_dirs:
  - /opt/devel/<username>/cache/conda
envs_dirs:
  - /opt/devel/<username>/envs
  - /opt/devel/pipeline/envs
channels:
  - conda-forge
  - defaults

# Create directories
mkdir -p /opt/devel/<username>/cache/conda
mkdir -p /opt/devel/<username>/envs

# Activate shared environment
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
```

### 4. Verify Setup

```bash
# Test RabbitMQ connection
python3 -c "
from orca.celery import app
print(app.control.ping(timeout=2))
"

# Test Redis connection  
python3 -c "
import redis
from orca.configmanager import queue_config
r = redis.Redis.from_url(queue_config.result_backend_uri)
print('Redis:', r.ping())
"
```

---

## Troubleshooting

### Workers Not Responding

```bash
# Check if workers are running
ps aux | grep celery

# Check if broker is reachable
nc -zv rabbitmq.calim.mcs.pvt 5672

# Check if Redis is reachable
nc -zv 10.41.0.85 6379

# Try starting worker with debug logging
celery -A orca.celery worker --loglevel=DEBUG
```

### Import Errors on Worker Start

```bash
# Common issue: casacore library conflict
unset LD_LIBRARY_PATH

# Test imports manually
python3 -c "from orca.celery import app; print('OK')"
```

### Tasks Stuck in Queue

```bash
# List queues and message counts
sudo rabbitmqctl list_queues -p <vhost>

# Purge a queue (deletes all pending tasks!)
celery -A orca.celery purge -Q <queue_name>

# Delete a queue entirely
celery -A orca.celery amqp queue.delete <queue_name>
```

### Worker Memory Issues

Workers are configured to restart after 20 tasks (`worker_max_tasks_per_child=20` in `celery.py`) to prevent memory leaks.

```bash
# Check memory usage
pdsh -w lwacalim[00-10] "ps aux | grep 'celery.*worker' | awk '{sum+=\$6} END {print sum/1024 \" MB\"}'"
```

### Redis Full

```bash
# Check Redis memory
python3 -c "
import redis
r = redis.Redis.from_url('redis://10.41.0.85:6379/0')
info = r.info('memory')
print(f\"Used: {info['used_memory_human']}\")
print(f\"Peak: {info['used_memory_peak_human']}\")
"

# Results expire after 2 hours (result_expires=7200 in celery.py)
# Spectrum cache expires after 10 hours
```

---

## Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `~/orca-conf.yml` | User configuration (broker, redis, telescope) |
| `orca/celery.py` | Celery app definition, queues, routing |
| `orca/configmanager.py` | Loads configuration |
| `orca/tasks/` | Task definitions |

### Key Commands

```bash
# Start worker
celery -A orca.celery worker -Q default --hostname=default@$(hostname) -c 40

# Check workers
python3 -c "from orca.celery import app; print(app.control.ping())"

# Stop workers
pkill -15 -f 'celery.*worker'

# Monitor (Flower)
celery -A orca.celery flower --port=5555

# Purge queue
celery -A orca.celery purge -Q <queue>
```

### Service Locations

| Service | Host | Port | Config |
|---------|------|------|--------|
| RabbitMQ | lwacalimhead | 5672 | `/etc/rabbitmq/` |
| Redis | lwacalim10 | 6379 | `/etc/redis.conf` |
| Flower | lwacalim10 | 5555 | Started manually |

---

## Infrastructure Reference

These services are already installed. This section is for reference/recovery only.

### RabbitMQ

- **Docs:** https://www.rabbitmq.com/docs
- **Install:** https://www.rabbitmq.com/docs/install-debian
- **Location:** lwacalimhead
- **Start:** `sudo systemctl start rabbitmq-server`
- **Config:** `/etc/rabbitmq/rabbitmq.conf`

### Redis

- **Docs:** https://redis.io/docs/
- **Install:** https://redis.io/docs/getting-started/installation/install-redis-on-linux/
- **Location:** lwacalim10
- **Start:** `sudo systemctl start redis`
- **Config:** `/etc/redis.conf`
