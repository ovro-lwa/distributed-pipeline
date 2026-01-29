# Installation

## Requirements

- Python 3.8 (required for CASA compatibility)
- Redis server (for Celery result backend)
- RabbitMQ or Redis (for Celery message broker)

## Installing from source

```bash
git clone https://github.com/ovro-lwa/distributed-pipeline.git
cd distributed-pipeline
pip install -e .
```

## Configuration

Create a configuration file at `~/orca-conf.yml`:

```yaml
queue:
  prefix: default
  broker_uri: pyamqp://user:pass@rabbitmq-host:5672/vhost
  result_backend_uri: redis://redis-host:6379/0

telescope:
  n_ant: 352
  n_subband: 16
  n_chan: 192

execs:
  wsclean: /opt/bin/wsclean
  aoflagger: /opt/bin/aoflagger

cluster: calim
```

## Dependencies

The full list of dependencies is in `requirements.txt`. Key packages include:

- `casatools` / `casatasks` - CASA measurement set handling
- `python-casacore` - Low-level MS access
- `celery[redis]` - Distributed task queue
- `astropy` - Astronomical utilities
- `numpy` / `scipy` - Numerical computing
