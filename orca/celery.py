"""Celery application configuration for the ORCA distributed pipeline.

This module initializes the Celery application with broker and backend
settings, configures task queues for different workload types (default,
cosmology, bandpass, imaging), and sets up task routing to direct specific
tasks to appropriate queues.

The Celery app is configured to:
- Use JSON serialization for tasks
- Limit worker prefetch to 1 task at a time
- Restart workers after 20 tasks to prevent memory leaks
- Expire results after 2 hours

Queues
------
default
    General-purpose queue for most tasks.
cosmology
    Queue for cosmology-specific processing tasks.
bandpass
    Queue for bandpass calibration tasks.
imaging
    Queue for imaging pipeline tasks.
calim00 .. calim10
    Per-node queues for NVMe-local subband processing.  Each lwacalimNN
    worker listens on its own ``calimNN`` queue so that subband data stays
    on the node's local NVMe.  Tasks are routed dynamically at submit time
    via ``task.apply_async(queue='calim08')``.

Example
-------
Start a worker for the imaging queue::

    celery -A orca.celery worker -Q imaging -c 4

Start a worker on lwacalim08 for subband processing::

    celery -A orca.celery worker -Q calim08 --hostname=calim08@lwacalim08 -c 4

"""
# orca/celery.py

from __future__ import absolute_import, unicode_literals
from celery import Celery
from orca.configmanager import queue_config

# You'll need these if you define custom Queues/Exchanges
from kombu import Queue, Exchange

CELERY_APP_NAME = 'orca'

app = Celery(
    CELERY_APP_NAME,
    broker=queue_config.broker_uri,
    backend=queue_config.result_backend_uri,
    include=[
        'orca.transform.calibration',
        'orca.transform.qa',
        'orca.tasks.fortests',
        'orca.transform.spectrum',
        'orca.transform.spectrum_v2',
        'orca.transform.spectrum_v3',
        'orca.transform.imaging',
        'orca.transform.photometry',
        'orca.tasks.pipeline_tasks',  
        'orca.tasks.imaging_tasks',
        'orca.tasks.subband_tasks',
        #'orca.tasks.peel_stage1_tasks',
    ]
)

# Basic configs
app.conf.update(
    result_expires=7200,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,
    task_serializer='json',
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)

######################
# Define your QUEUES
######################

# Per-node queues for NVMe-local subband processing.
# Each lwacalimNN worker listens on its own calimNN queue so that
# subband data stays on the node's local NVMe.
_calim_queues = tuple(
    Queue(f'calim{i:02d}', Exchange(f'calim{i:02d}'), routing_key=f'calim{i:02d}')
    for i in range(11)   # calim00 .. calim10
)

app.conf.task_queues = (
    Queue('default',   Exchange('default'),   routing_key='default'),
    Queue('cosmology', Exchange('cosmology'), routing_key='cosmology'),
    Queue('bandpass',  Exchange('bandpass'),  routing_key='bandpass'),
    Queue('imaging',   Exchange('imaging'),   routing_key='imaging'),
) + _calim_queues

# If you still want "default" to be the fallback for any tasks not explicitly routed
app.conf.task_default_queue = 'default'
app.conf.task_default_exchange = 'default'
app.conf.task_default_routing_key = 'default'

###################
# TASK ROUTING
###################
app.conf.task_routes = {
    # All pipeline tasks can stay on default, *except* the special one(s):
    #
    # Example: route the new cosmology tasks to "cosmology" queue
    'orca.tasks.pipeline_tasks.split_2pol_task': {'queue': 'cosmology'},
    # If you have other tasks you want, list them here
    #
    # e.g. 'orca.tasks.pipeline_tasks.flag_foo_task': {'queue': 'cosmology'},
    'orca.tasks.pipeline_tasks.bandpass_nvme_task': {'queue': 'bandpass'},
    'orca.tasks.imaging_tasks.imaging_pipeline_task': {'queue': 'imaging'},
    'orca.tasks.imaging_tasks.imaging_shared_pipeline_task': {'queue': 'imaging'},

    # Subband tasks: queue is set dynamically at submit time via .set(queue=...)
    # so they are NOT statically routed here.  The submission script uses
    # orca.resources.subband_config.get_queue_for_subband() to pick the right
    # calimNN queue.  See orca/tasks/subband_tasks.py for details.
    }

if __name__ == '__main__':
    app.start()

