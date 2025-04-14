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
        'orca.transform.imaging',
        'orca.transform.photometry',
        'orca.tasks.pipeline_tasks'  # where your tasks live
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
app.conf.task_queues = (
    Queue('default',   Exchange('default'),   routing_key='default'),
    Queue('cosmology', Exchange('cosmology'), routing_key='cosmology'),
)

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
    # If you have other tasks you want dedicated to cosmology, list them here
    #
    # e.g. 'orca.tasks.pipeline_tasks.flag_foo_task': {'queue': 'cosmology'},
}

if __name__ == '__main__':
    app.start()

