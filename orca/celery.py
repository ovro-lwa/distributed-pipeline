from __future__ import absolute_import, unicode_literals
from celery import Celery
from orca.configmanager import queue_config

CELERY_APP_NAME = 'orca'


app = Celery(CELERY_APP_NAME,
             broker=queue_config.broker_uri,
             backend=queue_config.result_backend_uri,
             include=['orca.transform.calibration',
                      'orca.transform.qa',
                      'orca.tasks.fortests',
                      'orca.transform.spectrum',
                      'orca.transform.imaging',
                      'orca.transform.photometry',
                      'orca.tasks.pipeline_tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=7200,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,
    task_serializer='json',
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)

app.conf.task_routes = {'*': queue_config.prefix}

if __name__ == '__main__':
    app.start()
