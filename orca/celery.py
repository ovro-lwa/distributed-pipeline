from __future__ import absolute_import, unicode_literals
from celery import Celery
from orca.configmanager import queue_config

CELERY_APP_NAME = 'orca'
REDIS_URL = 'redis://10.41.0.85:6379/0'

# TODO: what if I import a module that imports orca.transform.*?

app = Celery(CELERY_APP_NAME,
             broker='pyamqp://pipe:pipe@rabbitmq.calim.mcs.pvt:5672/pipe',
             backend=REDIS_URL,
             include=['orca.transform.calibration',
                      'orca.transform.qa',
                      'orca.tasks.fortests',
                      'orca.transform.spectrum',
                      'orca.transform.imaging'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=7200,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)

app.conf.task_routes = {'*': queue_config.prefix}

if __name__ == '__main__':
    app.start()
