from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

CELERY_APP_NAME = 'proj'
DEVELOPMENT_VARIABLE = 'ORCA_DEV'

app = Celery(CELERY_APP_NAME,
             broker='pyamqp://yuping:yuping@astm13:5672/yuping',
             backend='rpc://',
             include=['orca.proj.longbandpass',
                      'orca.proj.boilerplate'
                      'orca.transform',
                      'orca.proj.onedayaverage',
                      'orca.proj.gainvariation',
                      'orca.proj.pipeline'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    worker_prefetch_multiplier=1
)

# TODO: maybe I should split tasks in boilerplate based on queue?
if DEVELOPMENT_VARIABLE in os.environ:
    task_routes = {'*': os.environ[DEVELOPMENT_VARIABLE]}
else:
    task_routes = {'*': 'default-1-per-node'}

app.conf.task_routes = task_routes

if __name__ == '__main__':
    app.start()
