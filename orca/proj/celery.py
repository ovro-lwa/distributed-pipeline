from __future__ import absolute_import, unicode_literals
from celery import Celery
from orca.configmanager import queue_config

CELERY_APP_NAME = 'proj'

app = Celery(CELERY_APP_NAME,
             broker='pyamqp://yuping:yuping@astm13:5672/yuping',
             backend='rpc://',
             include=['orca.proj.boilerplate',
                      'orca.transform',
                      'orca.proj.onedayaverage',
                      'orca.proj.gainvariation',
                      'orca.proj.transientbatchtasks'
                      ])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_serializer='json'
)

task_routes = {'*': queue_config['prefix']}

app.conf.task_routes = task_routes

if __name__ == '__main__':
    app.start()
