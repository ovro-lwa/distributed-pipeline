from __future__ import absolute_import, unicode_literals
from celery import Celery

CELERY_APP_NAME = 'proj'

app = Celery(CELERY_APP_NAME,
             broker='pyamqp://yuping:yuping@astm13:5672/yuping',
             backend='rpc://',
             include=['orca.proj.longbandpass', 'orca.transform', 'orca.proj.onedayaverage'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    worker_prefetch_multiplier=1
)

if __name__ == '__main__':
    app.start()
