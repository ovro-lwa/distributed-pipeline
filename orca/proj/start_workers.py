"""
Convenience executable to start celery worker across the cluster.
"""
from fabric import ThreadingGroup
import argparse
import os

from orca.proj.celery import DEVELOPMENT_VARIABLE

ENV_DIR = '/opt/astro/devel/yuping/orca/'


def main(hosts, concurrency):
    with ThreadingGroup(*hosts) as sg:
        queue_name = 'default-1-per-node'
        if DEVELOPMENT_VARIABLE in os.environ:
            queue_name = os.environ[DEVELOPMENT_VARIABLE]
        worker_name = queue_name
        # Revisit this later https://docs.celeryproject.org/en/stable/reference/celery.bin.multi.html can have multiple
        # queues on the same worker.
        results = sg.run('/opt/astro/anaconda3/bin/activate py36_orca && '
                         f'cd {ENV_DIR} && /opt/astro/bin/pipenv run celery multi start {worker_name} -A orca.proj '
                         f'-Q {queue_name} '
                         f'--concurrency={concurrency} '
                         f'-l info -n %h --pidfile=/var/run/celery/%n.pid --logfile=/var/log/celery/%n%I.log')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start celery workers across the cluster. Default is to start them on'
                                                 'all nodes but astm13.')
    parser.add_argument('--concurrency', type=int, default=20, help='worker concurrency')
    parser.add_argument('--exclude', type=int, nargs='+', default=[], help='Space-delimited list of hosts to exclude.')
    args = parser.parse_args()
    host_list = [f'astm{i:02d}' for i in range(4, 13) if i not in args.exclude]
    main(host_list, args.concurrency)
