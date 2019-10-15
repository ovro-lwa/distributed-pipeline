"""
Convenience executable to start celery worker across the cluster.
"""
from fabric import ThreadingGroup
import argparse

ENV_DIR = '/opt/astro/devel/yuping/transient/'


def main(hosts, concurrency):
    with ThreadingGroup(*hosts) as sg:
        results = sg.run(f'cd {ENV_DIR} && /opt/astro/bin/pipenv run celery multi start w1 -A orca.proj '
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
