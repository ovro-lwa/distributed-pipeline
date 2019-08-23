# Prep work
# sudo pdsh -w astm[04-13] "mkdir -p /var/log/celery/"
# sudo pdsh -w astm[04-13] "chmod -R 777 /var/log/celery/"

# start
# pdsh -w astm[04-12] 'cd /home/yuping/transient && celery multi start w1 -A orca.proj --concurrency=20 -l info -n %h --pidfile=/var/run/celery/%n.pid --logfile=/var/log/celery/%n%I.log'

# Kill all pending tasks
#celery -A orca.proj purge

# kill
# pdsh -w astm[04-12] "pkill -9 -f 'celery worker'"
