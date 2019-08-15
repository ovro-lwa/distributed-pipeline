# Prep work
# sudo pdsh -w astm[04-13] "mkdir -p /var/log/celery/"
# sudo pdsh -w astm[04-13] "chmod -R 777 /var/log/celery/"

# start
# celery multi start w1 -A proj -l info -n w1@%n --pidfile=/var/run/celery/%n.pid --logfile=/var/log/celery/%n%I.log
# restart
# celery multi restart w1 -n w1@%n --pidfile=/var/run/celery/%n.pid --logfile=/var/run/celery/%n.pid

# kill
# pkill -9 -f 'celery worker'
