sudo pdsh -w astm[04-13] "mkdir -p /var/log/celery/"
sudo pdsh -w astm[04-13] "chmod -R 777 /var/log/celery/"
