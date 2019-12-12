## Prepare for launch
### Set up the log directories (they they don't already exist)
```
sudo pdsh -w astm[04-13] "mkdir -p /var/log/celery/"
sudo pdsh -w astm[04-13] "chmod -R 777 /var/log/celery/"
```
### Launch the rabbimq server
```
sudo /sbin/service rabbitmq-server start
```

## Launch workers 
### Start the workers with concurrencey 20 or whatever you like
```
start_workers.py --concurrency 20
```

## Shutdown/Cleanup
### Kill all pending tasks
```celery -A orca.proj purge```

### Kill the workers
This is needed after a task code update. Then you need to launch workers again for
the changes to take effect.

Send SIGTERM (usually works and wait till current job finishes on a worker.)
```pdsh -w astm[04-12] "pkill -15 -f 'celery worker'"```

Send SIGKILL (more disruptive)
```pdsh -w astm[04-12] "pkill -9 -f 'celery worker'"```

## Diagnostics
### check messages in the rabbitmq queue
```
sudo rabbitmqctl list_queues -p yuping
```
