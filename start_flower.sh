pkill -f 'flower'
nohup celery flower -A orca.proj --broker=amqp://yuping:yuping@localhost:5672/yuping --port=5555 &
