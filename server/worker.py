# from redis import Redis
# from rq import Worker, Queue
# from rq.connections import Connection

# listen = ["file_tasks"]
# redis_conn = Redis(host="localhost", port=6380, db=0)

# if __name__ == "__main__":
#     with Connection(redis_conn):
#         Worker(map(Queue, listen)).work()
from redis import Redis
from rq import Worker, Queue

listen = ["file_tasks"]
redis_conn = Redis(host="localhost", port=6379, db=0)

if __name__ == "__main__":
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = Worker(queues, connection=redis_conn)
    worker.work()