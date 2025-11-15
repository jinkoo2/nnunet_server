import redis
from rq import Queue
from app.core.test_worker import add_numbers
import time

# --- Configuration ---
# RQ uses Redis for storing job data and queue management.
# Assuming Redis is running on localhost (127.0.0.1) on the default port 6379.
# Adjust the host/port if your Redis is elsewhere.
redis_host = 'localhost'
redis_port = 6379
queue_name = 'nnunet_jobs' # Use the queue name your worker is listening to

# Connect to Redis
try:
    redis_conn = redis.Redis(host=redis_host, port=redis_port)
    # Ping Redis to ensure the connection is working
    redis_conn.ping()
    print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    print("Please ensure your Redis server is running.")
    exit(1)

# Create an RQ Queue object
nnunet_queue = Queue(queue_name, connection=redis_conn)

# --- Submission Logic ---
print(f"Submitting 10 test jobs to the '{queue_name}' queue...")

job_data = [
    (10, 5), (15, 3), (20, 1), (25, 6), (30, 2),
    (35, 7), (40, 4), (45, 9), (50, 0), (55, 8)
]

job_ids = []
for i, (num1, num2) in enumerate(job_data):
    # Enqueue the job. 
    # The job is executed by the worker listening to this queue.
    job = nnunet_queue.enqueue(
        add_numbers,         # The function to run
        num1, num2,          # Arguments passed to the function
        job_timeout='1h',    # Set a timeout for the job execution
        result_ttl=3600      # Keep results for 1 hour
    )
    job_ids.append(job.id)
    print(f"  Submitted Job {i+1}/10. ID: {job.id}")

print(f"\n10 jobs submitted successfully to the '{queue_name}' queue.")
print("The worker should start processing them immediately.")