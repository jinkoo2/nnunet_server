
import time
import os
import datetime

# This function simulates your long-running nnUNet prediction task.
# It takes about 5 seconds to complete.
def add_numbers(a, b):
    """
    Adds two numbers and logs the process, simulating a 5-second computation.
    """
    job_id = os.environ.get('RQ_JOB_ID', 'N/A')
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] JOB {job_id[:8]} STARTING...")
    print(f"  Simulating 5-second prediction for: {a} + {b}")
    
    # Simulate the time it takes to run an AI model or prediction
    time.sleep(5)
    
    result = a + b
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] JOB {job_id[:8]} FINISHED. Result: {result}")
    
    return result