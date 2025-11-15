"""
Test script to submit a single, highly specific nnU-Net inference job using RQ.

This submission script sets the exact parameters needed to generate the command
requested by the user.

Requirements:
1. Redis server running on localhost:6379.
2. RQ Worker listening to the 'nnunet_jobs' queue.
3. The nnU-Net input directory MUST exist and contain valid files:
   /home/jk/data/nnunet_data/predictions/Dataset015_CBCTBladderRectumBowel2/req_000
4. Your 'run_nnunet_predict' function must support the 'folds' and 'device' keys.
"""
import redis
from rq import Queue
import time
import json
from pathlib import Path

# --- IMPORTANT: Adjust the import path for your worker function ---
try:
    from app.core.nnunet_worker import run_nnunet_predict
except ImportError as e:
    print(f"Error importing run_nnunet_predict: {e}")
    print("Please ensure your Python path is set correctly.")
    exit(1)


# --- Configuration ---
redis_host = 'localhost'
redis_port = 6379
queue_name = 'nnunet_jobs'

# Connect to Redis
try:
    redis_conn = redis.Redis(host=redis_host, port=redis_port)
    redis_conn.ping()
    print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    print("Please ensure your Redis server is running.")
    exit(1)

# Create an RQ Queue object
nnunet_queue = Queue(queue_name, connection=redis_conn)


# --- Define the Job Metadata (Matching the Command) ---

# The dataset ID requested by the -d flag
DATASET_ID = "015" 

# The full input directory path requested by the -i flag
INPUT_PATH = "/home/jk/data/nnunet_data/predictions/Dataset015_CBCTBladderRectumBowel2/req_000"

# --- Pre-flight Check: Ensure the input directory exists ---
# Note: Since this path is likely on the server where the worker runs, 
# we rely on the worker to handle the error if it doesn't exist.
# But for a local test, we assume it does:
# Path(INPUT_PATH).mkdir(parents=True, exist_ok=True) 


JOB_METADATA = {
    # Unique ID for tracking this specific job
    "job_id": f"job_for_req_000", 
    
    # Corresponds to -d 015
    "dataset_id": DATASET_ID, 
    
    # Corresponds to -i /path/to/input
    "input_dir": INPUT_PATH, 
    
    # Corresponds to -c 3d_lowres
    "configuration": "3d_lowres",
      
    # Corresponds to -device gpu (If your worker uses this to set the device)
    "device": "gpu", 
    
    # Standard nnU-Net keys, usually defaults if not overridden
    "trainer": "nnUNetTrainer", 
    "plans": "nnUNetPlans",
    "requester_id": "vtk_image_labeler_3d@varianEclipseTest",
}


# --- Submission Logic ---
print(f"Submitting nnU-Net prediction job {JOB_METADATA['job_id']}...")

# Enqueue the job, passing the entire JOB_METADATA dictionary as the only argument
job = nnunet_queue.enqueue(
    run_nnunet_predict, # The function the worker will execute
    JOB_METADATA,       # The required dictionary of parameters
    job_timeout='3h',   # Allow ample time for a large segmentation job
    result_ttl=86400    # Keep results for 24 hours
)

print(f"\nJob submitted successfully to queue '{queue_name}'.")
print(f"  Job ID: {job.id}")
print(f"  Configuration: Dataset {JOB_METADATA['dataset_id']}")
print("\n--- Worker Action Check ---")
print("Verify your worker logs show the subprocess executing this command:")
print(f"  nnUNetv2_predict -d 015 -i {INPUT_PATH} -o <OUTPUT_PATH> -f 0 1 2 3 4 -c 3d_lowres -device gpu")