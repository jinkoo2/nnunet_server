from fastapi import APIRouter
from pydantic import BaseModel
import os
from pathlib import Path
import json
from redis import Redis
from rq import Queue
from rq.job import Job
import logging

from app.core.nnunet_plan_and_preprocess import plan_and_preprocess_slurm, plan_and_preprocess_sh
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# nnU-Net directories
nnunet_data_dir = settings.NNUNET_DATA_DIR
nnunet_raw_dir = os.path.join(nnunet_data_dir, 'raw')
nnunet_preprocessed_dir = os.path.join(nnunet_data_dir, 'preprocessed')
nnunet_results_dir = os.path.join(nnunet_data_dir, 'results')

# Ensure directories exist
Path(nnunet_raw_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_preprocessed_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_results_dir).mkdir(parents=True, exist_ok=True)

logger.info(f"nnunet_data_dir={nnunet_preprocessed_dir}")

# Redis connection and RQ queue
redis_conn = Redis(host="localhost", port=6379)
queue = Queue(connection=redis_conn)

# FastAPI router
router = APIRouter()

class PlanAndPreprocessTaskRequest(BaseModel):
    dataset_num: int
    planner: str
    verify_dataset_integrity: bool


def run_plan_and_preprocess_sh_rq(dataset_num: int, planner: str, verify_dataset_integrity: bool):
    """Function that will be executed by RQ worker."""
    try:
        logger.info(f"RQ Worker: RUNNING plan_and_preprocess_sh() for dataset {dataset_num}")
        plan_and_preprocess_sh(dataset_num, planner, verify_dataset_integrity)
        logger.info(f"RQ Worker: plan_and_preprocess_sh() completed successfully for dataset {dataset_num}")
    except Exception as e:
        logger.exception(f"RQ Worker: Error in plan_and_preprocess_sh(): {e}")
        raise e


@router.post("/plan-and-preprocess/")
async def plan_and_preprocess(request: PlanAndPreprocessTaskRequest):
    dataset_num = request.dataset_num
    planner = request.planner
    verify_dataset_integrity = request.verify_dataset_integrity

    logger.info(f"dataset_num={dataset_num}")
    logger.info(f"planner={planner}")
    logger.info(f"verify_dataset_integrity={verify_dataset_integrity}")

    if settings.JOB_PROCESSOR == "slurm":
        logger.info("RUNNING... plan_and_preprocess_slurm()")
        plan_and_preprocess_slurm(dataset_num, planner, verify_dataset_integrity)
        message = "Plan and preprocess task (SLURM) submitted successfully"
        return {"message": message}
    
    # Enqueue the SH job to RQ
    job = queue.enqueue(
        run_plan_and_preprocess_sh_rq,
        dataset_num,
        planner,
        verify_dataset_integrity,
        job_timeout=172800  # 2 days in seconds
    )

    logger.info(f"Enqueued plan_and_preprocess_sh() with job_id={job.id}")
    return {"message": "Plan and preprocess task (SH) enqueued successfully", "job_id": job.id}


@router.get("/plan-and-preprocess/status/{job_id}")
async def get_job_status(job_id: str):
    """Check RQ job status."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return {"job_id": job.id, "status": job.get_status(), "result": job.result}
    except Exception as e:
        return {"error": f"Job {job_id} not found or failed: {str(e)}"}
