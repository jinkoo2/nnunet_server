from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import json
import random
import os

from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
print(f'nnunet_data_dir={nnunet_data_dir}')

nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir =  os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir =  os.path.join(nnunet_data_dir,'results')

# Define the router
router = APIRouter()

class PlanAndPreprocessTaskRequest(BaseModel):
    dataset_num: int
    planner: str
    verify_dataset_integrity: bool

@router.post("/plan-and-preprocess/")
async def plan_and_preprocess(request: PlanAndPreprocessTaskRequest):
    """Submit a task for processing."""
    # Extract the `data` field from the TaskRequest object
    
    dataset_num = request.dataset_num
    planner = request.planner
    verify_dataset_integrity = request.verify_dataset_integrity

    print(f"dataset_num={dataset_num}")
    print(f"planner={planner}")
    print(f"verify_dataset_integrity={verify_dataset_integrity}")

    from nnunet_plan_and_preprocess import plan_and_preprocess_slurm
    print('RUNNING...plan_and_preprocess_slurm()')
    plan_and_preprocess_slurm(dataset_num, planner, verify_dataset_integrity)

    
