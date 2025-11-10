from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import json
import random
import os
import re

from logger import logger, log_exception as LE, log_and_raise_exception as LER

from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
print(f'nnunet_data_dir={nnunet_data_dir}')

nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir =  os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir =  os.path.join(nnunet_data_dir,'results')

from nnunet_raw import get_dataset_dirs, read_dataset_json

# Define the router
router = APIRouter()

import asyncio

@router.get("/dataset_json/list")
async def get_dataset_json_list():
    """Retrieve dataset list asynchronously."""
    dirnames = get_dataset_dirs(nnunet_raw_dir)
    tasks = [read_dataset_json(dirname) for dirname in dirnames]
    dataset_list = await asyncio.gather(*tasks)
    
    # Remove None values (failed reads)
    dataset_list = [ds for ds in dataset_list if ds]

    dataset_list = sorted(dataset_list, key=lambda x: x["id"])

    logger.info(f'dataset_list={dataset_list}')

    return dataset_list

@router.get("/dataset_json/id-list")
async def get_dataset_json_id_list():
    """Get dataset is list"""
    ids = get_dataset_dirs(nnunet_raw_dir)

    logger.info(f'ids={ids}')

    return ids

class DatasetJsonRequest(BaseModel):
    """Pydantic model for dataset validation."""
    name: str
    description: str
    reference: str
    licence: str
    tensorImageSize: str
    labels: dict
    channel_names: dict

@router.post("/dataset_json/new")
async def post_dataset(request: Request):
    """Create a new dataset with validation."""
    try:
        data = await request.json()  # Extract JSON body
        logger.info(f"Received dataset: {json.dumps(data, indent=4)}")

        # **Validation 1: Required fields**
        required_fields = ["name", "description", "reference", "licence", "tensorImageSize", "labels", "channel_names","file_ending"]
        for field in required_fields:
            if field not in data:
                LER(HTTPException(status_code=400, detail=f"Missing required field: {field}"))

        # **Validation 2: Check if "name" is alphanumeric without spaces**
        if not data["name"].isalnum():
            LER(HTTPException(status_code=400, detail="Error: 'name' must be alphanumeric and contain no spaces."))

        # **Validation 3: Check if "tensorImageSize" is either "2D" or "3D"**
        if data["tensorImageSize"] not in ["2D", "3D"]:
            LER(HTTPException(status_code=400, detail="Error: 'tensorImageSize' must be either '2D' or '3D'."))

        # **Validation 4: Ensure "labels" contains "background" with value 0**
        if "background" not in data["labels"] or data["labels"]["background"] != 0:
            LER(HTTPException(status_code=400, detail="Error: 'labels' must include 'background' with value 0."))

        # **Set default values if not present**
        data.setdefault("numTraining", 0)
        data.setdefault("numTest", 0)

        # **Find all used dataset numbers**
        existing_datasets = get_dataset_dirs(nnunet_raw_dir)
        used_numbers = set()
        for dataset in existing_datasets:
            match = re.match(r"Dataset(\d{3})_(.+)$", dataset)
            if match:
                used_numbers.add(int(match.group(1)))
        logger.info(f"Used dataset numbers: {used_numbers}")
        
        # **Check if the dataset name is already in use**
        existing_datasets_lowercase = {name.lower() for name in existing_datasets}
        logger.info(f"Existing datasets (lower case): {existing_datasets_lowercase}")
        if data["name"].lower() in existing_datasets_lowercase:
            LER(HTTPException(status_code=400, detail=f"Error: Dataset name '{data['name']}' already exists."))
    
        # **Generate a random unused dataset number**
        available_numbers = set(range(1, 1000)) - used_numbers  # Numbers 1-999 not in use
        if not available_numbers:
            LER(HTTPException(status_code=500, detail="Error: No available dataset numbers left."))

        new_dataset_num = random.choice(list(available_numbers))

        logger.info(f'new_dataset_num={new_dataset_num}')

        # **Generate new dataset ID**
        dataset_id = f"Dataset{new_dataset_num:03d}_{data['name']}"
        dataset_path = os.path.join(nnunet_raw_dir, dataset_id)

        logger.info(f'dataset_path={dataset_path}')

        # **Create new dataset directory and save dataset.json**
        os.makedirs(dataset_path, exist_ok=True)
        json_path = os.path.join(dataset_path, "dataset.json")
        logger.info(f'json_path={json_path}')
        logger.info(f'data={data}')

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        data["id"] = dataset_id

        logger.info(f"Dataset created successfully: {dataset_id}")

        return {
            "message": "Dataset successfully created!",
            "dataset": data
        }

    except json.JSONDecodeError:
        LER(HTTPException(status_code=400, detail="Invalid JSON format. Please send valid JSON."))

    except Exception as e:
        LER(HTTPException(status_code=500, detail=f"An error occurred: {str(e)}"))
    




    
