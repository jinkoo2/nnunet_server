from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import os, json, re, random, asyncio
from pathlib import Path
import aiofiles

# settings
from app.core.config import settings

# logging
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

logger.info(f"nnunet_data_dir={nnunet_data_dir}")

# Core module
import app.core.nnunet_raw as raw

router = APIRouter()

# ---------------- Pydantic Model ----------------
class DatasetJsonRequest(BaseModel):
    name: str
    description: str
    reference: str
    licence: str
    tensorImageSize: str
    labels: dict
    channel_names: dict
    file_ending: str
    numTraining: int = 0
    numTest: int = 0


# ---------------- Helper Functions ----------------
def log_and_raise_exception(e: Exception):
    logger.error("Exception occurred", exc_info=e)
    raise e


# ---------------- Routes ----------------
@router.get("/dataset_json/list")
async def get_dataset_json_list():
    """Retrieve dataset list asynchronously."""
    dirnames = raw.get_dataset_dirs()
    tasks = [raw.read_dataset_json(dirname) for dirname in dirnames]
    dataset_list = await asyncio.gather(*tasks)
    
    # Remove None values (failed reads)
    dataset_list = [ds for ds in dataset_list if ds]
    dataset_list = sorted(dataset_list, key=lambda x: x["id"])

    logger.info(f"dataset_list={dataset_list}")
    return dataset_list


@router.get("/dataset_json/id-list")
async def get_dataset_json_id_list():
    """Get dataset ID list"""
    ids = raw.get_dataset_dirs()
    logger.info(f"ids={ids}")
    return ids


@router.post("/dataset_json/new")
async def post_dataset(request: Request):
    """Create a new dataset with validation."""
    try:
        data_dict = await request.json()
        logger.info(f"Received dataset: {json.dumps(data_dict, indent=2)}")

        # Validate using Pydantic
        dataset = DatasetJsonRequest(**data_dict)

        # Check dataset name uniqueness
        existing_datasets = raw.get_dataset_dirs()
        existing_lower = {name.lower() for name in existing_datasets}
        if dataset.name.lower() in existing_lower:
            raise HTTPException(status_code=400, detail=f"Dataset '{dataset.name}' already exists.")

        # Find used dataset numbers
        used_numbers = set()
        pattern = r"Dataset(\d{3})_.+"
        for d in existing_datasets:
            match = re.match(pattern, d)
            if match:
                used_numbers.add(int(match.group(1)))

        # Generate new dataset number
        available_numbers = set(range(1, 1000)) - used_numbers
        if not available_numbers:
            raise HTTPException(status_code=500, detail="No available dataset numbers left.")
        new_number = random.choice(list(available_numbers))
        dataset_id = f"Dataset{new_number:03d}_{dataset.name}"
        dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
        Path(dataset_path).mkdir(parents=True, exist_ok=True)

        # Save dataset.json asynchronously
        json_path = os.path.join(dataset_path, "dataset.json")
        async with aiofiles.open(json_path, 'w') as f:
            await f.write(json.dumps(dataset.dict(), indent=4))

        logger.info(f"Dataset created successfully: {dataset_id}")
        response = dataset.dict()
        response["id"] = dataset_id
        return {"message": "Dataset successfully created!", "dataset": response}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format. Please send valid JSON.")
    except Exception as e:
        log_and_raise_exception(HTTPException(status_code=500, detail=str(e)))
