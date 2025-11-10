import re, os, json
from pathlib import Path
import aiofiles
import asyncio

# settings
from app.core.config import settings

# logging
from app.core.logging_config import get_logger
import app.core.nnunet_tools as nnunet_tools

logger = get_logger(__name__)

nnunet_data_dir = settings.NNUNET_DATA_DIR
nnunet_raw_dir = os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir = os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir = os.path.join(nnunet_data_dir,'results')

# Ensure directories exist
Path(nnunet_raw_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_preprocessed_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_results_dir).mkdir(parents=True, exist_ok=True)

logger.info(f'nnunet_data_dir={nnunet_data_dir}')

def get_dataset_dirs() -> list[str]:
    """Get a list of dataset directories matching DatasetXXX_name pattern."""
    pattern = r"^Dataset\d{3}_.+$"
    return [entry.name for entry in Path(nnunet_raw_dir).iterdir() 
            if entry.is_dir() and re.match(pattern, entry.name)]

async def read_dataset_json(dirname: str) -> dict | None:
    """Read dataset.json file asynchronously."""
    json_file = os.path.join(nnunet_raw_dir, dirname, 'dataset.json')
    try:
        async with aiofiles.open(json_file, 'r') as f:
            data = await f.read()
            data_dict = json.loads(data)
            data_dict['id'] = dirname
            return data_dict
    except Exception as e:
        logger.error(f'Failed reading file {json_file}. Exception: {e}')
        return None

async def get_image_name_list(dataset_id: str) -> dict:
    """Return lists of training/test image and label filenames."""
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")

    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f'Dataset {dataset_id} not found')

    try:
        async with aiofiles.open(dataset_json_path, "r") as f:
            data = await f.read()
            dataset_info = json.loads(data)
    except Exception as e:
        raise Exception(f"Failed to read dataset.json: {str(e)}")

    file_ending = dataset_info.get("file_ending")
    if not file_ending:
        raise Exception("Missing 'file_ending' in dataset.json.")

    def extract_image_files_with_ids(folder: str):
        return nnunet_tools.find_image_files(folder, file_ending)

    def extract_label_files_with_ids(folder: str):
        return nnunet_tools.find_label_files(folder, file_ending)

    return {
        "train_images": extract_image_files_with_ids(os.path.join(dataset_path, "imagesTr")),
        "train_labels": extract_label_files_with_ids(os.path.join(dataset_path, "labelsTr")),
        "test_images": extract_image_files_with_ids(os.path.join(dataset_path, "imagesTs")),
        "test_labels": extract_label_files_with_ids(os.path.join(dataset_path, "labelsTs")),
    }

async def get_dataset(dataset_id: str) -> dict:
    dataset_json = await read_dataset_json(dataset_id)
    image_list = await get_image_name_list(dataset_id)
    return {
        'dataset_json': dataset_json,
        'image_list': image_list
    }

if __name__ == '__main__':
    dataset_id = "Dataset935_Test1"
    async def main():
        data = await get_image_name_list(dataset_id)
        print(data)
    asyncio.run(main())
