import re, os, json
from pathlib import Path
import aiofiles

from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
print(f'nnunet_data_dir={nnunet_data_dir}')

nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')
nnunet_preprocessed_dir =  os.path.join(nnunet_data_dir,'preprocessed')
nnunet_results_dir =  os.path.join(nnunet_data_dir,'results')

def get_dataset_dirs(folder_path):
    """Get a list of data set."""
    pattern = r"^Dataset\d{3}_.+$"  # Regex for Datasetxxx_yyyyyy format
    return [entry.name for entry in Path(folder_path).iterdir() if entry.is_dir() and re.match(pattern, entry.name)]

async def read_dataset_json(dirname):
    """Read dataset.json file asynchronously."""
    json_file = os.path.join(nnunet_raw_dir, dirname, 'dataset.json')
    try:
        async with aiofiles.open(json_file, 'r') as f:
            data = await f.read()
            data_dict = json.loads(data)
            data_dict['id'] = dirname
            return data_dict
    except Exception as e:
        LE(f'Failed reading file {json_file}. Exception: {e}')
        return None

import nnunet_tools
async def get_image_name_list(dataset_id: str):
    """
    Returns lists of training/test image and label filenames and their extracted IDs.
    """
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")

    if not os.path.exists(dataset_json_path):
        raise Exception(f'Dataset {dataset_id} not found')

    # Read dataset.json asynchronously
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

async def get_dataset(dataset_id:str):
    dataset_json = await read_dataset_json(dataset_id)
    image_list = await get_image_name_list(dataset_id)
    return {
        'dataset_json':dataset_json,
        'image_list': image_list
    }

import asyncio

if __name__ == '__main__':
    dataset_id = "Dataset935_Test1"
    
    async def main():
        data = await get_image_name_list(dataset_id)
        print(data)
    
    asyncio.run(main())
