
import json
import os
import re
from pathlib import Path

from logger import logger, log_exception as LE, log_and_raise_exception as LER

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

def read_dataset_json(dirname):
    """Read dataset.json file asynchronously."""
    json_file = os.path.join(nnunet_raw_dir, dirname, 'dataset.json')
    try:
        with open(json_file, 'r') as f:
            data = f.read()
            data_dict = json.loads(data)
            data_dict['id'] = dirname
            return data_dict
    except Exception as e:
        LE(f'Failed reading file {json_file}. Exception: {e}')
        return None
    
def get_dataset_list():
    """Retrieve dataset list asynchronously."""
    dirnames = get_dataset_dirs(nnunet_raw_dir)
    dataset_list = [read_dataset_json(dirname) for dirname in dirnames]
    
    # Remove None values (failed reads)
    dataset_list = [ds for ds in dataset_list if ds]

    dataset_list = sorted(dataset_list, key=lambda x: x["id"])

    logger.info(f'dataset_list={dataset_list}')

    return dataset_list

def get_dataset_id_list():
    return get_dataset_dirs(nnunet_raw_dir)

if __name__ == '__main__':
    print('done')


