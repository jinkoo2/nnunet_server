import requests
import time

BASE_URL = "http://127.0.0.1:8000"  # FastAPI server URL

def get_dataset_list():
    """
    list of datasets
    """
    print('getting the list of dataset')
    response = requests.get(f"{BASE_URL}/dataset/list")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch task status: {response.status_code}, {response.text}")
        return None

def get_dataset_id_list():
    """
    list of datasets
    """
    print('getting the list of dataset')
    response = requests.get(f"{BASE_URL}/dataset/id-list")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch task status: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    result = get_dataset_list()
    if result is None:
        print("Failed to get dataset list.")
    else:
        for item in result:
            print(item)

    result = get_dataset_id_list()
    if result is None:
        print("Failed to get dataset id list.")
    else:
        for item in result:
            print(item)

    