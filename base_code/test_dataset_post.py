import requests
import time

BASE_URL = "http://127.0.0.1:8000"  # FastAPI server URL


def post_dataset(data):
    """
    post a dataset
    """
    response = requests.post(
        f"{BASE_URL}/dataset/new",
        json=data,  # Task input as JSON payload
        headers={"Content-Type": "application/json"}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to add a dataset: {response.status_code}, {response.text}")
        return None
    
if __name__ == "__main__":

    dataset = {
        "name": "NewName1",
        "description": "New description",
        "reference": "Your institution",
        "licence": "CC-BY-SA 4.0",
        "tensorImageSize": "3D",
        "file_ending": ".mha",
        "labels": {
            "background": 0,
            "label1": 1
        },
        "channel_names": {
            "0": "CT"
            }
        }
    
    result = post_dataset(dataset)
    if result is None:
        print("Failed to post.")
    else:
        print(result)
