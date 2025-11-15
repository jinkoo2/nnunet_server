import requests
import json
import os
import time
from pathlib import Path

# --- Configuration ---
# 1. API Endpoint URL
API_URL = "http://localhost:9000/predictions" 

# 2. Define the path to the actual image file
# IMPORTANT: Ensure this file exists at this exact path on the machine running this script.
IMAGE_FILE_PATH = "/home/jk/projects/nnunet_server/sample_data/bladder_cbct.mha"

# 3. Define the exact form data parameters required by the FastAPI endpoint.
TEST_DATA = {
    # Corresponds to your nnU-Net DATASET_ID "015"
    "dataset_id": "Dataset015_CBCTBladderRectumBowel2",
    
    # Corresponds to your nnU-Net requester_id 
    "requester_id": "vtk_image_labeler_3d@varianEclipseTest",
    
    # This ID is critical for the client to track its request.
    # We use the filename without extension as the image_id
    "image_id": Path(IMAGE_FILE_PATH).stem, 
}

def test_prediction_endpoint():
    """Submits a prediction request with form data and the specified file."""
    
    file_path = Path(IMAGE_FILE_PATH)

    # Pre-check if the file exists
    if not file_path.exists():
        print(f"\n[ERROR] File not found at specified path:")
        print(f"Path: {IMAGE_FILE_PATH}")
        print(f"Please check the file path and permissions.")
        return
    
    file_name = file_path.name
    
    print(f"\n--- Submitting request to {API_URL} ---")
    print(f"File to send: {IMAGE_FILE_PATH}")
    print(f"Form Data: {json.dumps(TEST_DATA, indent=4)}")

    # Open the file in binary mode
    uploaded_file = None
    try:
        # Prepare the file dictionary for requests.post
        # We use 'application/octet-stream' for general binary data like .mha
        uploaded_file = open(IMAGE_FILE_PATH, 'rb')
        files = {'image': (file_name, uploaded_file, 'application/octet-stream')}
        
        # The 'data' parameter holds the fields for Request, Dataset_ID, etc.
        response = requests.post(API_URL, data=TEST_DATA, files=files, timeout=60) # Added timeout

        print("\n--- Response ---")
        print(f"Status Code: {response.status_code}")
        
        # Check if the request succeeded
        if response.status_code == 200:
            print("Response Data (Job ID):")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Connection refused or host not found.")
        print(f"Ensure your API server (FastAPI) is running locally on port 9000.")
    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out. The server took too long to respond.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Crucial: ensure the file object is closed
        if uploaded_file:
             uploaded_file.close()

if __name__ == "__main__":
    test_prediction_endpoint()