import requests
import os
import zipfile

def zip_mha_files(source_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(source_dir):
            if filename.endswith('.mha'):
                file_path = os.path.join(source_dir, filename)
                zipf.write(file_path, arcname=filename)  # arcname avoids storing full path

def test_post_predictions_zip():
    url = "http://127.0.0.1:8000/predictions_zip"

    dataset_id = "Dataset847_FourCirclesOnJawCalKv2"
    requester_id = "tester_001"
    image_id_list = "image0|image1"

    zip_dir = os.path.join(os.path.dirname(__file__), "_test_images/temp")
    images_dir = os.path.join(os.path.dirname(__file__), "_test_images/predictions")
    zip_path = os.path.join(zip_dir, 'images.zip')

    os.makedirs(zip_dir, exist_ok=True)
    zip_mha_files(images_dir, zip_path)

    extra_fields = {
        "notes": "This is a test",
        "priority": "high",
        "name": "jinkoo kim",
        "inst": "stony brook"
    }

    with open(zip_path, "rb") as zip_file:
        form_data = {
            "dataset_id": dataset_id,
            "requester_id": requester_id,
            "image_id_list": image_id_list,
            **extra_fields
        }
        files = {
            "images_zip": ("images.zip", zip_file, "application/zip")
        }

        response = requests.post(url, data=form_data, files=files)

    if response.status_code == 200:
        print("Request succeeded:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

def test_post_predictions():
    url = "http://127.0.0.1:8000/predictions"

    dataset_id = "Dataset847_FourCirclesOnJawCalKv2"
    requester_id = "tester_001"
    image_id = "image0"

    
    images_dir = os.path.join(os.path.dirname(__file__), "_test_images/predictions")
    image_path = os.path.join(images_dir, '0.mha')

    extra_fields = {
        "notes": "This is a test for 1 image prediction",
        "priority": "high",
        "name": "jinkoo kim",
        "inst": "stony brook"
    }

    with open(image_path, "rb") as image_file:
        form_data = {
            "dataset_id": dataset_id,
            "requester_id": requester_id,
            "image_id": image_id,
            **extra_fields
        }
        files = {
            "image": image_file
        }

        response = requests.post(url, data=form_data, files=files)

    if response.status_code == 200:
        print("Request succeeded:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

def test_get_contour_points():
    base_url = "http://127.0.0.1:8000"
    endpoint = "/predictions/contour_points"

    # Example parameters (modify to match your test case)
    dataset_id = "Dataset015_CBCTBladderRectumBowel2"
    req_id = "req_034"
    image_number = 0
    contour_number = 1
    coordinate_systems = "woI"  # or "wI", "o", etc.

    params = {
        "dataset_id": dataset_id,
        "req_id": req_id,
        "image_number": image_number,
        "contour_number": contour_number,
        "coordinate_systems": coordinate_systems
    }

    response = requests.get(base_url + endpoint, params=params)

    if response.status_code == 200:
        print("Request succeeded.")
        result = response.json()
        for k, v in result.items():
            print(f"{k}: {len(v)} point(s)")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    #test_post_predictions_zip()
    #test_post_predictions()
    test_get_contour_points()