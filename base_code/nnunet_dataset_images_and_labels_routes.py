from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pathlib import Path
import os, re
import json
import shutil

from fastapi.responses import FileResponse, JSONResponse

# FastAPI Router
router = APIRouter()

# Config and paths
from config import get_config
nnunet_data_dir = get_config()["data_dir"] 
nnunet_raw_dir = os.path.join(nnunet_data_dir, 'raw')

import nnunet_raw 

@router.get("/dataset/image_name_list")
async def get_image_name_list(dataset_id: str):
    """
    Returns lists of training/test image and label filenames and their extracted IDs.
    """
    try:
        return await nnunet_raw.get_image_name_list(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get the image name list: {str(e)}")

@router.post("/dataset/add_image_and_labels")
async def add_image_and_labels(
    dataset_id: str = Form(...),
    images_for: str = Form(...),  # "train" or "test"
    base_image: UploadFile = File(...),
    labels: UploadFile = File(...)
):
    """
    Uploads a base image and label mask to the correct dataset folder.

    - `dataset_id`: Name of the dataset (e.g., "Dataset001_Sample").
    - `images_for`: "train" or "test".
    - `base_image`: The main CT scan image.
    - `labels`: The corresponding label mask.
    """
    # Validate dataset directory
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    
    if not os.path.exists(dataset_path) or not os.path.exists(dataset_json_path):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    
    # Load dataset.json
    try:
        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset.json: {str(e)}")
    
    # Validate required keys in dataset.json
    required_keys = ["numTraining", "numTest", "file_ending"]
    for key in required_keys:
        if key not in dataset_info:
            raise HTTPException(status_code=400, detail=f"Missing key '{key}' in dataset.json")

    file_ending = dataset_info["file_ending"]  # Example: ".nii.gz"

    # Determine target folders and naming
    if images_for == "train":
        num = dataset_info["numTraining"]
        images_folder = os.path.join(dataset_path, "imagesTr")
        labels_folder = os.path.join(dataset_path, "labelsTr")
        dataset_info["numTraining"] += 1  # Increment counter
    elif images_for == "test":
        num = dataset_info["numTest"]
        images_folder = os.path.join(dataset_path, "imagesTs")
        labels_folder = os.path.join(dataset_path, "labelsTs")
        dataset_info["numTest"] += 1  # Increment counter
    else:
        raise HTTPException(status_code=400, detail="Invalid images_for value. Must be 'train' or 'test'.")

    # Ensure target directories exist
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Construct filenames
    base_image_path = os.path.join(images_folder, f"image_{num}_0000{file_ending}")
    labels_path = os.path.join(labels_folder, f"image_{num}{file_ending}")

    # find num, which has not been used yet
    while os.path.exists(base_image_path) or os.path.exists(labels_path):
        num+=1
        base_image_path = os.path.join(images_folder, f"image_{num}_0000{file_ending}")
        labels_path = os.path.join(labels_folder, f"image_{num}{file_ending}")

    # Save files
    try:
        with open(base_image_path, "wb") as buffer:
            shutil.copyfileobj(base_image.file, buffer)
        
        with open(labels_path, "wb") as buffer:
            shutil.copyfileobj(labels.file, buffer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    # Save updated dataset.json
    try:
        with open(dataset_json_path, "w") as f:
            json.dump(dataset_info, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update dataset.json: {str(e)}")

    if "id" not in dataset_info:
        dataset_info['id'] = dataset_id
        
    return dataset_info  # Return updated dataset.json


import glob


def find_matching_image_file(folder, num, file_ending):
    """
    Finds a file in the folder with pattern {header}_{num}_{subnumber}{file_ending}
    where `num` is matched exactly as 3, 03, 003, or 0003 (not 33, etc).
    """
    all_files = glob.glob(os.path.join(folder, f"*_*_*{file_ending}"))

    # Build regex: match *_<num>_* ending with the given file_ending
    # Accepts zero-padded forms like 3, 03, 003, 0003
    padded_nums = [f"{num:0{w}d}" for w in range(1, 5)]
    pattern = re.compile(rf"^.+_({'|'.join(padded_nums)})_\d+{re.escape(file_ending)}$")

    for file in all_files:
        filename = os.path.basename(file)
        if pattern.match(filename):
            return file

    raise HTTPException(status_code=404, detail=f"image file not found for num={num} in the folder {folder}")


from fastapi.responses import FileResponse, JSONResponse
from fastapi import Query

@router.get("/dataset/get_image_and_labels")
async def get_image_and_labels(
    dataset_id: str = Query(...),
    images_for: str = Query(...),  # "train" or "test"
    num: int = Query(...)
):
    """
    Returns paths to the base image and label file for a specific image index.

    - `dataset_id`: Name of the dataset (e.g., "Dataset001_Sample").
    - `images_for`: "train" or "test".
    - `num`: Index number of the image/label pair.
    """
    # Validate dataset directory
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    
    if not os.path.exists(dataset_path) or not os.path.exists(dataset_json_path):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    
    # Load dataset.json
    try:
        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset.json: {str(e)}")

    file_ending = dataset_info.get("file_ending", ".mha")

    if images_for == "train":
        images_folder = os.path.join(dataset_path, "imagesTr")
        labels_folder = os.path.join(dataset_path, "labelsTr")
    elif images_for == "test":
        images_folder = os.path.join(dataset_path, "imagesTs")
        labels_folder = os.path.join(dataset_path, "labelsTs")
    else:
        raise HTTPException(status_code=400, detail="Invalid images_for value. Must be 'train' or 'test'.")

    # Construct filenames
    #base_image_filename = f"image_{num}_0000{file_ending}"
    #labels_filename = f"image_{num}{file_ending}"

    #base_image_path = os.path.join(images_folder, base_image_filename)
    #labels_path = os.path.join(labels_folder, labels_filename)

    # Locate files flexibly
    base_image_path = find_matching_image_file(images_folder, num, file_ending)
    filename = os.path.basename(base_image_path)
    elements = filename.split('_')
    labels_filename = f'{elements[0]}_{elements[1]}{file_ending}'
    labels_path = os.path.join(labels_folder, labels_filename)
    
    if not os.path.exists(base_image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {base_image_path}")
    if not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail=f"Label file not found: {labels_path}")

    return JSONResponse({
        "base_image_filename": os.path.basename(base_image_path),
        "base_image_url": f"/dataset/download_file?dataset_id={dataset_id}&images_for={images_for}&type=image&num={num}",
        "labels_filename": os.path.basename(labels_path),
        "labels_url": f"/dataset/download_file?dataset_id={dataset_id}&images_for={images_for}&type=label&num={num}"
    })


@router.get("/dataset/download_file")
async def download_file(
    dataset_id: str,
    images_for: str,
    type: str,  # "image" or "label"
    num: int
):
    """
    Downloads a single image or label file.
    """
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")

    if not os.path.exists(dataset_json_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)

    file_ending = dataset_info.get("file_ending", ".nii.gz")

    if images_for == "train":
        images_folder = os.path.join(dataset_path, "imagesTr")
        labels_folder = os.path.join(dataset_path, "labelsTr")
    elif images_for == "test":
        images_folder = os.path.join(dataset_path, "imagesTs")
        labels_folder = os.path.join(dataset_path, "labelsTs")
    else:
        raise HTTPException(status_code=400, detail="Invalid images_for value.")

    if type == "image":
        path = find_matching_image_file(images_folder, num, file_ending)
    elif type == "label":
        base_image_path = find_matching_image_file(images_folder, num, file_ending)
        filename = os.path.basename(base_image_path)
        elements = filename.split('_')
        labels_filename = f'{elements[0]}_{elements[1]}{file_ending}'
        path = os.path.join(labels_folder, labels_filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid type (must be 'image' or 'label').")

    filename = os.path.basename(path)
    return FileResponse(path, filename=filename)


@router.put("/dataset/update_image_and_labels")
async def update_image_and_labels(
    dataset_id: str = Form(...),
    images_for: str = Form(...),
    num: int = Form(...),
    base_image: UploadFile = File(...),
    labels: UploadFile = File(...)
):
    """
    Updates existing image and label files for a given dataset and index.
    """
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")

    if not os.path.exists(dataset_json_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)

    file_ending = dataset_info.get("file_ending", ".mha")

    if images_for == "train":
        images_folder = os.path.join(dataset_path, "imagesTr")
        labels_folder = os.path.join(dataset_path, "labelsTr")
    elif images_for == "test":
        images_folder = os.path.join(dataset_path, "imagesTs")
        labels_folder = os.path.join(dataset_path, "labelsTs")
    else:
        raise HTTPException(status_code=400, detail="Invalid images_for value.")

    # Find matching base image filename
    base_image_path = find_matching_image_file(images_folder, num, file_ending)
    if not os.path.exists(base_image_path):
        raise HTTPException(status_code=500, detail=f"Failed to update files. image not found for num={num} in dataset {dataset_id}")

    parts = os.path.basename(base_image_path).split('_')
    label_path = os.path.join(labels_folder, f"{parts[0]}_{parts[1]}{file_ending}")
    try:
        with open(base_image_path, "wb") as buffer:
            shutil.copyfileobj(base_image.file, buffer)

        with open(label_path, "wb") as buffer:
            shutil.copyfileobj(labels.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update files: {str(e)}")

    return {"image_file": os.path.basename(base_image_path), 
            "label_file": os.path.basename(label_path), 
            "message": "Image and label updated successfully."}

@router.delete("/dataset/delete_image_and_labels")
async def delete_image_and_labels(
    dataset_id: str,
    images_for: str,
    num: int
):
    """
    Deletes image and label files for a given dataset and index.
    """
    dataset_path = os.path.join(nnunet_raw_dir, dataset_id)
    dataset_json_path = os.path.join(dataset_path, "dataset.json")

    if not os.path.exists(dataset_json_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)

    file_ending = dataset_info.get("file_ending", ".mha")

    if images_for == "train":
        images_folder = os.path.join(dataset_path, "imagesTr")
        labels_folder = os.path.join(dataset_path, "labelsTr")
        dataset_info["numTraining"] = max(0, dataset_info["numTraining"] - 1)
    elif images_for == "test":
        images_folder = os.path.join(dataset_path, "imagesTs")
        labels_folder = os.path.join(dataset_path, "labelsTs")
        dataset_info["numTest"] = max(0, dataset_info["numTest"] - 1)
    else:
        raise HTTPException(status_code=400, detail="Invalid images_for value.")

    # Find matching base image filename
    base_image_path = find_matching_image_file(images_folder, num, file_ending)
    if not os.path.exists(base_image_path):
        raise HTTPException(status_code=404, detail=f"No matching image file found for (dataset_id={dataset_id}, images_for={images_for}, num={num}).")

    base_filename = os.path.basename(base_image_path)
    parts = base_filename.split('_')
    if len(parts) < 3:
        raise HTTPException(status_code=500, detail=f"Filename format is invalid: {base_filename}")
    label_filename = f"{parts[0]}_{parts[1]}{file_ending}"
    label_path = os.path.join(labels_folder, label_filename)
    if not os.path.exists(label_path):
        raise HTTPException(status_code=404, detail=f"No matching label file found for (dataset_id={dataset_id}, images_for={images_for}, num={num}).")

    # Delete both files
    deleted = []
    for path in [base_image_path, label_path]:
        if os.path.exists(path):
            print(f'deleting file: {path}')
            os.remove(path)
            deleted.append(os.path.basename(path))

    if not deleted:
        raise HTTPException(status_code=404, detail="No matching files found to delete.")

    # Save updated dataset.json
    try:
        with open(dataset_json_path, "w") as f:
            json.dump(dataset_info, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update dataset.json: {str(e)}")

    if "id" not in dataset_info:
        dataset_info['id'] = dataset_id

    return {"files_deleted": deleted, 'dataset_json': dataset_info }

import asyncio

if __name__ == '__main__':
    dataset_id = 'Dataset009_Spleen'
    images_for = 'train'
    num = 3

    # Call the async function using asyncio
    asyncio.run(get_image_and_labels(dataset_id, images_for, num))
