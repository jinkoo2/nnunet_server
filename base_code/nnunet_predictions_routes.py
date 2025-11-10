from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Query
import os, re, json, shutil, zipfile
from datetime import datetime
from json import JSONDecodeError
from logger import logger, log_exception, log_request

router = APIRouter()

from config import get_config
nnunet_data_dir = get_config()["data_dir"]
nnunet_predictions_dir = os.path.join(nnunet_data_dir, 'predictions')
import nnunet_raw

def create_unique_request_dir(dataset_path: str) -> str:
    os.makedirs(dataset_path, exist_ok=True)
    for num in range(1000):
        req_dir = os.path.join(dataset_path, f"req_{num:03}")
        if not os.path.exists(req_dir):
            os.makedirs(req_dir)
            return req_dir
    raise RuntimeError("Too many requests: could not create a unique request directory.")

async def get_file_ending(dataset_id):
    try:
        dataset_info = await nnunet_raw.read_dataset_json(dirname=dataset_id)
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to read dataset.json: {str(e)}")

    if "file_ending" not in dataset_info:
        logger.warning(f"nnunet_raw: dataset.json for {dataset_id} missing 'file_ending'")
        raise HTTPException(status_code=400, detail="Missing 'file_ending' in dataset.json")

    return dataset_info["file_ending"]

async def get_input_channel_names(dataset_id):
    try:
        dataset_info = await nnunet_raw.read_dataset_json(dirname=dataset_id)
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to read dataset.json: {str(e)}")

    if "channel_names" not in dataset_info:
        logger.warning(f"nnunet_raw: dataset.json for {dataset_id} missing 'channel_names'")
        raise HTTPException(status_code=400, detail="Missing 'channel_names' in dataset.json")

    return list(dataset_info["channel_names"].values())
    
@router.get("/predictions")
async def get_predictions_list(dataset_id: str, request: Request):
    
    log_request(request)
    logger.info(f"GET /predictions called with dataset_id={dataset_id}")

    results = []

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)

    if not os.path.exists(dataset_path):
        logger.warning(f"Predictions folder for dataset '{dataset_id}' not found at: {dataset_path}")
        return results

    # file_ending (.mha)
    file_ending = await get_file_ending(dataset_id)
    
    for entry in sorted(os.listdir(dataset_path)):
        req_dir = os.path.join(dataset_path, entry)
        if not os.path.isdir(req_dir) or not entry.startswith("req_"):
            logger.debug(f"Skipping non-request directory: {req_dir}")
            continue

        item = {"req_id": entry}

        # Load req.json
        req_json_path = os.path.join(req_dir, "req.json")
        if os.path.exists(req_json_path):
            try:
                with open(req_json_path, "r") as f:
                    item["req_info"] = json.load(f)
            except (JSONDecodeError, OSError) as e:
                log_exception(e)
                item["req_info"] = {}
                continue
        else:
            logger.warning(f"Missing req.json in {entry}")
            item["req_info"] = {}
            continue

        # List input images
        item["input_images"] = sorted([
            fname for fname in os.listdir(req_dir)
            if fname.startswith("image_") and fname.endswith(file_ending)
        ])

        # Check for outputs
        expected_output_label_images = [f'image_{fname.split("_")[1]}{file_ending}' for fname in item["input_images"]]
        completed = True
        outputs_dir = os.path.join(req_dir, "outputs")
        output_labels = []
        for expected_output_label in expected_output_label_images:
            output_label_path = os.path.join(outputs_dir, expected_output_label)
            if not os.path.exists(output_label_path):
                completed = False
                break
            else:
                output_labels.append(expected_output_label)

        item["completed"] = completed
        item["output_labels"] = sorted(output_labels)

        results.append(item)

    return results

@router.get("/prediction")
async def get_prediction(dataset_id: str = Query(...), req_id: str = Query(...), request: Request = None):
    log_request(request)
    logger.info(f"GET /prediction called with dataset_id={dataset_id}, req_id={req_id}")

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)
    req_dir = os.path.join(dataset_path, req_id)

    if not os.path.isdir(req_dir):
        raise HTTPException(status_code=404, detail=f"Request directory not found: {req_dir}")

    item = {"req_id": req_id}

    # Load req.json
    req_json_path = os.path.join(req_dir, "req.json")
    if os.path.exists(req_json_path):
        try:
            with open(req_json_path, "r") as f:
                item["req_info"] = json.load(f)
        except (JSONDecodeError, OSError) as e:
            log_exception(e)
            item["req_info"] = {}
    else:
        logger.warning(f"Missing req.json in {req_dir}")
        item["req_info"] = {}

    # file_ending (.mha)
    file_ending = await get_file_ending(dataset_id)

    # List input images
    item["input_images"] = sorted([
        fname for fname in os.listdir(req_dir)
        if fname.startswith("image_") and fname.endswith(file_ending)
    ])

    # Check outputs
    expected_output_label_images = [f'image_{fname.split("_")[1]}{file_ending}' for fname in item["input_images"]]
    outputs_dir = os.path.join(req_dir, "outputs")
    output_labels = []
    completed = True
    for expected_output_label in expected_output_label_images:
        output_label_path = os.path.join(outputs_dir, expected_output_label)
        if not os.path.exists(output_label_path):
            completed = False
            break
        output_labels.append(expected_output_label)

    item["completed"] = completed
    item["output_labels"] = sorted(output_labels)

    logger.info(f"Returning item={item}")

    return item

@router.post("/predictions")
async def post_prediction_request(
    request: Request,
    dataset_id: str = Form(...),
    requester_id: str = Form(...),
    image_id: str = Form(...),
    image: UploadFile = File(...),
):
    
    log_request(request)
    logger.info(
        f"POST /predictions called with dataset_id={dataset_id}, "
        f"requester_id={requester_id}, image_id={image_id}, "
        f"filename={image.filename}"
    )

    form_data = await request.form()

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)

    try:
        req_dir = create_unique_request_dir(dataset_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # file_ending
        file_ending = await get_file_ending(dataset_id)

        # Save image file
        image_path = os.path.join(req_dir, f'image_0_0000{file_ending}')
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Save request metadata
        req = {
            "requester_id": requester_id,
            "image_id_list": [image_id],
            "req_id": os.path.basename(req_dir),
            "at": datetime.now().isoformat()
        }
        known_keys = {"dataset_id", "requester_id", "image_id"}
        for key, value in form_data.items():
            if key not in known_keys and isinstance(value, str):
                req[key] = value

        req_path = os.path.join(req_dir, "req.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=4)

        return req

    except Exception as e:
        # Cleanup request folder on error
        if os.path.exists(req_dir):
            shutil.rmtree(req_dir)
        raise HTTPException(status_code=500, detail=f"Error processing prediction request: {str(e)}")
    
@router.post("/predictions_zip")
async def post_prediction_request(
    request: Request,
    dataset_id: str = Form(...),
    requester_id: str = Form(...),
    image_id_list: str = Form(...),
    images_zip: UploadFile = File(...),
):
    log_request(request)
    logger.info(
        f"POST /predictions_zip called with dataset_id={dataset_id}, "
        f"requester_id={requester_id}, image_count={len(image_id_list.split('|'))}, "
        f"zip_name={images_zip.filename}"
    )

    form_data = await request.form()

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)

    # Create request folder
    try:
        req_dir = create_unique_request_dir(dataset_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # file_ending
        file_ending = await get_file_ending(dataset_id)

        # Save zip file
        images_zip_path = os.path.join(req_dir, 'images.zip')
        with open(images_zip_path, "wb") as buffer:
            shutil.copyfileobj(images_zip.file, buffer)

        # Extract zip
        with zipfile.ZipFile(images_zip_path, "r") as zip_ref:
            filename_list = sorted(zip_ref.namelist())
            zip_ref.extractall(req_dir)

        # Validate image_id_list length
        image_ids = image_id_list.split('|')
        if len(filename_list) != len(image_ids):
            raise HTTPException(status_code=400, detail="Mismatch between number of images and image_id_list")

        # Rename files
        for i, filename in enumerate(filename_list):
            src = os.path.join(req_dir, filename)
            dst = os.path.join(req_dir, f"image_{i}_0000{file_ending}")
            os.rename(src, dst)

        # Save request metadata
        req = {
            "requester_id": requester_id,
            "image_id_list": image_ids,
            "req_id": os.path.basename(req_dir),
            "at": datetime.now().isoformat()
        }
        known_keys = {"dataset_id", "requester_id", "image_id_list"}
        for key, value in form_data.items():
            if key not in known_keys and isinstance(value, str):
                req[key] = value

        req_path = os.path.join(req_dir, "req.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=4)

        return {"status": "success", "req": req}

    except Exception as e:
        # Cleanup request folder on error
        if os.path.exists(req_dir):
            shutil.rmtree(req_dir)
        raise HTTPException(status_code=500, detail=f"Error processing prediction request: {str(e)}")


@router.delete("/predictions")
async def delete_prediction_request(dataset_id: str, req_id: str, request: Request):
    log_request(request)
    logger.info(f"DELETE /predictions called with dataset_id={dataset_id}, req_id={req_id}")

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)
    req_dir = os.path.join(dataset_path, req_id)

    if not os.path.exists(req_dir):
        logger.warning(f"Request directory not found: {req_dir}")
        raise HTTPException(status_code=404, detail=f"Request '{req_id}' not found in dataset '{dataset_id}'")

    try:
        shutil.rmtree(req_dir)
        logger.info(f"Deleted request directory: {req_dir}")
        return {"status": "success", "message": f"Request '{req_id}' deleted."}
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to delete request '{req_id}': {str(e)}")


@router.get("/predictions/image_and_label_metadata")
async def get_image_label_metadata(
    dataset_id: str = Query(...),
    req_id: str = Query(...),
    image_number: int = Query(...)
):
    try:
        file_ending = await get_file_ending(dataset_id)
        ch_names = await get_input_channel_names(dataset_id)
        
        image_names = [f"image_{image_number}_{i:04}{file_ending}" for i in range(len(ch_names))]
        label_name = f"image_{image_number}{file_ending}"
        
        return {
            "image_names": image_names,
            "label_name": label_name,
            "download_url": f"/predictions/download_images_and_label_files?dataset_id={dataset_id}&req_id={req_id}&image_number={image_number}"
        }
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")



import json

def load_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    

@router.get("/predictions/contour_points")
async def get_contour_points(
    dataset_id: str = Query(...),
    req_id: str = Query(...),
    image_number: int = Query(...),
    contour_number: int = Query(...),
    coordinate_systems: str = Query("woI")
):
    import dict_helper
    import image_tools  # assumes extract_binary_label_image and binary_image_to_contour_list_json_files are defined here

    try:
        # Build paths
        dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"dataset_path not found: {dataset_path}")

        req_path = os.path.join(dataset_path, req_id)
        if not os.path.exists(req_path):
            raise HTTPException(status_code=404, detail=f"req_path not found: {req_path}")

        outputs_dir = os.path.join(req_path, "outputs")
        if not os.path.exists(outputs_dir):
            raise HTTPException(status_code=404, detail=f"outputs_dir not found: {outputs_dir}")

        # Load dataset and label info
        file_ending = await get_file_ending(dataset_id)
        dataset_json_path = os.path.join(outputs_dir, "dataset.json")
        dataset = dict_helper.load_from_json(dataset_json_path)

        labels_map = dataset.get("labels")
        if not labels_map or len(labels_map) < 2:
            raise HTTPException(status_code=400, detail="Invalid 'labels' in dataset.json. Must contain at least 2 label entries.")

        # ✅ Check if contour_number is in label_map values
        if contour_number not in labels_map.values():
            raise HTTPException(status_code=400, detail=f"Contour number {contour_number} is not in label map.")

        # Validate coordinate_systems
        valid_coords = {'w', 'o', 'I'}
        invalid = set(coordinate_systems) - valid_coords
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid coordinate system(s): {', '.join(invalid)}. Allowed values are: w, o, I"
            )

        # Label and binary image paths
        label_image_path = os.path.join(outputs_dir, f"image_{image_number}.mha")
        if not os.path.exists(label_image_path):
            raise HTTPException(status_code=404, detail=f"Label image not found: {label_image_path}")

        binary_image_fname = f"image_{image_number}{file_ending}.{contour_number}.mha"
        binary_image_file = os.path.join(outputs_dir, binary_image_fname)

        # Generate binary label image if missing
        if not os.path.exists(binary_image_file):
            image_tools.extract_binary_label_image(label_image_path, contour_number, binary_image_file)

        # Coordinate-to-file mapping
        coord_map = {
            'w': 'points_w',
            'o': 'points_o',
            'I': 'points_I',
        }
        selected_coords = {k: v for k, v in coord_map.items() if k in coordinate_systems}
        print(f'selected_coords={selected_coords}')
        contour_paths = {
            key: os.path.join(outputs_dir, f"{binary_image_fname}.{key}.json")
            for key in selected_coords.values()
        }
        print(f'contour_paths={contour_paths}')


        # Generate contour .json files if any are missing
        if any(not os.path.exists(p) for p in contour_paths.values()):
            image_tools.binary_image_to_contour_list_json_files(binary_image_file)

        # Load and return selected coordinate outputs
        return {
            key: load_from_json(path)
            for key, path in contour_paths.items()
        }

    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")



        
from fastapi.responses import FileResponse
from fastapi import Query
import tempfile
import zipfile

@router.get("/predictions/download_images_and_label_files")
async def download_image_and_label(
    dataset_id: str = Query(...),
    req_id: str = Query(...),
    image_number: int = Query(...),
    request: Request = None
):
    log_request(request)
    logger.info(f"GET /download_image_label called with dataset_id={dataset_id}, req_id={req_id}, image_number={image_number}")

    dataset_path = os.path.join(nnunet_predictions_dir, dataset_id)
    req_dir = os.path.join(dataset_path, req_id)
    outputs_dir = os.path.join(req_dir, "outputs")

    if not os.path.exists(req_dir):
        logger.warning(f"Request folder not found: {req_dir}")
        raise HTTPException(status_code=404, detail="Request folder not found.")

    try:
        file_ending = await get_file_ending(dataset_id)

        ch_names = await get_input_channel_names(dataset_id)
        
        # image names & paths
        image_names = [f"image_{image_number}_{image_number2:04}{file_ending}" for image_number2,_ in enumerate(ch_names)]
        image_paths = [os.path.join(req_dir, image_name) for image_name in image_names]
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Input image not found: {image_path}")
                raise HTTPException(status_code=404, detail="Input image not found.")
        
        # label name lable path
        label_name = f"image_{image_number}{file_ending}"
        label_path = os.path.join(outputs_dir, label_name)
        label_exist = os.path.exists(label_path)        
        
        # Create a temp ZIP file with both images
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zipf:
                for image_path in image_paths:
                    zipf.write(image_path, arcname=os.path.basename(image_path))
                if label_exist:
                    zipf.write(label_path, arcname=os.path.basename(label_path))
            logger.info(f"Zipped image and label to: {tmp.name}")
            return FileResponse(
                tmp.name,
                media_type='application/zip',
                filename=f"{req_id}_image_{image_number}.zip"
            )
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail=f"Error preparing download: {str(e)}")


import asyncio

if __name__ == "__main__":
    result = asyncio.run(get_predictions_list("Dataset015_CBCTBladderRectumBowel2", request=None))
    print(result)