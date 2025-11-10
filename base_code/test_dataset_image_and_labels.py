import requests


def get_image_and_labels(dataset_id, images_for, num, out_dir):
    url = "http://127.0.0.1:8000/dataset/get_image_and_labels"
    params = {
        "dataset_id": dataset_id,
        "images_for": images_for,
        "num": num
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        result = response.json()
        print("Image filename:", result["base_image_filename"])
        print("Label filename:", result["labels_filename"])
        print("Image download URL:", result["base_image_url"])
        print("Label download URL:", result["labels_url"])

        if not os.path.exists(out_dir):
            print(f'makedirs({out_dir})')
            os.makedirs(out_dir)

        # Download base image
        import os
        img_response = requests.get(f"http://127.0.0.1:8000{result['base_image_url']}")
        if img_response.status_code == 200:
            image_path = os.path.join(out_dir, 'image.mha')
            print(f'saving image: {image_path}')
            with open(image_path, "wb") as f:
                f.write(img_response.content)

        # Download labels
        lbl_response = requests.get(f"http://127.0.0.1:8000{result['labels_url']}")
        if lbl_response.status_code == 200:
            label_path = os.path.join(out_dir, 'label.mha')
            print(f'saving label: {label_path}')
            with open("label.mha", "wb") as f:
                f.write(lbl_response.content)

    else:
        print(f"Error {response.status_code}: {response.text}")

def get_image_name_list(dataset_id):
    url = "http://127.0.0.1:8000/dataset/image_name_list"
    params = {"dataset_id": dataset_id}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        print("Training Images:")
        for item in data["train_images"]:
            print(f"  NUM: {item['num']:>3}, NUM2: {item['num2']:>3}, Image Filename: {item['filename']}")

        print("\nTraining Labels:")
        for item in data["train_labels"]:
            print(f"  NUM: {item['num']:>3}, Label Filename: {item['filename']}")

        print("\nTest Images:")
        for item in data["test_images"]:
            print(f"  NUM: {item['num']:>3}, NUM2: {item['num2']:>3}, Image Filename: {item['filename']}")

        print("\nTest Labels:")
        for item in data["test_labels"]:
            print(f"  NUM: {item['num']:>3}, Label Filename: {item['filename']}")

        return response
    else:
        print(f"Error {response.status_code}: {response.text}")
        return response

def post_image_and_labels():
    import requests
    import os

    url = "http://127.0.0.1:8000/dataset/add_image_and_labels"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current script directory:", script_dir)

    image_path = os.path.join(script_dir, "_test_images/base_image.mha")
    labels_path = os.path.join(script_dir, "_test_images/labels.mha")

    # Required metadata
    dataset_id = "Dataset935_Test1"
    images_for = "train"  # Must be "train" or "test"


    # Open files safely using 'with' to avoid leaks
    with open(image_path, "rb") as img_file, open(labels_path, "rb") as lbl_file:
        files = {
            "base_image": img_file,
            "labels": lbl_file,
        }
        data = {
            "dataset_id": dataset_id,
            "images_for": images_for,  # Renamed field
        }
        
        response = requests.post(url, files=files, data=data)

    # Print response with error handling
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

def update_image_and_labels():
    import requests
    import os

    url = "http://127.0.0.1:8000/dataset/update_image_and_labels"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current script directory:", script_dir)

    image_path = os.path.join(script_dir, "_test_images/base_image.mha")
    labels_path = os.path.join(script_dir, "_test_images/labels.mha")

    # Required metadata
    dataset_id = "Dataset935_Test1"
    images_for = "train"  # Must be "train" or "test"


    # Open files safely using 'with' to avoid leaks
    with open(image_path, "rb") as img_file, open(labels_path, "rb") as lbl_file:
        files = {
            "base_image": img_file,
            "labels": lbl_file,
        }
        data = {
            "dataset_id": dataset_id,
            "images_for": images_for,  # Renamed field
        }
        
        response = requests.post(url, files=files, data=data)

    # Print response with error handling
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == '__main__':

    #dataset_id = 'Dataset009_Spleen'
    #num = 3

    dataset_id = "Dataset935_Test1"
    images_for = 'train'

    # get list of image names
    response = get_image_name_list(dataset_id)
    
    if response.status_code == 200:
        data = response.json()
        item = data["train_images"][0]
        num = item['num'] 
        #num2 = item['num2']
        filename =  item['filename']
        get_image_and_labels(dataset_id, images_for, num)

    post_image_and_labels()
