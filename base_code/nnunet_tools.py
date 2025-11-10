import os, re

def find_image_files(folder: str, file_ending):
    if not os.path.exists(folder):
        return []

    file_list = [f for f in os.listdir(folder) if f.endswith(file_ending)]

    pattern = re.compile(r"^(.+)_(\d+)_(\d+)$")

    result = []
    for fname in file_list:
        fname_noext = fname.split('.')[0]
        match = pattern.match(fname_noext)
        if match:
            result.append({
                "filename": fname,
                "prefix": match.group(1),
                "num": int(match.group(2)),
                "num2": int(match.group(3)),
            })
    return sorted(result, key=lambda x: (x["num"], x["num2"]))


def find_label_files(folder: str, file_ending):
    if not os.path.exists(folder):
        return []

    file_list = [f for f in os.listdir(folder) if f.endswith(file_ending)]

    pattern = re.compile(r"^(.+)_(\d+)$")

    result = []
    for fname in file_list:
        try:
            fname_noext = fname.split('.')[0]
            match = pattern.match(fname_noext)
            if match:
                result.append({
                    "filename": fname,
                    "prefix": match.group(1),
                    "num": int(match.group(2))
                })
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        
    return sorted(result, key=lambda x: x["num"])

if __name__ == '__main__':

    from config import get_config
    nnunet_data_dir = get_config()["data_dir"] 
    print(f'nnunet_data_dir={nnunet_data_dir}')

    nnunet_raw_dir =  os.path.join(nnunet_data_dir,'raw')

    folder = os.path.join(nnunet_raw_dir, 'Dataset009_Spleen', 'imagesTr')
    for file in find_image_files(folder, '.nii.gz'):
        print(file)

    folder = os.path.join(nnunet_raw_dir, 'Dataset009_Spleen', 'labelsTr')
    for file in find_label_files(folder, '.nii.gz'):
        print(file)

