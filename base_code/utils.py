import os



def _join_dir(dir1, dir2):
    joined = os.path.join(dir1, dir2)
    if not os.path.exists(joined):
        os.makedirs(joined)
    return joined

def _error(msg):
    raise Exception(f'Error: msg')

def _info(msg):
    print(msg)

def _must_exist(path):
    if not os.path.exists(path):
        raise Exception('Paht not found:{path}')

def get_current_datetime_str():    
    from datetime import datetime

    # Get current date and time
    now = datetime.now()

    # Format the datetime as a folder-compatible string
    return now.strftime("%Y%m%d_%H%M%S")

def copy_file_to_folder(source_file, destination_folder):
    import shutil
    import os

    # Get the full destination path (including the file name)
    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    # Copy the file
    shutil.copy(source_file, destination_file)

import os

def get_unique_file_path(base_filename, directory):
    # Extract the base name and extension
    base_name, extension = os.path.splitext(base_filename)
    
    # Start with the original filename
    file_path = os.path.join(directory, base_filename)
    counter = 1

    # Loop until we find a filename that doesn't exist
    while os.path.exists(file_path):
        # Generate new filename with counter
        file_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
       
    return file_path



