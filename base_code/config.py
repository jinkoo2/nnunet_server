import os

#################
# ENV load .env file
from dotenv import load_dotenv
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path and os.path.exists(dotenv_path):
    print(f'loading .env file:{dotenv_path}')
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f'running load_dotenv()')
    load_dotenv()

from utils import _error, _info, _join_dir

def get_config():
   
    home_dir = os.getenv('home_dir')
    venv_dir = os.getenv('venv_dir')
    scripts_dir = os.getenv('scripts_dir')
    nnunet_dir = os.getenv('nnunet_dir')
    data_dir = os.getenv('data_dir')
    log_dir = os.getenv('log_dir')

    if not os.path.exists(scripts_dir):
        _error(f'scripts_dir not found:{scripts_dir}')

    if not os.path.exists(nnunet_dir):
        _error(f'nnunet_dir not found:{nnunet_dir}')

    if not os.path.exists(data_dir):
        _error(f'data_dir not found:{data_dir}')

    raw_dir = _join_dir(data_dir, 'raw')
    preprocessed_dir= _join_dir(data_dir, 'preprocessed')
    results_dir = _join_dir(data_dir, 'results')

    slurm_files_dir = _join_dir(data_dir, 'slurm_files')
    sh_files_dir = _join_dir(data_dir, 'sh_files')
    #script_output_files_dir = _join_dir(data_dir, 'script_output_files')
    script_output_files_dir = _join_dir(scripts_dir, 'script_output_files')

    ret = {
        'home_dir': home_dir,
        'venv_dir': venv_dir,
        'scripts_dir': scripts_dir,
        'nnunet_dir': nnunet_dir,
        'data_dir': data_dir,
        'raw_dir': raw_dir,
        'preprocessed_dir': preprocessed_dir,
        'results_dir': results_dir,
        'slurm_files_dir': slurm_files_dir,
        'sh_files_dir': sh_files_dir,
        'script_output_files_dir': script_output_files_dir,
        'log_dir': log_dir
    }

    #print('get_config().return=', ret)

    return ret
