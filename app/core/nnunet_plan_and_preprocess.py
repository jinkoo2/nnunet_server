from pathlib import Path
from app.core.config import settings
from app.core.logging_config import get_logger
import subprocess
import os

# logger
logger = get_logger(__name__)

# nnU-Net directories
nnunet_data_dir = settings.NNUNET_DATA_DIR
nnunet_raw_dir = os.path.join(nnunet_data_dir, 'raw')
nnunet_preprocessed_dir = os.path.join(nnunet_data_dir, 'preprocessed')
nnunet_results_dir = os.path.join(nnunet_data_dir, 'results')

# Ensure directories exist
Path(nnunet_raw_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_preprocessed_dir).mkdir(parents=True, exist_ok=True)
Path(nnunet_results_dir).mkdir(parents=True, exist_ok=True)
logger.info(f"nnunet_data_dir={nnunet_data_dir}")

def plan_and_preprocess_slurm(dataset_num, planner, verify_dataset_integrity):
    logger.info(f'plan_and_preprocess_slurm(dataset_num={dataset_num},planner={planner},verify_dataset_integrity={verify_dataset_integrity})')

    # config params
    venv_dir = settings.venv_dir
    scripts_dir = settings.scripts_dir
    nnunet_dir = settings.nnunet_dir
    data_dir = settings.data_dir
    script_output_files_dir = settings.script_output_files_dir
    
    # template.sh
    tmplt_file = os.path.join(scripts_dir,'template.slurm')
    logger.info(f'making slurm file from template...{tmplt_file}')
    if not os.path.exists(tmplt_file):
        logger.error('template file not found:', tmplt_file) 
        exit(-1)

    #  output case dir
    case_dir = os.path.join(script_output_files_dir, f"{dataset_num:03}")
    logger.info(f'case_dir={case_dir}')
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # output script file
    script_file = os.path.join(case_dir, f'pp_{planner}.slurm')

    job_name = f'pp_ds{dataset_num}'
    # output log file
    log_file = script_file+'.log'

    # commandd line
    cmd_lines = f'nnUNetv2_plan_and_preprocess -d {dataset_num} -pl {planner} --verbose'
    if verify_dataset_integrity:
        cmd_lines = cmd_lines + ' --verify_dataset_integrity '

    # replace variables
    with open(tmplt_file) as f:
        txt = f.read()
    txt = txt.replace('{job_name}', job_name)
    txt = txt.replace('{log_file}', log_file)
    txt = txt.replace('{venv_dir}', venv_dir)
    txt = txt.replace('{data_dir}', data_dir)
    txt = txt.replace('{nnunet_dir}', nnunet_dir)
    txt = txt.replace('{cmd_lines}', cmd_lines)

    logger.info(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(txt)

    # sbatch
    cmd = f'module load slurm && sbatch {script_file}'
    logger.info(f'running "{cmd}"')
    subprocess.run(cmd, shell=True)

def plan_and_preprocess_sh(dataset_num, planner, verify_dataset_integrity):
    logger.info(f'plan_and_preprocess_sh(dataset_num={dataset_num}, planner={planner}, verify_dataset_integrity={verify_dataset_integrity})')

    scripts_dir = settings.scripts_dir
    data_dir = settings.data_dir
    script_output_files_dir = settings.script_output_files_dir

    tmplt_file = os.path.join(scripts_dir, 'template.sh')
    if not os.path.exists(tmplt_file):
        logger.error(f"Template file not found: {tmplt_file}")
        raise FileNotFoundError(f"Template file not found: {tmplt_file}")

    case_dir = os.path.join(script_output_files_dir, f"{dataset_num:03}")
    os.makedirs(case_dir, exist_ok=True)
    logger.info(f'case_dir={case_dir}')

    script_file = os.path.join(case_dir, f'pp_{planner}.sh')
    log_file = script_file + '.log'

    cmd_lines = f'nnUNetv2_plan_and_preprocess -d {dataset_num} -pl {planner} --verbose'
    if verify_dataset_integrity:
        cmd_lines += ' --verify_dataset_integrity'
    cmd_lines += f' 2>&1 | tee {log_file}'

    logger.info(f'cmd_lines={cmd_lines}')

    with open(tmplt_file) as f:
        txt = f.read()
    txt = txt.replace('{data_dir}', data_dir)
    txt = txt.replace('{cmd_lines}', cmd_lines)

    with open(script_file, 'w') as file:
        file.write(txt)
    logger.info(f'Saved script file: {script_file}')

    cmd = f'chmod +x "{script_file}" && bash "{script_file}"'
    logger.info(f'Running: {cmd}')

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script: {e}")


if __name__ == "__main__":

    #dataset_num = 9 # Dataset009_Spleen
    #dataset_num = 101 # Dataset101_Eye[ul]L
    #dataset_num = 102 # Dataset102_ProneLumpStessin
    #dataset_num = 103
    #dataset_num = 104
    #dataset_num = 105
    dataset_num = 847

    verify_dataset_integrity = True
    planner='ExperimentPlanner'
    #planner='nnUNetPlannerResEncM'
    #planner='nnUNetPlannerResEncL'
    #planner='nnUNetPlannerResEncXL'

    plan_and_preprocess_slurm(dataset_num, planner, verify_dataset_integrity)
    #plan_and_preprocess_sh(dataset_num, planner, verify_dataset_integrity)

    # sbatch
    print('done')



