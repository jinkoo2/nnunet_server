import os

from logger import logger

def plan_and_preprocess_slurm(dataset_num, planner, verify_dataset_integrity):
    logger.info(f'plan_and_preprocess_slurm(dataset_num={dataset_num},planner={planner},verify_dataset_integrity={verify_dataset_integrity})')

     # params
    from config import get_config
    config = get_config()
    venv_dir = config['venv_dir']
    scripts_dir = config['scripts_dir']
    nnunet_dir = config['nnunet_dir']
    data_dir = config['data_dir']
    script_output_files_dir = config['script_output_files_dir']
    
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
    import subprocess
    cmd = f'module load slurm && sbatch {script_file}'
    logger.info(f'running "{cmd}"')
    #subprocess.run(cmd, shell=True)

def plan_and_preprocess_sh(dataset_num, planner, verify_dataset_integrity):
    logger.info(f'plan_and_preprocess_sh(dataset_num={dataset_num},planner={planner},verify_dataset_integrity={verify_dataset_integrity})')
    # params
    from config import get_config
    config = get_config()
    venv_dir = config['venv_dir']
    scripts_dir = config['scripts_dir']
    nnunet_dir = config['nnunet_dir']
    data_dir = config['data_dir']
    script_output_files_dir = config['script_output_files_dir']
    
    # template.sh
    tmplt_file = os.path.join(scripts_dir,'template.sh')
    logger.info(f'making sh file from template...{tmplt_file}')
    if not os.path.exists(tmplt_file):
        print('template file not found:', tmplt_file) 
        exit(-1)

    #  output case dir
    case_dir = os.path.join(script_output_files_dir, f"{dataset_num:03}")
    logger.info(f'case_dir={case_dir}')
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # output script file
    script_file = os.path.join(case_dir, f'pp_{planner}.sh')

    # output log file
    log_file = script_file+'.log'

    # commandd line
    cmd_lines = f'nnUNetv2_plan_and_preprocess -d {dataset_num} -pl {planner} --verbose '
    if verify_dataset_integrity:
        cmd_lines = cmd_lines + ' --verify_dataset_integrity'
    cmd_lines = cmd_lines + f' 2>&1 | tee {log_file}'
    logger.info(f'cmd_lines={cmd_lines}')

    # replace variables
    with open(tmplt_file) as f:
        txt = f.read()
    txt = txt.replace('{venv_dir}', venv_dir)
    txt = txt.replace('{data_dir}', data_dir)
    txt = txt.replace('{nnunet_dir}', nnunet_dir)
    txt = txt.replace('{cmd_lines}', cmd_lines)

    logger.info(f'saving script file - {script_file}')
    with open(script_file, 'w') as file:
        file.write(txt)

    # run script
    import subprocess
    cmd = f'chmod +x {script_file} && {script_file}'
    logger.info(f'running "{cmd}"')
    subprocess.run(cmd, shell=True)

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



