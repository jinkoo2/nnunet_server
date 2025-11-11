#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p a100-large
#SBATCH --gres=gpu:1            # number of GPUs per node (gres=gpu:N)
#SBATCH --gpus=1 # number of GPUs per node (gres=gpu:N), this is a backup of --gres
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jinkoo.kim@stonybrook.edu

cd {venv_dir}
source ./bin/activate
cd {nnunet_dir}

export nnUNet_raw="{data_dir}/raw"
export nnUNet_preprocessed="{data_dir}/preprocessed"
export nnUNet_results="{data_dir}/results"

{cmd_lines} 