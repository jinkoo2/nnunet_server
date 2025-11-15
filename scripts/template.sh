cd {project_root}
conda activate nnunet_server

export nnUNet_raw="{data_dir}/raw"
export nnUNet_preprocessed="{data_dir}/preprocessed"
export nnUNet_results="{data_dir}/results"

{cmd_lines} 

