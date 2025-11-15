#!/bin/bash

# --- 1. Set Environment Variables ---
# NOTE: Replace these paths with your actual data directory paths
export nnUNet_raw="/home/jk/data/nnunet_data/raw"
export nnUNet_preprocessed="/home/jk/data/nnunet_data/preprocessed"
export nnUNet_results="/home/jk/data/nnunet_data/results"

# --- 2. Activate the Conda Environment Robustly ---
VENV_NAME="nnunet_server"
CONDA_PROFILE="/home/jk/miniconda3/etc/profile.d/conda.sh" # <-- Your initialization script path

# Source the main conda functions setup script
if [ -f "$CONDA_PROFILE" ]; then
    # The dot command '.' or 'source' executes the script in the current shell context.
    . "$CONDA_PROFILE" 
    echo "Conda environment functions initialized."
else
    echo "FATAL ERROR: Conda profile script not found at $CONDA_PROFILE"
    exit 1
fi

# --- 2. Activate the Virtual Environment ---
VENV_NAME="nnunet_server"
conda activate $VENV_NAME

# --- 3. Execute nnUNet Command ---
# $1: input_dir, $2: output_dir, $3: dataset_id, $4: configuration
INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATASET_ID="$3"
CONFIGURATION="$4"
TRAINER="${5:-nnUNetTrainer}"
PLANS="${6:-nnUNetPlans}"
# Add other arguments as necessary (trainer, plans, device)

echo "Starting nnUNetv2_predict for Dataset $DATASET_ID, Config $CONFIGURATION"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# Execute the nnUNet command. Since the VENV is active, 'nnUNetv2_predict' is found in PATH.
nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIGURATION" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -f 0 1 2 3 4 \
    -device cuda \
    --disable_progress_bar

# Check the exit status of the nnUNet command
NNUNET_STATUS=$?

# Deactivate the environment (good practice)
deactivate

# Exit with nnU-Net's status code. RQ will mark the job as failed if this is non-zero.
exit $NNUNET_STATUS