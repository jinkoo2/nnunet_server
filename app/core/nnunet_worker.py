"""
RQ Worker task for nnU-Net inference.

This worker is executed automatically by RQ when a job is enqueued
from FastAPI (/submit).  It performs inference using nnUNetv2_predict
and stores the segmentation result in the configured results directory.
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Define the absolute path to the script
NNUNET_SCRIPT_PATH = "/home/jk/projects/nnunet_server/scripts/nnunet_predict.sh"

def run_nnunet_predict(job_metadata: dict) -> dict:
    """
    Run nnU-Net inference on the provided input file.

    Parameters
    ----------
    job_metadata : dict
        Information about the job, including:
        {
            "job_id": str,
            "dataset_id": str,
            "input_dir": str,
            "configuration": str,
            "trainer": str,
            "plans": str,
            ...
        }

    Returns
    -------
    dict
        Job result information including output directory and status.
    """

    job_id = job_metadata.get("job_id")
    dataset_id = job_metadata.get("dataset_id")
    input_dir = job_metadata.get("input_dir")
    configuration = job_metadata.get("configuration", "3d_lowres")
    trainer = job_metadata.get("trainer", "nnUNetTrainer")
    plans = job_metadata.get("plans", "nnUNetPlans")


    if not job_id or not dataset_id or not input_dir:
        logger.error("Job metadata missing required fields: %s", job_metadata)
        return {"status": "failed", "reason": "missing metadata"}

    # --- Prepare output directory ---
    output_dir = Path(os.path.join(input_dir, "outputs")) 
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{job_id}] Starting nnU-Net inference using model '{dataset_id}-{configuration}'")
    logger.info(f"[{job_id}] Input file: {input_dir}")
    logger.info(f"[{job_id}] Output directory: {output_dir}")

    # example command:
    #nnUNetv2_predict -d 015 -i /home/jk/data/nnunet_data/predictions/Dataset015_CBCTBladderRectumBowel2/req_000 -o /home/jk/data/nnunet_data/predictions/Dataset015_CBCTBladderRectumBowel2/req_000/outputs -f  0 1 2 3 4 -c 3d_lowres -device cuda

    try:
        # --- Construct the command to run the script ---
        cmd = [
            NNUNET_SCRIPT_PATH,
            input_dir, 
            str(output_dir), 
            dataset_id,
            configuration,
            trainer,
            plans,
        ]
        
        logger.info(f"[{job_id}] Executing command: {' '.join(cmd)}")

        # --- Run inference ---
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            # IMPORTANT: Do NOT use shell=True unless necessary for security reasons.
            # Using the list format is safer and preferred.
        )

        if result.returncode != 0:
            logger.error(f"[{job_id}] Script failed (Exit Code {result.returncode}): {result.stderr}")
            return {
                "status": "failed",
                "job_id": job_id,
                "stderr": result.stderr,
                "stdout": result.stdout, # Log stdout too for debugging
            }

        logger.info(f"[{job_id}] nnU-Net inference completed successfully")

        # --- Save a small JSON summary for downstream usage ---
        summary_path = output_dir / "summary.json"
        summary = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "input_dir": input_dir,
            "output_dir": str(output_dir),
            "completed_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    except Exception as e:
        logger.exception(f"[{job_id}] Exception during nnunet_predict shell execution: {e}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }

