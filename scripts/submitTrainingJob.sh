#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning    # Job name
#SBATCH --output=job_results/job_%j/output.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-24:0                                # Time limit (DD-HH:MM)
#SBATCH --partition=teaching --gpus=0               # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --cpus-per-task=8 --tasks=1                # Number of CPU cores to use

# Run the python training script
# Pass through all user arguments using "$@" while capturing SLURM environment variables
python vex_model_training.py \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-1}" \
    --job-id "${SLURM_JOB_ID:-local}" \
    --num-gpus "${SLURM_GPUS:-0}" \
    --partition "${SLURM_JOB_PARTITION:-unknown}" \
    "$@"