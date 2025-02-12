#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning       # Job name
#SBATCH --output=job_results/job_%j/output.txt       # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt         # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-6:0                 # Time limit (DD-HH:MM)
#SBATCH --partition=teaching         # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here

# Default values
NUM_AGENTS=5
TOTAL_TIMESTEPS=10000
MODEL_PATH=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents) NUM_AGENTS="$2"; shift ;;
        --timesteps) TOTAL_TIMESTEPS="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run your code here
if [ -z "$MODEL_PATH" ]; then
    python train_models.py --agents $NUM_AGENTS --timesteps $TOTAL_TIMESTEPS --job_id $SLURM_JOB_ID
else
    python train_models.py --agents $NUM_AGENTS --timesteps $TOTAL_TIMESTEPS --job_id $SLURM_JOB_ID --model_path $MODEL_PATH
fi
