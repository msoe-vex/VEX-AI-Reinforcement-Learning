#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning       # Job name
#SBATCH --output=job_results/job_%j/output.txt       # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt         # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-6:0                 # Time limit (DD-HH:MM)
#SBATCH --partition=teaching         # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents) NUM_AGENTS="$2"; shift ;;
        --timesteps) TOTAL_TIMESTEPS="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --entropy) ENTROPY_I="$2"; shift ;;
        --learning_rate) LR_I="$2"; shift ;;
        --discount) DISCOUNT_I="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run your code here
python train_models.py \
    --agents ${NUM_AGENTS:-1} \
    --timesteps ${TOTAL_TIMESTEPS:-1000000} \
    --entropy ${ENTROPY_I:-0.01} \
    --learning_rate ${LR_I:-0.0003} \
    --discount ${DISCOUNT_I:-0.99} \
    --job_id $SLURM_JOB_ID \
    --model_path ${MODEL_PATH:-""}