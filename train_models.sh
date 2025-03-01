#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning       # Job name
#SBATCH --output=job_results/job_%j/output.txt       # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt         # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-6:0                 # Time limit (DD-HH:MM)
#SBATCH --partition=teaching         # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents) NUM_AGENTS="$2"; shift 2 ;;
        --timesteps) TOTAL_TIMESTEPS="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --entropy) ENTROPY_I="$2"; shift 2 ;;
        --learning-rate) LR_I="$2"; shift 2 ;;
        --discount-factor) DISCOUNT_I="$2"; shift 2 ;;
        --randomize) RANDOMIZE="--randomize"; shift ;;
        --no-randomize) RANDOMIZE="--no-randomize"; shift ;;
        --num-layers) NUM_LAYERS="$2"; shift 2 ;;
        --num-nodes) NUM_NODES="$2"; shift 2 ;;
        --realistic-pathing) REALISTIC_PATHING="--realistic-pathing"; shift ;;
        --no-realistic-pathing) REALISTIC_PATHING="--no-realistic-pathing"; shift ;;
        --realistic-vision) REALISTIC_VISION="--realistic-vision"; shift ;;
        --no-realistic-vision) REALISTIC_VISION="--no-realistic-vision"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Run your code here
python train_models.py \
    --agents ${NUM_AGENTS:-1} \
    --timesteps ${TOTAL_TIMESTEPS:-1000000} \
    --entropy ${ENTROPY_I:-0.01} \
    --learning-rate ${LR_I:-0.0005} \
    --discount-factor ${DISCOUNT_I:-0.99} \
    --job-id $SLURM_JOB_ID \
    --model-path ${MODEL_PATH:-""} \
    ${RANDOMIZE:-"--randomize"} \
    --num-layers ${NUM_LAYERS:-2} \
    --num-nodes ${NUM_NODES:-64} \
    ${REALISTIC_PATHING:-"--no-realistic-pathing"} \
    ${REALISTIC_VISION:-"--realistic-vision"}