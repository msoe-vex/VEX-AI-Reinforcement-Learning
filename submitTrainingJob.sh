#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning    # Job name
#SBATCH --output=job_results/job_%j/output.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-12:0                                # Time limit (DD-HH:MM)
#SBATCH --partition=teaching --gpus=1               # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --cpus-per-task=8 --tasks=1                # Number of CPU cores to use

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --entropy) ENTROPY_I="$2"; shift 2 ;;
        --learning-rate) LR_I="$2"; shift 2 ;;
        --discount-factor) DISCOUNT_I="$2"; shift 2 ;;
        --randomize) RANDOMIZE="$2"; shift 2 ;;
        --num-layers) NUM_LAYERS="$2"; shift 2 ;;
        --num-nodes) NUM_NODES="$2"; shift 2 ;;
        --num-iters) NUM_ITERS="$2"; shift 2 ;;
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --verbose) VERBOSE="$2"; shift 2 ;;
        --game) GAME="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Run your code here
python vexEnvTraining.py \
    --num-iters ${NUM_ITERS:-10} \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --entropy ${ENTROPY_I:-0.01} \
    --learning-rate ${LR_I:-0.0005} \
    --discount-factor ${DISCOUNT_I:-0.99} \
    --job-id $SLURM_JOB_ID \
    --randomize ${RANDOMIZE:-True} \
    --num-layers ${NUM_LAYERS:-2} \
    --num-nodes ${NUM_NODES:-64} \
    --num-gpus $SLURM_GPUS \
    --partition $SLURM_JOB_PARTITION \
    --algorithm ${ALGORITHM:-PPO} \
    --checkpoint-path ${CHECKPOINT_PATH:-""} \
    --verbose ${VERBOSE:-0} \
    --game ${GAME:-vexai_skills}