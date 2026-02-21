#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning    # Job name
#SBATCH --output=job_results/job_%j/output.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-24:0                                # Time limit (DD-HH:MM)
#SBATCH --partition=teaching --gpus=0               # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --cpus-per-task=8 --tasks=1                # Number of CPU cores to use

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment-path) EXPERIMENT_PATH="$2"; shift 2 ;;
        --entropy) ENTROPY_I="$2"; shift 2 ;;
        --entropy-final) ENTROPY_F="$2"; shift 2 ;;
        --learning-rate) LR_I="$2"; shift 2 ;;
        --learning-rate-final) LR_F="$2"; shift 2 ;;
        --discount-factor) DISCOUNT_I="$2"; shift 2 ;;
        --randomize) RANDOMIZE="$2"; shift 2 ;;
        --no-randomize) RANDOMIZE="False"; shift ;;
        --num-iters) NUM_ITERS="$2"; shift 2 ;;
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --verbose) VERBOSE="$2"; shift 2 ;;
        --game) GAME="$2"; shift 2 ;;
        --enable-communication) ENABLE_COMMUNICATION="$2"; shift 2 ;;
        --no-enable-communication) ENABLE_COMMUNICATION="False"; shift ;;
        --deterministic) DETERMINISTIC="$2"; shift 2 ;;
        --no-deterministic) DETERMINISTIC="False"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Run your code here
# Normalize ENABLE_COMMUNICATION (accept True/False/1/0/etc.) and convert to a flag
if [[ "${ENABLE_COMMUNICATION:-True}" =~ ^([Tt][Rr][Uu][Ee]|1)$ ]]; then
    COMM_FLAG="--enable-communication"
else
    COMM_FLAG="--no-enable-communication"
fi

# Normalize RANDOMIZE (accept True/False/1/0/etc.) and convert to a flag
if [[ "${RANDOMIZE:-True}" =~ ^([Tt][Rr][Uu][Ee]|1)$ ]]; then
    RAND_FLAG="--randomize"
else
    RAND_FLAG="--no-randomize"
fi

# Normalize DETERMINISTIC (accept True/False/1/0/etc.) and convert to a flag
if [[ "${DETERMINISTIC:-True}" =~ ^([Tt][Rr][Uu][Ee]|1)$ ]]; then
    DETERMINISTIC_FLAG="--deterministic"
else
    DETERMINISTIC_FLAG="--no-deterministic"
fi

python vex_model_training.py \
    --num-iters ${NUM_ITERS:-10} \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --entropy ${ENTROPY_I:-0.05} \
    --entropy-final ${ENTROPY_F:-0.005} \
    --learning-rate ${LR_I:-0.0005} \
    --learning-rate-final ${LR_F:-0.00005} \
    --discount-factor ${DISCOUNT_I:-0.98} \
    --job-id $SLURM_JOB_ID \
    ${RAND_FLAG} \
    --num-gpus $SLURM_GPUS \
    --partition $SLURM_JOB_PARTITION \
    --algorithm ${ALGORITHM:-PPO} \
    --experiment-path ${EXPERIMENT_PATH:-""} \
    --verbose ${VERBOSE:-1} \
    --game ${GAME:-vexai_skills} \
    ${COMM_FLAG} \
    ${DETERMINISTIC_FLAG}