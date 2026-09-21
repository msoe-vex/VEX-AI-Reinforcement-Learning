#!/bin/bash
#SBATCH --job-name=vex_ai_reinforcement_learning    # Job name
#SBATCH --output=job_results/job_%j/output.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=job_results/job_%j/error.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-24:0                               # Time limit (DD-HH:MM)
#SBATCH --partition=teaching --gpus=1               # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --cpus-per-task=8 --tasks=1                 # Number of CPU cores to use
#SBATCH --mem=16G                                   # Memory per node (allocate at least 16GB for training to prevent getting killed due to out-of-memory errors)

# Use the repository virtual environment when it exists so the batch job uses
# the same interpreter as the documented setup.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${SLURM_SUBMIT_DIR:-$PWD}/venv/bin/python" ]]; then
    PYTHON_BIN="${SLURM_SUBMIT_DIR:-$PWD}/venv/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

if ! "${PYTHON_BIN}" -c "import ray" >/dev/null 2>&1; then
    echo "Ray is not installed for ${PYTHON_BIN}. Run: python -m venv venv && source venv/bin/activate && python -m pip install -r requirements.txt" >&2
    exit 1
fi

# Run the python training script
# Pass through all user arguments using "$@" while capturing SLURM environment variables
srun "${PYTHON_BIN}" vex_model_training.py \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-1}" \
    --job-id "${SLURM_JOB_ID:-local}" \
    --num-gpus "${SLURM_GPUS:-0}" \
    --partition "${SLURM_JOB_PARTITION:-unknown}" \
    "$@"