# VEX AI Reinforcement Learning

This project involves training and running reinforcement learning agents in a custom VEX environment using Gymnasium/PettingZoo and RLlib.

## Setup

Use Python 3.8

```bash
pip install ray[default] gymnasium pettingzoo dm-tree matplotlib casadi typer opencv-python scipy lz4 torch gputil imageio
```

## General Workflow

1. **Test the Environment**

   Run the environment to ensure everything is working and to generate a random simulation GIF:

   ```bash
   python pettingZooEnv.py
   ```
   - Runs the environment with random actions, renders each step, and creates a GIF in the `pettingZooEnv/steps` directory.

2. **Train the Model**

   Train a new agent or continue training from a checkpoint:

   ```bash
   python pettingZooTraining.py [OPTIONS]
   ```
   - Trains a PPO agent using RLlib and, after training, automatically compiles the best checkpoint to a TorchScript model.

   **Example:**
   ```bash
   python pettingZooTraining.py --num-iters 100 --learning-rate 0.0003 --entropy 0.02
   ```

   - **Note:** Due to a bug with RLlib's `local_dir` parameter, restoring from a checkpoint may not be functional.

3. **Compile a Checkpoint to TorchScript (if needed)**

   The training script will automatically compile the best checkpoint. If you need to recompile (e.g., if training ends early or you want to compile a different checkpoint), run:

   ```bash
   python pettingZooCompile.py --checkpoint-path /path/to/checkpoint_000005 --output-path /path/to/output
   ```
   - Loads a PPO checkpoint and saves the policy as a TorchScript `.pt` file.

4. **Run a Simulation with a Trained Model**

   To run a simulation using a compiled TorchScript model and generate a GIF:

   ```bash
   python pettingZooRun.py --model-path /path/to/shared_policy.pt
   ```
   - Loads a TorchScript model and runs a simulation in the environment, generating a GIF in the `pettingZooRun/steps` directory.

---

### (Optional) Submit a Training Job with SLURM

   To submit a training job to SLURM:

   ```bash
   sbatch submitTrainingJob.sh [OPTIONS]
   ```
   - Submits a job to SLURM to run `pettingZooTraining.py` with the specified options.

   **Example:**
   ```bash
   sbatch submitTrainingJob.sh --num-iters 1000 --learning-rate 0.0001 --entropy 0.005
   ```

   To continue training from a checkpoint:
   ```bash
   sbatch submitTrainingJob.sh --num-iters 500 --checkpoint-path /path/to/job_results/job_XXXX/checkpoint_YYYY
   ```

### SLURM Job Management

View running jobs:
```bash
squeue -u $USER
```

Cancel a job:
```bash
scancel <JOB_ID>
```

---

## Detailed Argument Descriptions

### pettingZooEnv.py

Environment definition and random simulation runner.

- **No command-line arguments.**  
  Edit the file directly to change:
  - `render_mode`: Rendering mode (`"all"`, `"human"`, or `"image"`)
  - `output_directory`: Where to save images/GIFs
  - `randomize`: Whether to randomize environment on reset

---

### pettingZooTraining.py

RLlib training script.

| Argument           | Type    | Default   | Description                                                                 |
|---------------------|---------|-----------|-----------------------------------------------------------------------------|
| `--learning-rate`   | float   | 0.0005    | Learning rate for optimizer                                                 |
| `--discount-factor` | float   | 0.99      | Discount factor gamma for future rewards                                    |
| `--entropy`         | float   | 0.01      | Entropy coefficient for exploration                                         |
| `--num-iters`       | int     | 1         | Number of training iterations                                               |
| `--cpus-per-task`   | int     | 1         | Number of CPUs to use                                                       |
| `--job-id`          | str     | ""        | SLURM job ID for logging/output directory                                   |
| `--model-path`      | str     | ""        | Path to save/load the model (not used for RLlib, use checkpoint-path)       |
| `--randomize`       | bool    | True      | Enable or disable randomization of environment                              |
| `--num-layers`      | int     | 2         | Number of hidden layers in the policy network                               |
| `--num-nodes`       | int     | 64        | Number of nodes per hidden layer                                            |
| `--num-gpus`        | int     | 0         | Number of GPUs to use                                                       |
| `--partition`       | str     | "teaching"| SLURM partition to use                                                      |
| `--algorithm`       | str     | "PPO"     | RLlib algorithm to use                                                      |
| `--checkpoint-path` | str     | ""        | Path to checkpoint directory to continue training                           |
| `--verbose`         | int     | 0         | Verbosity level (0=silent, 1=default, 2=verbose)                            |

---

### pettingZooCompile.py

Compile RLlib checkpoint to TorchScript.

| Argument           | Type    | Required | Description                                                                 |
|---------------------|---------|----------|-----------------------------------------------------------------------------|
| `--checkpoint-path` | str     | Yes      | Path to the RLlib checkpoint directory (e.g., `/path/to/checkpoint_000005`) |
| `--output-path`     | str     | No       | Directory to save the TorchScript model(s). If not specified, saves to checkpoint directory |

---

### pettingZooRun.py

Run simulation using a TorchScript model.

| Argument         | Type    | Required | Description                                               |
|-------------------|---------|----------|-----------------------------------------------------------|
| `--model-path`    | str     | Yes      | Path to the trained TorchScript model (`.pt` file)        |

---

### submitTrainingJob.sh

SLURM job script to run on ROSIE.

All arguments are passed as command-line arguments and forwarded to `pettingZooTraining.py`:

| Argument           | Type    | Default   | Description                                                                 |
|---------------------|---------|-----------|-----------------------------------------------------------------------------|
| `--checkpoint-path` | str     | ""        | Path to checkpoint directory to continue training                           |
| `--entropy`         | float   | 0.01      | Entropy coefficient for exploration                                         |
| `--learning-rate`   | float   | 0.0005    | Learning rate                                                               |
| `--discount-factor` | float   | 0.99      | Discount factor gamma                                                       |
| `--randomize`       | bool    | True      | Enable/disable randomization                                                |
| `--num-layers`      | int     | 2         | Number of hidden layers                                                     |
| `--num-nodes`       | int     | 64        | Number of nodes per layer                                                   |
| `--num-iters`       | int     | 10        | Number of training iterations                                               |
| `--algorithm`       | str     | "PPO"     | RLlib algorithm                                                             |
| `--verbose`         | int     | 0         | Verbosity level                                                             |