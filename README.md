# VEX AI Reinforcement Learning

This project involves training and running reinforcement learning agents in a custom VEX environment using Gymnasium/PettingZoo and RLlib. It simulates the VEX games including variations: VEX U Competition, VEX U Skills, VEX AI Competition, and VEX AI Skills.

For general setup instructions such as VS Code, Git, SSH keys, Python installation, and cloning repositories, follow [CONTROLS_SETUP.md](CONTROLS_SETUP.md) first.

This README covers the setup and workflow that are specific to this repository. For best results, use WSL or run the code on the MSOE computing cluster (ROSIE) with SLURM. The environment is designed to be flexible and can be run locally for testing, but training is optimized for ROSIE's resources.

## Setup

First follow instructions in [CONTROLS_SETUP.md](CONTROLS_SETUP.md) to complete your general development setup. Then continue here for the repository-specific workflow.

### Connect to ROSIE

Follow these instructions to set up your local machine to get access to ROSIE: <https://msoe-maic.com/library/?nav=Articles&article=1-getting-rosie-access>.

Once you've verified you can sign in, you can follow these instructions to set up an SSH key for passwordless authentication from your local machine to ROSIE:

1. **Open the ssh config file:**

```powershell
code C:\Users\%USERNAME%\.ssh\config
```

2. **Add the following configuration:**

```plaintext
Host ROSIE
    HostName dh-mgmt2.hpc.msoe.edu
    User needhama@ad.msoe.edu

Host dh-node*
    HostName %h
    User needhama@ad.msoe.edu
    ProxyJump ROSIE
```

> Note: Replace `username` with your actual MSOE username.

3. **Create the SSH key:**

First, check if you already have an SSH key:

```powershell
dir C:\Users\%USERNAME%\.ssh
```

If you see files like `id_ed25519` and `id_ed25519.pub`, you already have an SSH key. If not, create one:

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

> Note: Replace `your_email@example.com` with your actual email address.

4. **Add the SSH key to authorized keys**

```powershell
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh ROSIE "cat >> ~/.ssh/authorized_keys"
```

5. **Test the connection**

```powershell
ssh ROSIE
```

If you see a welcome message and a command prompt, you're successfully connected to ROSIE!

### Clone the repository

Once in WSL or on ROSIE, clone the repository in your desired directory:

```bash
git clone git@github.com:msoe-vex/VAIC_25_26.git
cd VAIC_25_26
```

### Create a virtual environment

Once the repository is cloned and you are in the project directory, create and activate the virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you are using a local Windows shell instead of WSL or ROSIE, use the activation command that matches your shell.

## Features

- **Multi-Game Support**: Integrated support for all 4 game variants:
  - `vexu_comp`: VEX U Competition (Team-based, 24" & 15" robots)
  - `vexu_skills`: VEX U Skills (Cooperative, 2 Red robots)
  - `vexai_comp`: VEX AI Competition (Fully autonomous, special scoring rules)
  - `vexai_skills`: VEX AI Skills (Cooperative, Red & Blue robots working for Red score)
- **Explicit Field Setup**: Accurate block coordinates, loader sequences, and robot starting positions derived from official setup notes.
- **Advanced Randomization**: Full-field block scatter (-70" to +70") for robust training.
- **Standardized Scoring**: Unified `Dict[str, int]` scoring interface for all modes.

## General Workflow (ROSIE)

All commands use SLURM to run on the MSOE computing cluster (ROSIE).

1. **Test the Environment**

   Run the environment to check generation and physics. You can select the game mode via CLI:

   ```bash
   # Default: vexai_skills
   srun python vex_env_test.py --game vexai_skills --steps 100
   
   # Other game modes: vexu_comp, vexu_skills, vexai_comp
   srun python vex_env_test.py --game vexu_skills --steps 100 --no-render
   ```
   - Runs a random policy simulation.
   - Saves GIFs to `vex_env_test/` directory by default.

2. **Train the Model**

   Submit a training job to SLURM. Results will be saved to `job_results/job_#####/` where `#####` is the SLURM job ID.

   ```bash
   # Basic training with default parameters
   sbatch scripts/submitTrainingJob.sh --num-iters 100 --learning-rate 0.0003
   
   # Custom game variant and settings
   sbatch scripts/submitTrainingJob.sh --game vexu_skills --num-iters 50 --entropy 0.02
   
   # Training with experiment resumption (auto-loads metadata + latest checkpoint)
   sbatch scripts/submitTrainingJob.sh --experiment-path job_results/job_220065/PPO_2026-01-29_00-36-09
   ```
   - Check job status: `squeue -u $USER`
   - View output: `cat job_results/job_#####/output.txt`
   - View errors: `cat job_results/job_#####/error.txt`

3. **Compile an Experiment (Manual)**

   If you need to recompile the latest checkpoint in an experiment to TorchScript:

   ```bash
   srun python vex_model_compile.py --experiment-path job_results/job_220065/PPO_2026-01-29_00-36-09
   ```

4. **Run a Trained Model**

   Visualize a trained policy from a compiled checkpoint:

   ```bash
   srun python vex_model_test.py --experiment-path job_results/job_220065/PPO_2026-01-29_00-36-09/ --output-dir vex_model_test
   
   # Specify game variant if metadata not found
   srun python vex_model_test.py --experiment-path /path/to/experiment --game vexai_skills
   ```
   - Saves rendered GIFs to `vex_model_test/` directory by default.

---

## Detailed Argument Descriptions

### vex_env_test.py

Environment definition and random simulation runner.

| Argument         | Type    | Default         | Description                                      |
|-------------------|---------|-----------------|--------------------------------------------------|
| `--game`         | str     | `vexai_skills`  | Game variant to test (vexai_skills, vexu_skills, vexai_comp, vexu_comp) |
| `--steps`        | int     | 100             | Number of simulation steps.                      |
| `--no-render`    | flag    | False           | Disable rendering and GIF creation.              |
| `--output-dir`   | str     | `vex_env_test`  | Output directory for renders and GIFs.           |

### vex_model_training.py

RLlib training script (submit via SLURM using `sbatch scripts/submitTrainingJob.sh`).

| Argument           | Type    | Default      | Description                                                                 |
|---------------------|---------|--------------|-----------------------------------------------------------------------------|
| `--num-iters`       | int     | 1            | Number of training iterations                                               |
| `--learning-rate`   | float   | 0.0005       | Learning rate for optimizer                                                 |
| `--discount-factor` | float   | 0.99         | Discount factor (gamma)                                                     |
| `--entropy`         | float   | 0.05         | Entropy coefficient for exploration                                         |
| `--cpus-per-task`   | int     | 1            | Number of CPU cores to use per task                                         |
| `--game`            | str     | `vexai_skills` | Game variant to train (vexai_skills, vexu_skills, vexai_comp, vexu_comp)   |
| `--randomize`       | bool    | True         | Enable full-field block randomization                                       |
| `--num-gpus`        | int     | 0            | Number of GPUs to use (auto-set by SLURM script)                            |
| `--experiment-path` | str     | ""           | Path to experiment directory for resuming (loads metadata + latest checkpoint) |
| `--verbose`         | int     | 1            | Verbosity level (0=silent, 1=default, 2=verbose)                            |
| `--job-id`          | str     | auto         | SLURM Job ID (auto-set by SLURM script)                                     |

Training results are saved to `job_results/job_#####/` where `#####` is the SLURM job ID.

### vex_model_compile.py

Compile RLlib checkpoint to TorchScript (run via `srun`).

| Argument           | Type    | Default         | Description                                                                 |
|---------------------|---------|-----------------|-----------------------------------------------------------------------------|
| `--experiment-path` | str     | Required        | Path to experiment directory (loads metadata + latest checkpoint)            |

### vex_model_test.py

Run simulation using trained TorchScript models (run via `srun`).

| Argument         | Type    | Default         | Description                                               |
|-------------------|---------|-----------------|-----------------------------------------------------------|
| `--experiment-path` | str | Required      | Path to experiment directory containing `.pt` model files |
| `--game`         | str     | auto-detect     | Game variant (auto-detected from training_metadata.json if available) |
| `--output-dir`   | str     | `vex_model_test` | Output directory for rendered GIFs                        |

---

## SLURM Job Management (ROSIE)

### Submitting Training Jobs

Training jobs must be submitted through SLURM using the provided wrapper script:

```bash
sbatch scripts/submitTrainingJob.sh [OPTIONS]
```

The script will:
- Submit the training job to the `teaching` partition
- Save results to `job_results/job_#####/` (where `#####` is the job ID)
- Create separate `output.txt` and `error.txt` files for logging
- Allocate 8 CPU cores by default

Example submissions:

```bash
# Basic training
sbatch scripts/submitTrainingJob.sh --num-iters 100

# Advanced settings
sbatch scripts/submitTrainingJob.sh \
  --game vexu_skills \
  --num-iters 200 \
  --learning-rate 0.0001 \
  --entropy 0.02

# Resume from experiment (latest checkpoint auto-selected)
sbatch scripts/submitTrainingJob.sh \
   --experiment-path job_results/job_220065/PPO_2026-01-29_00-36-09
```

### Monitoring Jobs

```bash
# View all your jobs
squeue -u $USER

# View specific job details
scontrol show job <JOB_ID>

# View job output (while running or after completion)
cat job_results/job_<JOB_ID>/output.txt

# View job errors
cat job_results/job_<JOB_ID>/error.txt

# Cancel a job
scancel <JOB_ID>
```

### Available Game Variants

All scripts support the following game modes via the `--game` argument:
- `vexai_skills` (default): VEX AI Skills - Cooperative red and blue robots working for red score
- `vexu_skills`: VEX U Skills - Two cooperative red robots
- `vexai_comp`: VEX AI Competition - Fully autonomous with special scoring rules
- `vexu_comp`: VEX U Competition - Team-based with 24" and 15" robots
