# VEX AI Reinforcement Learning

This project involves training and running reinforcement learning agents in a custom VEX environment using Gymnasium/PettingZoo and RLlib. It simulates the VEX High Stakes game (2024-2025) variations: VEX U Competition, VEX U Skills, VEX AI Competition, and VEX AI Skills.

## Setup

Use Python 3.12

```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

## Features

- **Multi-Game Support**: Integrated support for all 4 game variants:
  - `vexu_comp`: VEX U Competition (Team-based, 24" & 15" robots)
  - `vexu_skills`: VEX U Skills (Cooperative, 2 Red robots)
  - `vexai_comp`: VEX AI Competition (Fully autonomous, special scoring rules)
  - `vexai_skills`: VEX AI Skills (Cooperative, Red & Blue robots working for Red score)
- **Explicit Field Setup**: Accurate block coordinates, loader sequences, and robot starting positions derived from official setup notes.
- **Advanced Randomization**: Full-field block scatter (-70" to +70") for robust training.
- **Standardized Scoring**: Unified `Dict[str, int]` scoring interface for all modes.

## General Workflow

1. **Test the Environment**

   Run the environment to check generation and physics. You can select the game mode via CLI:

   ```bash
   # Default: vex_ai_competition
   python vexEnv.py --mode vex_ai_competition --steps 100
   
   # Other modes: vex_u_competition, vex_u_skills, vex_ai_skills
   python vexEnv.py --mode vexu_skills
   ```
   - Runs a random policy simulation.
   - Saves GIFs to `vexEnv/steps`.

2. **Train the Model**

   Train a PPO agent using RLlib. The script automatically compiles the best checkpoint to TorchScript after training.

   ```bash
   python vexEnvTraining.py --num-iters 100 --learning-rate 0.0003 --algorithm PPO
   ```

3. **Compile a Checkpoint (Manual)**

   If you need to recompile a specific checkpoint:

   ```bash
   python vexModelCompile.py --checkpoint-path /path/to/checkpoint_000005
   ```

4. **Run a Trained Model**

   Visualize a trained policy:

   ```bash
   python vexEnvRun.py --model-path /path/to/policy.pt
   ```

---

## Detailed Argument Descriptions

### vexEnv.py

Environment definition and random simulation runner.

| Argument         | Type    | Default              | Description                                      |
|-------------------|---------|----------------------|--------------------------------------------------|
| `--mode`         | str     | `vex_ai_competition` | Game variant to run.                             |
| `--steps`        | int     | 100                  | Number of simulation steps.                      |
| `--render_mode`  | str     | `rgb_array`          | Rendering mode.                                  |

### vexEnvTraining.py

RLlib training script.

| Argument           | Type    | Default   | Description                                                                 |
|---------------------|---------|-----------|-----------------------------------------------------------------------------|
| `--learning-rate`   | float   | 0.0005    | Learning rate for optimizer                                                 |
| `--discount-factor` | float   | 0.99      | Discount factor gamma                                                       |
| `--entropy`         | float   | 0.01      | Entropy coefficient                                                         |
| `--num-iters`       | int     | 1         | Number of training iterations                                               |
| `--randomize`       | bool    | True      | Enable full-field block randomization                                       |
| `--model-path`      | str     | ""        | Path to load a pre-trained model (for transfer learning)                    |

### vexModelCompile.py

Compile RLlib checkpont to TorchScript.

| Argument           | Type    | Required | Description                                                                 |
|---------------------|---------|----------|-----------------------------------------------------------------------------|
| `--checkpoint-path` | str     | Yes      | Path to the RLlib checkpoint directory                                      |
| `--output-path`     | str     | No       | Output directory for `.pt` file                                             |

### vexEnvRun.py

Run simulation using a TorchScript model.

| Argument         | Type    | Required | Description                                               |
|-------------------|---------|----------|-----------------------------------------------------------|
| `--model-path`    | str     | Yes      | Path to the trained TorchScript model (`.pt` file)        |

---

### SLURM Submission (submitTrainingJob.sh)

Wrapper script for submitting `vexEnvTraining.py` jobs to SLURM (e.g., on ROSIE). Arguments passed to this script are forwarded to the python script.

```bash
sbatch submitTrainingJob.sh --num-iters 1000 --learning-rate 0.0001
```

### SLURM Job Management

View jobs: `squeue -u $USER`
Cancel job: `scancel <JOB_ID>`