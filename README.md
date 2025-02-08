# VEX AI Reinforcement Learning

This project involves training and running reinforcement learning agents in a custom VEX environment using the Stable Baselines3 library.

## Setup

1. **Install Dependencies**:
    ```bash
    pip install gymnasium stable-baselines3 matplotlib imageio
    ```

2. **Ensure the `rl_environment.py` file is free of null bytes**:
    ```bash
    python remove_null_bytes.py
    ```

## Training Agents

To train multiple agents concurrently, use the `train_models.py` script. You can specify the number of agents and total timesteps for training.

### Command

```bash
python train_models.py --agents <NUM_AGENTS> --timesteps <TOTAL_TIMESTEPS> --job_id <JOB_ID>
```

### Example

```bash
python train_models.py --agents 3 --timesteps 100000 --job_id 12345
```

## Running an Agent

To run a trained agent, use the `run_agent.py` script. You can specify the path to an existing model to load and run.

### Command

```bash
python run_agent.py --model_path <MODEL_PATH>
```

### Example

```bash
python run_agent.py --model_path vex_high_stakes_ppo
```

## Training or Continuing Training an Agent

To train a new model or continue training an existing model, use the `run_agent.py` script with the `--train` flag. You can also specify the path to an existing model to continue training.

### Command

```bash
python run_agent.py --train --timesteps <TOTAL_TIMESTEPS> [--model_path <MODEL_PATH>]
```

### Example

Train a new model:
```bash
python run_agent.py --train --timesteps 500
```

Continue training an existing model:
```bash
python run_agent.py --train --timesteps 500 --model_path vex_high_stakes_ppo
```

## SLURM Job Script

To submit a job to a SLURM cluster, use the `train_models.sh` script. You can specify the number of agents and total timesteps for training.

### Command

```bash
sbatch train_models.sh --agents <NUM_AGENTS> --timesteps <TOTAL_TIMESTEPS>
```

### Example

```bash
sbatch train_models.sh --agents 3 --timesteps 100000
```

## File Descriptions

- `rl_environment.py`: Defines the custom VEX environment.
- `run_agent.py`: Script to train or run a PPO agent.
- `train_models.py`: Script to train multiple agents concurrently.
- `train_models.sh`: SLURM job script to submit training jobs to a cluster.
- `remove_null_bytes.py`: Script to remove null bytes from `rl_environment.py`.