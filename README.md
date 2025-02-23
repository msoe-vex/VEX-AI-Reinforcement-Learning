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

### Arguments

- `--model_path`: Path to an existing model to load and run.
- `--timesteps`: Total timesteps for training the model.
- `--train`: Flag to indicate whether to train a new model.
- `--randomize`: Randomize positions in the environment.
- `--no-randomize`: Do not randomize positions in the environment.

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

## Training with SLURM Job Script

To submit a job to a SLURM cluster, use the `train_models.sh` script. You can specify the number of agents and total timesteps for training.

### Command

```bash
sbatch train_models.sh --agents <NUM_AGENTS> --timesteps <TOTAL_TIMESTEPS>
```

### Example

```bash
sbatch train_models.sh --agents 3 --timesteps 100000
```

### Arguments

- `--agents`: Number of agents to train.
- `--timesteps`: Total timesteps for training each agent.
- `--model_path`: Path to a pretrained model.
- `--entropy`: Entropy coefficient for agent exploration.
- `--learning_rate`: Magnitude of updates to make to the model.
- `--discount`: Value to place on potential future rewards.
- `--randomize`: Randomize positions in the environment.
- `--no-randomize`: Do not randomize positions in the environment.
- `--num_layers`: Number of layers in the policy network.
- `--num_nodes`: Number of nodes per layer in the policy network.

### Cancel job

To view your running jobs
```bash
squeue -u $USER
```

To cancel the job
```bash
scancel <JOB_ID>
```

## File Descriptions

- `rl_environment.py`: Defines the custom VEX environment.
- `run_agent.py`: Script to train or run a PPO agent.
- `train_models.py`: Script to train multiple agents concurrently.
- `train_models.sh`: SLURM job script to submit training jobs to a cluster.
- `remove_null_bytes.py`: Script to remove null bytes from `rl_environment.py`.