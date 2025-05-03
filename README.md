# VEX AI Reinforcement Learning

This project involves training and running reinforcement learning agents in a custom VEX environment using Gymnasium/PettingZoo and RlLib.

## Setup

Use Python 3.8

```bash
pip install ray[tune] gymnasium pettingzoo dm-tree matplotlib casadi typer opencv-python scipy lz4 torch gputil
```

<!-- 

Old Readme

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
python run_agent.py --model-path <MODEL_PATH>
```

### Example

```bash
python run_agent.py --model-path vex_high_stakes_ppo
```

### Arguments

- `--model-path`: Path to an existing model to load and run.
- `--timesteps`: Total timesteps for training the model.
- `--train`: Flag to indicate whether to train a new model.
- `--randomize`: Randomize positions in the environment.
- `--no-randomize`: Do not randomize positions in the environment.
- `--realistic-pathing`: Use realistic pathing.
- `--no-realistic-pathing`: Do not use realistic pathing.
- `--realistic-vision`: Use realistic vision.
- `--no-realistic-vision`: Do not use realistic vision.
- `--robot-num`: Specify robot num to limit available elements (0-2). `0` to use all objects on the field, `1` to for the top half (inclusive), and `2` for the bottom half (exclusive).

## Running Custom Instructions

To run the environment based on a list of custom actions, use the `run_instructions.py` script. You can specify the path to an instruction file containing the custom actions.

### Command

```bash
python run_instructions.py --instructions-path <INSTRUCTIONS_PATH>
```

### Example

```bash
python run_instructions.py --instructions-path instructions.txt
```

### Arguments

- `--instructions-path`: Path to the instruction file containing custom actions.
- `--randomize`: Randomize positions in the environment.
- `--no-randomize`: Do not randomize positions in the environment.
- `--realistic-pathing`: Use realistic pathing.
- `--no-realistic-pathing`: Do not use realistic pathing.
- `--realistic-vision`: Use realistic vision.
- `--no-realistic-vision`: Do not use realistic vision.
- `--robot-num`: Specify robot num to limit available elements (0-2). `0` to use all objects on the field, `1` to for the top half (inclusive), and `2` for the bottom half (exclusive).

## Training or Continuing Training an Agent

To train a new model or continue training an existing model, use the `run_agent.py` script with the `--train` flag. You can also specify the path to an existing model to continue training.

### Command

```bash
python run_agent.py --train --timesteps <TOTAL_TIMESTEPS> [--model-path <MODEL_PATH>]
```

### Example

Train a new model:
```bash
python run_agent.py --train --timesteps 500
```

Continue training an existing model:
```bash
python run_agent.py --train --timesteps 500 --model-path vex_high_stakes_ppo
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
- `--model-path`: Path to a pretrained model.
- `--entropy`: Entropy coefficient for agent exploration.
- `--learning-rate`: Magnitude of updates to make to the model.
- `--discount-factor`: Value to place on potential future rewards.
- `--randomize`: Randomize positions in the environment.
- `--no-randomize`: Do not randomize positions in the environment.
- `--num-layers`: Number of layers in the policy network.
- `--num-nodes`: Number of nodes per layer in the policy network.
- `--realistic-pathing`: Use realistic pathing.
- `--no-realistic-pathing`: Do not use realistic pathing.
- `--realistic-vision`: Use realistic vision.
- `--no-realistic-vision`: Do not use realistic vision.
- `--robot-num`: Specify robot num to limit available elements (0-2). `0` to use all objects on the field, `1` to for the top half (inclusive), and `2` for the bottom half (exclusive).

### Cancel job

To view your running jobs
```bash
squeue -u $USER
```

To cancel the job
```bash
scancel <JOB_ID>
```

## Compiling Output

To compile the output of the RL model's action sequence into C++ autonomous code, use the `compile_output.py` script.

### Command

```bash
python compile_output.py --sequence <SEQUENCE_FILE> --output <OUTPUT_FILE>
```

### Example

```bash
python compile_output.py --sequence auton_sequence.txt --output auton1.h
```

### Arguments

- `--sequence`: The action sequence to translate.
- `--output`: The file path for the translated code.

## File Descriptions

- `rl_environment.py`: Defines the custom VEX environment.
- `run_agent.py`: Script to train or run a PPO agent.
- `run_instructions.py`: Script to run the environment based on a list of custom actions.
- `train_models.py`: Script to train multiple agents concurrently.
- `train_models.sh`: SLURM job script to submit training jobs to a cluster.
- `compile_output.py`: Script to translate an RL model's action sequence into C++ autonomous code.
- `remove_null_bytes.py`: Script to remove null bytes from `rl_environment.py`. 
-->