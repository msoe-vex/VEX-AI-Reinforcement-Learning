from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import Process
import argparse
import numpy as np
import torch as th
import csv
import os
import time
import multiprocessing

# Check for null bytes in the rl_environment.py file
with open('rl_environment.py', 'rb') as f:
    content = f.read()
    if b'\x00' in content:
        raise ValueError("rl_environment.py contains null bytes.")

from rl_environment import VEXHighStakesEnv

# ---------------------------------------------------------------------------
# Evaluate Agent
# Runs one episode with the trained model and returns its total score.
# ---------------------------------------------------------------------------
def evaluate_agent(model, env_class, save_path, randomize_positions, realistic_pathing, realistic_vision, robot_num):
    env = env_class(save_path, randomize_positions=randomize_positions, realistic_pathing=realistic_pathing, realistic_vision=realistic_vision, robot_num=robot_num)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return env.total_score

# ---------------------------------------------------------------------------
# Train Agent
# Trains a single agent and saves the best model.
# ---------------------------------------------------------------------------
def train_agent(env_class, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path, randomize_positions, num_layers, num_nodes, realistic_pathing, realistic_vision, robot_num, algorithm):
    def make_env():
        return env_class(save_path, randomize_positions=randomize_positions, realistic_pathing=realistic_pathing, realistic_vision=realistic_vision, robot_num=robot_num)

    # Validate the environment with a single instance
    single_env = make_env()
    check_env(single_env, warn=True)
    single_env.close()  # Close the single environment instance

    # Create SubprocVecEnv for parallel environments
    num_cpus = min(max(1, multiprocessing.cpu_count()), 32)
    print(f"Creating {num_cpus} parallel environments")
    env = SubprocVecEnv([make_env for _ in range(num_cpus)])
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Training agent with device {device}")
    
    # Define network architecture and policy parameters.
    net_arch = [num_nodes] * num_layers
    policy_kwargs = dict(net_arch=net_arch, activation_fn=th.nn.ReLU)
    
    if model_path:
        print(f"Loading model from {model_path}")
        if algorithm == "PPO":
            model = PPO.load(model_path, env=env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device)
        elif algorithm == "DQN":
            model = DQN.load(model_path, env=env, verbose=1, learning_rate=learning_rate, gamma=discount_factor, device=device)
    else:
        print("Creating new model")
        if algorithm == "PPO":
            model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device, policy_kwargs=policy_kwargs)
        elif algorithm == "DQN":
            model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=learning_rate, gamma=discount_factor, device=device)

    check_steps = int(16384/num_cpus)  # Number of steps to check the model
    print(f"Evaluating every {check_steps} timesteps; saving to {save_path}")
    best_score = -np.inf
    initial_timesteps = model.num_timesteps
    n_steps = 0

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open CSV file for writing
    csv_file = open(f"{save_path}_scores.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time Elapsed', 'Timesteps', 'Score', 'Best Score'])

    start_time = time.time()

    while n_steps < total_timesteps:
        model.learn(total_timesteps=check_steps, reset_num_timesteps=False)
        n_steps = model.num_timesteps - initial_timesteps
        score = evaluate_agent(model, env_class, save_path, randomize_positions, realistic_pathing, realistic_vision, robot_num)
        elapsed_time = time.time() - start_time
        print(f"Score: {score}")
        
        if score > best_score:
            best_score = score
            model.save(save_path)
            print(f"Model saved to {save_path}.")
        
        csv_writer.writerow([elapsed_time, n_steps, score, best_score])
        csv_file.flush()  # Ensure data is written to the file

        print(f"Best Score: {best_score} | {n_steps}/{total_timesteps} timesteps")
    
    # Close CSV file
    csv_file.close()

    score = evaluate_agent(model, env_class, save_path, randomize_positions, realistic_pathing, realistic_vision, robot_num)
    model.save(save_path+"_final")
    print("Training complete.")
    print(f"Final model saved to {save_path}_final with score {score}")

# ---------------------------------------------------------------------------
# Main: Parse arguments and train agents concurrently.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to train')  # Default: 1
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total timesteps for training each agent')  # Default: 1000000
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')  # Default: 0.01
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')  # Default: 0.0005
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor')  # Default: 0.99
    parser.add_argument('--job-id', type=str, required=True, help='Job ID for saving models')
    parser.add_argument('--model-path', type=str, default="", help='Path to a pretrained model')  # Default: ""
    parser.add_argument('--randomize', action='store_true', help='Randomize positions')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of network layers')  # Default: 2
    parser.add_argument('--num-nodes', type=int, default=64, help='Nodes per layer')  # Default: 64
    parser.add_argument('--realistic-pathing', action='store_true', help='Use realistic pathing')
    parser.add_argument('--no-realistic-pathing', action='store_false', dest='realistic_pathing', help='Do not use realistic pathing')
    parser.add_argument('--realistic-vision', action='store_true', help='Use realistic vision')
    parser.add_argument('--no-realistic-vision', action='store_false', dest='realistic_vision', help='Do not use realistic vision')
    parser.add_argument('--robot-num', type=int, choices=[0, 1, 2], default=0, help='Specify which robot to use (0-2)')  # Default: 0
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use')  # Default: 0
    parser.add_argument('--cpus-per-task', type=int, default=32, help='Number of CPUs per task')  # Default: 32
    parser.add_argument('--partition', type=str, default='teaching', help='Partition to use for training')  # Default: teaching
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'DQN'], default='PPO', help='Algorithm to use for training (PPO or DQN)')  # Default: PPO
    parser.set_defaults(realistic_pathing=False, realistic_vision=True, randomize=True)
    args = parser.parse_args()

    print("Training with arguments:")
    print(f"Agents: {args.agents}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Entropy: {args.entropy}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Discount factor: {args.discount_factor}")
    print(f"Job ID: {args.job_id}")
    print(f"Model path: {args.model_path}")
    print(f"Randomize: {args.randomize}")
    print(f"Layers: {args.num_layers}, Nodes: {args.num_nodes}")
    print(f"Use realistic pathing: {args.realistic_pathing}")
    print(f"Use realistic vision: {args.realistic_vision}")
    print(f"Robot number: {args.robot_num}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"CPUs per task: {args.cpus_per_task}")
    print(f"Partition: {args.partition}")
    print(f"Algorithm: {args.algorithm}")

    processes = []
    for i in range(args.agents):
        save_path = f"job_results/job_{args.job_id}/models/model_{i}"
        p = Process(target=train_agent, args=(VEXHighStakesEnv, args.timesteps, save_path, args.entropy, args.learning_rate, args.discount_factor, args.model_path, args.randomize, args.num_layers, args.num_nodes, args.realistic_pathing, args.realistic_vision, args.robot_num, args.algorithm))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All agents trained.")
