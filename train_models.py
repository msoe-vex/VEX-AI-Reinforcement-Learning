import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from multiprocessing import Process
import os
import argparse
import numpy as np

# Ensure the rl_environment.py file is read correctly
with open('rl_environment.py', 'rb') as f:
    content = f.read()
    if b'\x00' in content:
        raise ValueError("The file rl_environment.py contains null bytes.")

from rl_environment import VEXHighStakesEnv

def train_agent(env_class, total_timesteps, save_path, entropy, learning_rate, discount_factor):
    env = env_class()
    check_env(env, warn=True)
    model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")
    print(f"Final score: {env.total_score}\n")  # Print the final score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to train')
    parser.add_argument('--timesteps', type=int, default=1000, help='Total timesteps for training each agent')
    parser.add_argument('--entropy', type=float, default=0.0, help='Entropy coefficient for agent exploration')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Magnitude of updates to make to model')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Value to place on potential future rewards')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID for saving models')
    args = parser.parse_args()

    num_agents = args.agents
    total_timesteps = args.timesteps
    entropy = args.entropy
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    job_id = args.job_id

    processes = []
    for i in range(num_agents):
        save_path = f"job_results/job_{job_id}/models/vex_high_stakes_ppo_agent_{i}"
        p = Process(target=train_agent, args=(VEXHighStakesEnv, total_timesteps, save_path, entropy, learning_rate, discount_factor))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All agents have been trained.")
