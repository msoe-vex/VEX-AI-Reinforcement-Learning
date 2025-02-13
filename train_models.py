import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from multiprocessing import Process
import os
import argparse
import numpy as np
import torch

# Ensure the rl_environment.py file is read correctly
with open('rl_environment.py', 'rb') as f:
    content = f.read()
    if b'\x00' in content:
        raise ValueError("The file rl_environment.py contains null bytes.")

from rl_environment import VEXHighStakesEnv

def evaluate_agent(model, env_class):
    # Create a fresh environment for evaluation
    env = env_class()
    obs, _ = env.reset()
    done = False
    # Run one episode using the trained model
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return env.total_score

def train_agent(env_class, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path):
    env = env_class()
    check_env(env, warn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path:
        # Load the provided pretrained model
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device)
    else:
        print("Creating new model")
        model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device)
    
    check_steps = 10000

    print(f"Training agent with entropy={entropy}, learning_rate={learning_rate}, discount_factor={discount_factor}")
    print(f"Evaluating agent every {check_steps} timesteps and saving to {save_path}")
    best_score = -np.inf
    n_steps = 0
    while n_steps < total_timesteps:
        model.learn(total_timesteps=check_steps, reset_num_timesteps=False)
        n_steps += check_steps
        score = evaluate_agent(model, env_class)
        if score > best_score:
            best_score = score
            model.save(save_path)
            print(f"Model saved to {save_path}.")
        print(f"Current best score: {best_score}")

    # Run evaluation to get an accurate final score
    score = evaluate_agent(model, env_class)
    print(f"Training complete.")
    print(f"Model saved to {save_path}.")
    print(f"Final score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to train')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training each agent')
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient for agent exploration')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Magnitude of updates to make to model')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Value to place on potential future rewards')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID for saving models')
    parser.add_argument('--model_path', type=str, default="", help='Path to a pretrained model')
    args = parser.parse_args()

    num_agents = args.agents
    total_timesteps = args.timesteps
    entropy = args.entropy
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    job_id = args.job_id
    model_path = args.model_path

    processes = []
    for i in range(num_agents):
        save_path = f"job_results/job_{job_id}/models/vex_high_stakes_ppo_agent_{i}"
        p = Process(target=train_agent, args=(VEXHighStakesEnv, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All agents have been trained.")
