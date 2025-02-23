import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from multiprocessing import Process
import os
import argparse
import numpy as np
import torch as th

# Ensure the rl_environment.py file is read correctly
with open('rl_environment.py', 'rb') as f:
    content = f.read()
    if b'\x00' in content:
        raise ValueError("The file rl_environment.py contains null bytes.")

from rl_environment import VEXHighStakesEnv

def evaluate_agent(model, env_class, randomize_positions):
    # Create a fresh environment for evaluation
    env = env_class(randomize_positions=randomize_positions)
    obs, _ = env.reset()
    done = False
    # Run one episode using the trained model
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return env.total_score

def train_agent(env_class, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path, randomize_positions, num_layers, num_nodes):
    env = env_class(randomize_positions=randomize_positions)
    check_env(env, warn=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Define the network architecture
    net_arch = [num_nodes] * num_layers
    
    # Define policy keyword arguments
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=th.nn.ReLU,  # Activation function
    )
    
    if model_path:
        # Load the provided pretrained model
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device)
    else:
        print("Creating new model")
        model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device, policy_kwargs=policy_kwargs)
    
    check_steps = 10000
    print(f"Evaluating agent every {check_steps} timesteps and saving to {save_path}")
    
    best_score = -np.inf
    initial_timesteps = model.num_timesteps
    n_steps = 0
    while n_steps < total_timesteps:
        model.learn(total_timesteps=check_steps, reset_num_timesteps=False)
        n_steps = model.num_timesteps - initial_timesteps
        score = evaluate_agent(model, env_class, randomize_positions)
        if score > best_score:
            best_score = score
            model.save(save_path)
            print(f"Model saved to {save_path}.")
        print(f"Current best score: {best_score}")
        print(f"Current timestep: {n_steps}/{total_timesteps}")

    # Run evaluation to get an accurate final score
    score = evaluate_agent(model, env_class, randomize_positions)
    model.save(save_path+"_final")
    print(f"Training complete.")
    print(f"Model saved to {save_path}_final.")
    print(f"Final score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--agents', type=int, required=True, help='Number of agents to train')
    parser.add_argument('--timesteps', type=int, required=True, help='Total timesteps for training each agent')
    parser.add_argument('--entropy', type=float, required=True, help='Entropy coefficient for agent exploration')
    parser.add_argument('--learning_rate', type=float, required=True, help='Magnitude of updates to make to model')
    parser.add_argument('--discount_factor', type=float, required=True, help='Value to place on potential future rewards')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID for saving models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pretrained model')
    parser.add_argument('--randomize', action='store_true', help='Randomize positions in the environment')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions in the environment')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the policy network')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes per layer in the policy network')
    parser.set_defaults(randomize=True)
    args = parser.parse_args()

    print(f"Training with the following arguments:")
    print(f"Number of agents: {args.agents}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Entropy coefficient: {args.entropy}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Discount factor: {args.discount_factor}")
    print(f"Job ID: {args.job_id}")
    print(f"Model path: {args.model_path}")
    print(f"Randomize positions: {args.randomize}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Number of nodes: {args.num_nodes}")

    num_agents = args.agents
    total_timesteps = args.timesteps
    entropy = args.entropy
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    job_id = args.job_id
    model_path = args.model_path
    randomize_positions = args.randomize
    num_layers = args.num_layers
    num_nodes = args.num_nodes

    processes = []
    for i in range(num_agents):
        save_path = f"job_results/job_{job_id}/models/vex_high_stakes_ppo_agent_{i}"
        p = Process(target=train_agent, args=(VEXHighStakesEnv, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path, randomize_positions, num_layers, num_nodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All agents have been trained.")
