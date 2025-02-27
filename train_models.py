import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from multiprocessing import Process
import os
import argparse
import numpy as np
import torch as th

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
def evaluate_agent(model, env_class, save_path, randomize_positions):
    env = env_class(save_path, randomize_positions=randomize_positions)
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
def train_agent(env_class, total_timesteps, save_path, entropy, learning_rate, discount_factor, model_path, randomize_positions, num_layers, num_nodes):
    env = env_class(save_path, randomize_positions=randomize_positions)
    check_env(env, warn=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Define network architecture and policy parameters.
    net_arch = [num_nodes] * num_layers
    policy_kwargs = dict(net_arch=net_arch, activation_fn=th.nn.ReLU)
    
    if model_path:
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1, ent_coef=entropy, learning_rate=learning_rate, gamma=discount_factor, device=device)
    else:
        print("Creating new model")
        model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=entropy, learning_rate=learning_rate,
                    gamma=discount_factor, device=device, policy_kwargs=policy_kwargs)
    
    check_steps = 10000
    print(f"Evaluating every {check_steps} timesteps; saving to {save_path}")
    best_score = -np.inf
    initial_timesteps = model.num_timesteps
    n_steps = 0
    while n_steps < total_timesteps:
        model.learn(total_timesteps=check_steps, reset_num_timesteps=False)
        n_steps = model.num_timesteps - initial_timesteps
        score = evaluate_agent(model, env_class, save_path, randomize_positions)
        if score > best_score:
            best_score = score
            model.save(save_path)
            print(f"Model saved to {save_path}.")
        print(f"Best: {best_score} | {n_steps}/{total_timesteps} timesteps")
    score = evaluate_agent(model, env_class, save_path, randomize_positions)
    model.save(save_path+"_final")
    print("Training complete.")
    print(f"Final model saved to {save_path}_final with score {score}")

# ---------------------------------------------------------------------------
# Main: Parse arguments and train agents concurrently.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--agents', type=int, required=True, help='Number of agents to train')
    parser.add_argument('--timesteps', type=int, required=True, help='Total timesteps for training each agent')
    parser.add_argument('--entropy', type=float, required=True, help='Entropy coefficient')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--discount_factor', type=float, required=True, help='Discount factor')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID for saving models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pretrained model')
    parser.add_argument('--randomize', action='store_true', help='Randomize positions')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of network layers')
    parser.add_argument('--num_nodes', type=int, required=True, help='Nodes per layer')
    parser.set_defaults(randomize=True)
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

    processes = []
    for i in range(args.agents):
        save_path = f"job_results/job_{args.job_id}/models/model_{i}"
        p = Process(target=train_agent, args=(VEXHighStakesEnv, args.timesteps, save_path, args.entropy, args.learning_rate, args.discount_factor, args.model_path, args.randomize, args.num_layers, args.num_nodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All agents trained.")
