import torch
import argparse
import os
import numpy as np
import glob
import csv
import time
from gymnasium import spaces

# Import from new modular architecture
from vex_core.base_env import VexMultiAgentEnv, MESSAGE_SIZE
from vex_core.config import VexEnvConfig, CommunicationOption
from pushback import PushBackGame
import json

def load_agent_models(model_dir, agents, device):
    """
    Load models for each agent from the model directory.
    
    Args:
        model_dir: Directory containing .pt model files
        agents: List of agent IDs
        device: torch device to load models to
        
    Returns:
        Dict mapping agent_id to loaded model
    """
    models = {}
    
    for agent_id in agents:
        # Search for model files matching the agent_id
        # Try exact match first, then partial match
        patterns = [
            os.path.join(model_dir, f"{agent_id}.pt"),
            os.path.join(model_dir, f"{agent_id}_*.pt"),
            os.path.join(model_dir, f"*_{agent_id}.pt"),
            os.path.join(model_dir, f"*{agent_id}*.pt"),
        ]
        
        model_path = None
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                break
        
        if model_path is None:
            # Fall back to shared_policy.pt if no agent-specific model found
            shared_path = os.path.join(model_dir, "shared_policy.pt")
            if os.path.exists(shared_path):
                model_path = shared_path
                print(f"Using shared policy for agent {agent_id}")
            else:
                raise FileNotFoundError(f"No model found for agent {agent_id} in {model_dir}")
        else:
            print(f"Found model for agent {agent_id}: {model_path}")
        
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            model = torch.jit.optimize_for_inference(model)
            models[agent_id] = model
        except Exception as e:
            raise RuntimeError(f"Error loading model for agent {agent_id}: {e}")
    
    return models

def run_simulation(
    config: VexEnvConfig,
    iterations=1,
    test_communication_mode=None,
):
    """
    Loads trained models and runs one or more simulations in the VEX environment.

    Args:
        config (VexEnvConfig): Configuration object containing parameters.
        iterations (int): Number of episodes to run.
        test_communication_mode (CommunicationOption): Whether to test communication. Only used if config.communication_mode is not NONE.
    """
    model_dir = config.experiment_path
    output_dir = config.experiment_path if config.experiment_path else os.path.join(os.getcwd(), "vex_model_test")
    render_mode = config.render_mode

    if test_communication_mode is None:
        test_communication_mode = config.communication_mode
    
    use_random_actions = model_dir is None or str(model_dir).strip() == ""
    if not use_random_actions and not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return

    # Device Awareness: Automatically use GPU (CUDA) if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the game/environment with matching communication configuration
    game = PushBackGame.get_game(
        config.game_name,
        communication_mode=config.communication_mode,
        deterministic=config.deterministic,
    )
    
    env = VexMultiAgentEnv(
        game=game,
        config=config,
    )
    
    models = {}
    if use_random_actions:
        print("No experiment-path provided. Running random-action baseline test (no model loading).")
    else:
        # Load models for each agent
        print(f"Loading models from {model_dir}...")
        try:
            models = load_agent_models(model_dir, env.possible_agents, device)
            print(f"Successfully loaded models for {len(models)} agents")
        except Exception as e:
            print(f"Error loading models: {e}")
            return
    
    # Get observation and action space shapes for a sample agent
    # These are used to correctly shape tensors for the model
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_space(sample_agent)
    if hasattr(obs_space, 'spaces') and 'observations' in obs_space.spaces:
        obs_shape = obs_space['observations'].shape
    else:
        obs_shape = obs_space.shape
    
    # Determine action mask dim for warmup/random fallback
    sample_act_space = env.action_space(sample_agent)
    if isinstance(sample_act_space, spaces.Tuple):
        mask_dim = sample_act_space[0].n
    elif hasattr(sample_act_space, 'n'):
        mask_dim = sample_act_space.n
    else:
        mask_dim = obs_shape[0]

    if not use_random_actions:
        # Warmup Pass: Initialize CUDA context before simulation starts
        print("Warming up models...")
        dummy_input = torch.randn(1, *obs_shape, device=device)
        dummy_mask = torch.ones(1, mask_dim, device=device)
        with torch.no_grad():
            for agent_id, model in models.items():
                _ = model(dummy_input, dummy_mask)
        print("Models warmed up")

    team_score_totals = {}
    team_score_counts = {}
    total_reward_totals = {}
    total_reward_counts = {}
    inference_time_total_s = 0.0
    inference_call_count = 0
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "results.csv")
    iteration_results = []

    for iteration in range(1, iterations + 1):
        print(f"\nRunning simulation iteration {iteration}/{iterations}...")
        observations, infos = env.reset()

        if render_mode == "image":
            env.clearTicksDirectory()
            env.render()  # Initial render

        done = False
        step_count = 0
        last_actions = {agent: None for agent in env.possible_agents}

        total_reward = 0
        iteration_inference_time_s = 0.0
        iteration_inference_calls = 0

        while not done:
            step_count += 1
            if not env.agents:
                if render_mode in ["terminal", "image"]:
                    print(f"Step {step_count}: All agents are done (env.agents is empty). Ending simulation.")
                break

            actions_to_take = {}
            current_agents_in_step = list(env.agents)

            for agent_id in current_agents_in_step:
                if agent_id not in observations:
                    if render_mode in ["terminal", "image"]:
                        print(f"Warning: Agent {agent_id} is in env.agents but not in observations. Skipping.")
                    continue

                obs_dict = observations[agent_id]
                obs_np = obs_dict["observations"] if isinstance(obs_dict, dict) and "observations" in obs_dict else obs_dict

                # If communication_mode is copy but test_communication_mode is not NONE, fill the second half of the observation vector with zeros
                if config.communication_mode == CommunicationOption.COPY and test_communication_mode is not CommunicationOption.COPY:
                    obs_np[0:obs_np.shape[0] // 2] = 0

                action_mask_np = obs_dict["action_mask"] if isinstance(obs_dict, dict) and "action_mask" in obs_dict else None

                if use_random_actions:
                    if action_mask_np is not None:
                        valid_actions = np.flatnonzero(action_mask_np > 0.5)
                    else:
                        action_space = env.action_space(agent_id)
                        if isinstance(action_space, spaces.Tuple):
                            valid_actions = np.arange(action_space[0].n)
                        else:
                            valid_actions = np.arange(action_space.n)

                    if valid_actions.size == 0:
                        action = env.game.fallback_action
                    else:
                        action = int(np.random.choice(valid_actions))

                    action_space = env.action_space(agent_id)
                    if isinstance(action_space, spaces.Tuple):
                        if test_communication_mode == CommunicationOption.NONE:
                            message_vector = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                        else:
                            message_vector = np.random.uniform(-1.0, 1.0, size=(MESSAGE_SIZE,)).astype(np.float32)
                        actions_to_take[agent_id] = (action, message_vector)
                    else:
                        actions_to_take[agent_id] = action

                    last_actions[agent_id] = action
                    continue

                obs_np_float = obs_np.astype(np.float32) if obs_np.dtype != np.float32 else obs_np
                obs_tensor = torch.from_numpy(obs_np_float).unsqueeze(0).to(device)

                # # Print observation
                # print(f"Observation for agent {agent_id}: {obs_np}")
                
                # Build action mask tensor
                if action_mask_np is not None:
                    mask_tensor = torch.from_numpy(action_mask_np.astype(np.float32)).unsqueeze(0).to(device)
                else:
                    mask_tensor = torch.ones(1, mask_dim, device=device)

                model = models[agent_id]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    model_output = model(obs_tensor, mask_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_elapsed = time.perf_counter() - inference_start
                inference_time_total_s += inference_elapsed
                inference_call_count += 1
                iteration_inference_time_s += inference_elapsed
                iteration_inference_calls += 1

                action_space = env.action_space(agent_id)
                if isinstance(action_space, spaces.Tuple):
                    num_actions = action_space[0].n
                    action_logits = model_output[:, :num_actions]
                    message_params = model_output[:, num_actions:]

                    message_vector = None
                    remaining = model_output.shape[1] - num_actions
                    if test_communication_mode == CommunicationOption.NONE: # If communication is explicitly disabled
                        message_vector = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                    elif remaining >= MESSAGE_SIZE:
                        message_mean = model_output[:, num_actions:num_actions + MESSAGE_SIZE]
                        # Apply temperature to message sampling std (T>1 -> more random)
                        temperature = max(1e-6, getattr(config, "temperature", 1.0))

                        if not config.deterministic and remaining >= 2 * MESSAGE_SIZE:
                            msg_log_std = model_output[:, num_actions + MESSAGE_SIZE:num_actions + 2 * MESSAGE_SIZE]
                            msg_std = torch.exp(msg_log_std) * temperature
                            msg_dist = torch.distributions.Normal(message_mean, msg_std)
                            message_vector = msg_dist.sample().cpu().numpy()[0]
                        else:
                            message_vector = message_mean.cpu().numpy()[0]
                    elif remaining > 0:
                        mm = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                        mm[:remaining] = model_output[:, num_actions:].cpu().numpy()[0]
                        message_vector = mm
                    else:
                        message_vector = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                    # Temperature scaling for action selection
                    temperature = max(1e-6, getattr(config, "temperature", 1.0))
                    scaled_logits = action_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                else:
                    num_actions = action_space.n
                    action_logits = model_output[:, :num_actions]
                    # Temperature scaling for action selection
                    temperature = max(1e-6, getattr(config, "temperature", 1.0))
                    scaled_logits = action_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    message_vector = None
                
                action_dim = num_actions

                if not config.deterministic:
                    action_probs = probs.squeeze(0).cpu().numpy().copy()
                    
                    prob_sum = action_probs.sum()
                    if prob_sum > 0:
                        action_probs = action_probs / prob_sum
                        action = np.random.choice(action_dim, p=action_probs)
                    else:
                        action = env.game.fallback_action
                        
                    # If the model still somehow chooses an invalid action, fallback gracefully
                    if not env.is_valid_action(agent_id, action, obs_np):
                        action = env.game.fallback_action
                else:
                    # Deterministic: take highest probability valid action
                    # Use same temperature-scaled logits for deterministic selection
                    sorted_actions = torch.argsort(scaled_logits, dim=1, descending=True).squeeze(0).tolist()
                    action = None
                    for candidate_action in sorted_actions:
                        if env.is_valid_action(agent_id, candidate_action, obs_np, last_actions[agent_id]):
                            action = candidate_action
                            break
                    if action is None:
                        action = env.game.fallback_action

                actions_to_take[agent_id] = (action, message_vector) if message_vector is not None else action
                last_actions[agent_id] = action

            if not actions_to_take and env.agents:
                if render_mode in ["terminal", "image"]:
                    print(f"Step {step_count}: env.agents is not empty, but no actions were generated. This might indicate all remaining agents are terminating. Ending simulation.")

            if not env.agents and not actions_to_take:
                if render_mode in ["terminal", "image"]:
                    print(f"Step {step_count}: No active agents to take actions. Ending simulation.")
                break

            next_observations, step_rewards, terminations, truncations, infos = env.step(actions_to_take)

            for agent_id, reward in step_rewards.items():
                total_reward += reward

            # Rendering is handled internally by env.step() during fast-forward

            observations = next_observations

            all_terminated = terminations.get("__all__", False)
            all_truncated = truncations.get("__all__", False)
            done = all_terminated or all_truncated

        print(
            f"Simulation iteration {iteration} ended after {step_count} steps "
            f"(env steps: {env.num_steps}, internal ticks: {env.num_ticks}). "
            f"Final score: {env.score} "
            f"Total reward: {total_reward}"
        )
        if iteration_inference_calls > 0:
            iter_avg_inference_ms = (iteration_inference_time_s / iteration_inference_calls) * 1000.0
            print(f"Average inference time (iteration {iteration}): {iter_avg_inference_ms:.3f} ms over {iteration_inference_calls} calls")

        if isinstance(env.score, dict):
            for team_name, score in env.score.items():
                team_score_totals[team_name] = team_score_totals.get(team_name, 0.0) + float(score)
                team_score_counts[team_name] = team_score_counts.get(team_name, 0) + 1

            row = {
                "iteration": iteration,
                "steps": step_count,
                "env_steps": env.num_steps,
                "internal_ticks": env.num_ticks,
            }
            if iteration_inference_calls > 0:
                row["avg_inference_ms"] = (iteration_inference_time_s / iteration_inference_calls) * 1000.0
            for team_name, score in env.score.items():
                row[f"score_{team_name}"] = float(score)
            iteration_results.append(row)
        else:
            row = {
                "iteration": iteration,
                "steps": step_count,
                "env_steps": env.num_steps,
                "internal_ticks": env.num_ticks,
                "score": env.score,
            }
            if iteration_inference_calls > 0:
                row["avg_inference_ms"] = (iteration_inference_time_s / iteration_inference_calls) * 1000.0
            iteration_results.append(row)
        
        total_reward_totals[team_name] = total_reward_totals.get(team_name, 0.0) + float(total_reward)
        total_reward_counts[team_name] = total_reward_counts.get(team_name, 0) + 1

        if render_mode == "image":
            print("Creating GIF of the simulation...")
            env.createGIF()

    print("\nAverage team scores across all iterations:")
    if team_score_totals:
        for team_name in sorted(team_score_totals.keys()):
            avg_score = team_score_totals[team_name] / max(team_score_counts.get(team_name, 1), 1)
            print(f"  {team_name}: {avg_score:.3f}")
    else:
        print("  No per-team score dictionary found on env.score.")

    print("\nAverage total rewards across all iterations:")
    if total_reward_totals:
        for team_name in sorted(total_reward_totals.keys()):
            avg_reward = total_reward_totals[team_name] / max(total_reward_counts.get(team_name, 1), 1)
            print(f"  {team_name}: {avg_reward:.3f}")
    else:
        print("  No per-team total reward dictionary found on env.score.")

    print("\nAverage model inference time:")
    if inference_call_count > 0:
        avg_inference_ms = (inference_time_total_s / inference_call_count) * 1000.0
        print(f"  {avg_inference_ms:.3f} ms over {inference_call_count} calls")
    else:
        print("  No model inference calls were made (random-action mode).")

    if iteration_results:
        fieldnames = []
        for row in iteration_results:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(results_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in iteration_results:
                writer.writerow(row)
        print(f"Saved per-iteration results to: {results_csv_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VEX environment simulation with trained models.")
    VexEnvConfig.add_cli_args(
        parser,
        game=None,
        render_mode="image",
        experiment_path="",
        randomize=False,
        communication_mode=None,
        deterministic=False
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of simulation iterations to run"
    )
    parser.add_argument(
        "--test-communication-mode",
        type=str,
        choices=[opt.value for opt in CommunicationOption],
        help="Enable testing communication between agents",
    )
    
    args = parser.parse_args()
    config = VexEnvConfig.from_args(args)
    

    test_communication_mode_val = None
    if hasattr(args, "test_communication_mode") and args.test_communication_mode is not None:
        test_communication_mode_val = CommunicationOption(args.test_communication_mode)
    
    run_simulation(
        config=config,
        iterations=args.iterations,
        test_communication_mode=test_communication_mode_val,
    )
