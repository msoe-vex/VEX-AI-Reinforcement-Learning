import torch
import argparse
import os
import numpy as np
import glob
import csv
from gymnasium import spaces

# Import from new modular architecture
from vex_core.base_env import VexMultiAgentEnv, MESSAGE_SIZE
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
    model_dir,
    game_name,
    output_dir,
    iterations=1,
    export_gif=True,
    render_mode="image",
    deterministic=True,
    communication_override=None,
    randomize=False,
):
    """
    Loads trained models and runs one or more simulations in the VEX environment.

    Args:
        model_dir (str): Path to the directory containing trained TorchScript models (.pt files).
        game_name (str): Name of the game variant to use.
        output_dir (str): Output directory for renders.
        iterations (int): Number of episodes to run.
        export_gif (bool): Whether to render frames and export GIF(s).
        render_mode (str): Mode for rendering: 'terminal', 'image', or 'none'.
        deterministic (bool): Whether to run deterministic environment mechanics.
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return

    # Device Awareness: Automatically use GPU (CUDA) if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read enable_communication from metadata
    enable_communication = False
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                enable_communication = metadata.get("enable_communication", False)
            print(f"Read enable_communication={enable_communication} from metadata")
        except Exception as e:
            print(f"Warning: Could not read enable_communication from metadata: {e}")

    if communication_override is not None:
        print(f"Overridden enable_communication={enable_communication} from arguments")

    # Initialize the game/environment with matching communication configuration
    game = PushBackGame.get_game(
        game_name,
        enable_communication=enable_communication,
        deterministic=deterministic,
    )
    
    env = VexMultiAgentEnv(
        game=game,
        render_mode=render_mode,
        output_directory=output_dir,
        randomize=randomize,
        enable_communication=enable_communication,
        deterministic=deterministic,
    )
    
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
    obs_shape = env.observation_space(sample_agent).shape
    
    # Warmup Pass: Initialize CUDA context before simulation starts
    print("Warming up models...")
    dummy_input = torch.randn(1, *obs_shape, device=device)
    with torch.no_grad():
        for agent_id, model in models.items():
            _ = model(dummy_input)
    print("Models warmed up")

    team_score_totals = {}
    team_score_counts = {}
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "results.csv")
    iteration_results = []

    for iteration in range(1, iterations + 1):
        print(f"\nRunning simulation iteration {iteration}/{iterations}...")
        observations, infos = env.reset()

        if export_gif:
            env.clearTicksDirectory()
            env.render()  # Initial render

        done = False
        step_count = 0
        last_actions = {agent: None for agent in env.possible_agents}

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

                obs_np = observations[agent_id]
                obs_np_float = obs_np.astype(np.float32) if obs_np.dtype != np.float32 else obs_np
                obs_tensor = torch.from_numpy(obs_np_float).unsqueeze(0).to(device)

                model = models[agent_id]
                with torch.no_grad():
                    model_output = model(obs_tensor)

                action_space = env.action_space(agent_id)
                if isinstance(action_space, spaces.Tuple):
                    num_actions = action_space[0].n
                    action_logits = model_output[:, :num_actions]
                    message_params = model_output[:, num_actions:]

                    message_vector = None
                    remaining = model_output.shape[1] - num_actions
                    if not communication_override: # If communication is disabled, set message vector to zeros
                        message_vector = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                    elif remaining >= MESSAGE_SIZE:
                        message_mean = model_output[:, num_actions:num_actions + MESSAGE_SIZE]
                        message_vector = message_mean.cpu().numpy()[0]
                    elif remaining > 0:
                        mm = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                        mm[:remaining] = model_output[:, num_actions:].cpu().numpy()[0]
                        message_vector = mm
                    else:
                        message_vector = np.zeros(MESSAGE_SIZE, dtype=np.float32)
                else:
                    num_actions = action_space.n
                    action_logits = model_output[:, :num_actions]
                    message_vector = None

                probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                action = None
                for _ in range(20):
                    sampled_action = dist.sample().item()
                    if env.is_valid_action(sampled_action, obs_np, last_actions[agent_id]):
                        action = sampled_action
                        break

                if action is None:
                    sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
                    for candidate_action in sorted_actions:
                        if env.is_valid_action(candidate_action, obs_np, last_actions[agent_id]):
                            action = candidate_action
                            break
                    else:
                        action = env.game.fallback_action()

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

            # Rendering is handled internally by env.step() during fast-forward

            observations = next_observations

            all_terminated = terminations.get("__all__", False)
            all_truncated = truncations.get("__all__", False)
            done = all_terminated or all_truncated

        print(
            f"Simulation iteration {iteration} ended after {step_count} steps "
            f"(env steps: {env.num_steps}, internal ticks: {env.num_ticks}). "
            f"Final score: {env.score}"
        )

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
            for team_name, score in env.score.items():
                row[f"score_{team_name}"] = float(score)
            iteration_results.append(row)
        else:
            iteration_results.append(
                {
                    "iteration": iteration,
                    "steps": step_count,
                    "env_steps": env.num_steps,
                    "internal_ticks": env.num_ticks,
                    "score": env.score,
                }
            )

        if export_gif:
            print("Creating GIF of the simulation...")
            env.createGIF()

    print("\nAverage team scores across all iterations:")
    if team_score_totals:
        for team_name in sorted(team_score_totals.keys()):
            avg_score = team_score_totals[team_name] / max(team_score_counts.get(team_name, 1), 1)
            print(f"  {team_name}: {avg_score:.3f}")
    else:
        print("  No per-team score dictionary found on env.score.")

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
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path to the experiment directory containing trained TorchScript models (.pt files)."
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Game variant (if not provided, reads from training_metadata.json in model directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vex_model_test",
        help="Output directory for renders"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of simulation iterations to run"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["terminal", "image", "none"],
        default="image",
        help="Rendering mode: 'image' (saves frames & GIF), 'terminal' (prints text only), 'none' (silent)"
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic environment mechanics (use --no-deterministic for stochastic outcomes)"
    )
    parser.add_argument(
        "--communication",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable communication (overrides metadata if provided)"
    )
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize initial agent positions and orientations"
    )
    
    args = parser.parse_args()
    
    # Try to read game from metadata if not provided
    game_name = args.game
    if game_name is None:
        # Look for metadata in the model directory
        metadata_path = os.path.join(args.experiment_path, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                game_name = metadata.get("game", "vexai_skills")
            print(f"Read game variant from metadata: {game_name}")
        else:
            game_name = "vexai_skills"
            print(f"No metadata found, using default game: {game_name}")
    
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")

    run_simulation(
        args.experiment_path,
        game_name,
        args.output_dir,
        iterations=args.iterations,
        export_gif = args.render_mode == "image",
        render_mode = args.render_mode if args.render_mode != "none" else None,
        deterministic=args.deterministic,
        communication_override=args.communication,
        randomize=args.randomize,
    )
