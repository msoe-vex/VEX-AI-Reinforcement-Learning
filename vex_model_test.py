import torch
import argparse
import os
import numpy as np
import glob
from gymnasium import spaces

# Import from new modular architecture
from vex_core.base_env import VexMultiAgentEnv
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

def run_simulation(model_dir, game_name, output_dir):
    """
    Loads trained models and runs a simulation in the VEX environment.

    Args:
        model_dir (str): Path to the directory containing trained TorchScript models (.pt files).
        game_name (str): Name of the game variant to use.
        output_dir (str): Output directory for renders.
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return

    # Device Awareness: Automatically use GPU (CUDA) if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the environment using new modular architecture
    game = PushBackGame.get_game(game_name)
    env = VexMultiAgentEnv(
        game=game,
        render_mode="all", 
        output_directory=output_dir, 
        randomize=True
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

    print("Running simulation...")
    observations, infos = env.reset()
    env.clearStepsDirectory()
    
    env.render() # Initial render

    done = False
    step_count = 0 # Optional: for logging or a safety break

    last_actions = {agent: None for agent in env.possible_agents}  # Track last actions for each agent

    while not done:
        step_count += 1
        if not env.agents:  # No active agents left
            print(f"Step {step_count}: All agents are done (env.agents is empty). Ending simulation.")
            break

        actions_to_take = {}
        
        # Iterate over a copy of env.agents, as it might be modified by env.step
        current_agents_in_step = list(env.agents) 

        for agent_id in current_agents_in_step:
            if agent_id not in observations:
                print(f"Warning: Agent {agent_id} is in env.agents but not in observations. Skipping.")
                continue

            obs_np = observations[agent_id]
            # Zero-Copy Tensor Creation: Use from_numpy() for better performance
            obs_np_float = obs_np.astype(np.float32) if obs_np.dtype != np.float32 else obs_np
            obs_tensor = torch.from_numpy(obs_np_float).unsqueeze(0).to(device)

            # Use the agent-specific model
            model = models[agent_id]
            with torch.no_grad():
                model_output = model(obs_tensor)
                
            # Handle Tuple Action Space (Discrete + Box)
            # Output is concatenated: [DiscreteLogits(N), MessageMean(8), MessageLogStd(8)]
            # We assume N=5 (WAIT, need to check env.num_actions)
            # Better to infer from shape or env?
            # env.action_space(agent) is Tuple(Discrete(N), Box(8))
            action_space = env.action_space(agent_id)
            if isinstance(action_space, spaces.Tuple):
                num_actions = action_space[0].n
                # Slice:
                action_logits = model_output[:, :num_actions]
                message_params = model_output[:, num_actions:]
                # Message is Mean part (first 8 of remainder)
                # Box params = Mean(8) + LogStd(8) = 16
                message_mean = message_params[:, :8]
                
                # We use message_mean directly (deterministic message for inference?)
                # Or sample? For simplicity, use mean.
                message_vector = message_mean.cpu().numpy()[0]
            else:
                # Normal Discrete
                action_logits = model_output
                message_vector = None
            
            # CRITICAL: Use stochastic sampling like RLlib does during training!
            # The model outputs logits, and RLlib uses a Categorical distribution 
            # to sample actions based on softmax probabilities.
            # Using deterministic argmax (always picking highest logit) causes the
            # agent to get stuck in repetitive behavior patterns.
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Sample action and retry if invalid (up to 20 attempts)
            action = None
            for _ in range(20):
                sampled_action = dist.sample().item()
                if env.is_valid_action(sampled_action, obs_np, last_actions[agent_id]):
                    action = sampled_action
                    break
            
            # Fallback: if no valid action found via sampling, use highest valid logit
            if action is None:
                sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
                for candidate_action in sorted_actions:
                    if env.is_valid_action(candidate_action, obs_np, last_actions[agent_id]):
                        action = candidate_action
                        break
                else:
                    action = env.game.fallback_action()

            actions_to_take[agent_id] = (action, message_vector) if message_vector is not None else action
            last_actions[agent_id] = action  # Update last action for the agent (Control only?)
            # is_valid_action expects int, so we track int part.

        if not actions_to_take and env.agents:
            print(f"Step {step_count}: env.agents is not empty, but no actions were generated. This might indicate all remaining agents are terminating. Ending simulation.")
            # This state implies that all agents in env.agents might have been skipped or had no obs.
            # Or, if actions_to_take is empty because current_agents_in_step was empty (but env.agents was not, which is contradictory).
            # If env.agents is truly not empty, but we couldn't generate actions, it's an issue.
            # However, env.step({}) should correctly terminate if all agents are done.
            pass # Allow env.step({}) to be called, which should handle termination.
        
        if not env.agents and not actions_to_take: # If env.agents was already empty, loop condition caught it.
                                                 # If actions_to_take is empty because current_agents_in_step was empty.
            print(f"Step {step_count}: No active agents to take actions. Ending simulation.")
            break


        # Step the environment
        print(f"\nStep {step_count}: Scores: {env.score}")
        next_observations, step_rewards, terminations, truncations, infos = env.step(actions_to_take)
        
        env.render(actions=actions_to_take, rewards=step_rewards) # Render after the step

        observations = next_observations
        
        # Check for episode termination from environment signals
        # Ensure __all__ key exists, default to False if not (though it should always be present)
        all_terminated = terminations.get("__all__", False)
        all_truncated = truncations.get("__all__", False)
        done = all_terminated or all_truncated
            
    print(f"\nSimulation ended after {step_count} steps.")
    print("Creating GIF of the simulation...")
    env.createGIF()
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VEX environment simulation with trained models.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the directory containing trained TorchScript models (.pt files)."
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
    
    args = parser.parse_args()
    
    # Try to read game from metadata if not provided
    game_name = args.game
    if game_name is None:
        # Look for metadata in the model directory
        metadata_path = os.path.join(args.model_dir, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                game_name = metadata.get("game", "vexai_skills")
            print(f"Read game variant from metadata: {game_name}")
        else:
            game_name = "vexai_skills"
            print(f"No metadata found, using default game: {game_name}")
    
    run_simulation(args.model_dir, game_name, args.output_dir)
