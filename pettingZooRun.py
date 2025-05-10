import torch
import argparse
import os
import numpy as np

# Ensure pettingZooEnv.py is accessible
# If runPettingZoo.py and pettingZooEnv.py are in the same directory, this should work.
from pettingZooEnv import High_Stakes_Multi_Agent_Env, POSSIBLE_AGENTS

def run_simulation(model_path):
    """
    Loads a trained model and runs a simulation in the High_Stakes_Multi_Agent_Env.

    Args:
        model_path (str): Path to the trained TorchScript model (.pt file).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the TorchScript model
    try:
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()  # Set the model to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the environment
    # The render_mode "human" is assumed to be available in your High_Stakes_Multi_Agent_Env
    env = High_Stakes_Multi_Agent_Env(render_mode="all", output_directory="pettingZooRun")
    
    # Get observation and action space shapes for a sample agent
    # These are used to correctly shape tensors for the model
    sample_agent = env.possible_agents[0]
    obs_shape = env.observation_space(sample_agent).shape
    act_shape = env.action_space(sample_agent).shape # For Discrete, this is ()

    print("Starting simulation...")
    observations, infos = env.reset()
    
    env.render() # Initial render

    done = False
    step_count = 0 # Optional: for logging or a safety break

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
                # This can happen if an agent terminated in the previous step and env.agents
                # wasn't updated yet, or if observations are stale.
                # Should ideally not happen if observations are fresh from previous step.
                print(f"Warning: Agent {agent_id} is in env.agents but not in observations. Skipping.")
                continue

            obs_np = observations[agent_id]
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_logits = loaded_model(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
            actions_to_take[agent_id] = action

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
        next_observations, step_rewards, terminations, truncations, infos = env.step(actions_to_take)
        
        env.render() # Render after the step

        observations = next_observations
        
        # Check for episode termination from environment signals
        # Ensure __all__ key exists, default to False if not (though it should always be present)
        all_terminated = terminations.get("__all__", False)
        all_truncated = truncations.get("__all__", False)
        done = all_terminated or all_truncated

        if done:
            print(f"Step {step_count}: Episode finished (all_terminated: {all_terminated}, all_truncated: {all_truncated}).")
            
    print(f"\nSimulation ended after {step_count} steps.")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PettingZoo simulation with a trained model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained TorchScript model (.pt file, e.g., traced_model.pt)."
    )
    
    args = parser.parse_args()
    
    run_simulation(args.model_path)
