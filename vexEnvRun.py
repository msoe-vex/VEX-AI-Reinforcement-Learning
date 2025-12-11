import torch
import argparse
import os
import numpy as np

# Import from new modular architecture
from vex_core import VexMultiAgentEnv
from pushback import VexUSkillsGame

def run_simulation(model_path):
    """
    Loads a trained model and runs a simulation in the VEX environment.

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

    # Initialize the environment using new modular architecture
    game = VexUSkillsGame()
    env = VexMultiAgentEnv(
        game=game,
        render_mode="all", 
        output_directory="vexEnvRun", 
        randomize=False
    )
    
    # Get observation and action space shapes for a sample agent
    # These are used to correctly shape tensors for the model
    sample_agent = env.possible_agents[0]
    obs_shape = env.observation_space(sample_agent).shape
    act_shape = env.action_space(sample_agent).shape # For Discrete, this is ()

    print("Starting simulation...")
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
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_logits = loaded_model(obs_tensor)
            # Sort actions by descending logit value (best first)
            sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
            # Find the first valid action
            for candidate_action in sorted_actions:
                if env.is_valid_action(candidate_action, obs_np, last_actions[agent_id]):
                    action = candidate_action
                    break
            else:
                # Fallback: if no valid action found, pick top choice
                action = torch.argmax(action_logits, dim=1).item()

            actions_to_take[agent_id] = action
            last_actions[agent_id] = action  # Update last action for the agent

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
    parser = argparse.ArgumentParser(description="Run VEX environment simulation with a trained model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained TorchScript model (.pt file, e.g., traced_model.pt)."
    )
    
    args = parser.parse_args()
    
    run_simulation(args.model_path)
