import torch
import argparse
import os
import numpy as np

# Ensure vexEnv.py is accessible
# If vexEnvRun.py and vexEnv.py are in the same directory, this should work.
from vexEnv import Push_Back_Multi_Agent_Env
from vexEnv import Actions
from path_planner import PathPlanner

def is_valid_action(action, observation):
    """
    Check if the proposed action is valid given the current observation and last action.

    Args:
        env: The environment instance.
        action: The proposed action to validate.
        observation: The current observation for the agent.
        last_action: The last action taken by the agent.
    Returns:
        bool: True if the action is valid, False otherwise.
    """

    return True # TODO: implement actual validity checks based on env rules

def get_actions(model_path, observation):
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


    obs_np = observation # TODO: verify correct preprocessing
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action_logits = loaded_model(obs_tensor)
    # Sort actions by descending logit value (best first)
    sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
    # Find the first valid action
    action = None
    for candidate_action in sorted_actions:
        if is_valid_action(candidate_action, obs_np):
            action = candidate_action
            break
    
    if action is None:
        # Fallback: if no valid action found, pick top choice
        action = torch.argmax(action_logits, dim=1).item()


def split_action(action, observation):
    """
    Splits a combined action into individual agent actions based on validity.

    Args:
        action: The combined action output from the model.
        observation: The current observation for the agent.
    Returns:
        The valid action for the agent.
    """
    # Core actions that can be taken by the robot

    # FOLLOW;(0,0),(1,1),(2,2);speed
    # INTAKE;speed
    # TURN;degrees;speed
    # OUTTAKE;speed
    # DRIVE;12;30

    path_planner = PathPlanner(15,15,2,70,70)

    actions = []
    if action is Actions.PICK_UP_NEAREST_BLOCK.value:
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[0.5,0.5], obstacles=[]) # Will update later
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append("INTAKE;100")
        actions.append(f"FOLLOW;{points_str};50")
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")

    return action # TODO: implement actual splitting logic
