import torch
import argparse
import os
import numpy as np

# Import from new modular architecture
from pushback import Actions
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
    # TURN_TO_POINT;(x,y);speed
    # DRIVE;inches;speed

    path_planner = PathPlanner(15,15,2,70,70)

    actions = []
    if action is Actions.PICK_UP_NEAREST_BLOCK.value:
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[0.5,0.5], obstacles=[]) # Will update later
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append("INTAKE;100")
        actions.append(f"FOLLOW;{points_str};50")
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")

    elif action is Actions.SCORE_IN_LONG_GOAL_1.value:
        # Approach one end of long goal 1, orient, and release block
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, 1.0], obstacles=[])  # arbitrary waypoint
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};60")   # drive to approach point
        actions.append("TURN;30;40")                 # turn appropriate amount toward goal end (arbitrary 30 deg)
        actions.append("DRIVE;6;40")                 # move forward into scoring position (arbitrary 6 inches)
        actions.append("INTAKE;0")                   # release

    elif action is Actions.SCORE_IN_LONG_GOAL_2.value:
        # Approach one end of long goal 2 (different location), orient, and release block
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, 1.0], obstacles=[])  # arbitrary waypoint
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};60")   # drive to approach point
        actions.append("TURN;-30;40")                # turn opposite direction for the other end (arbitrary -30 deg)
        actions.append("DRIVE;6;40")                 # move forward into scoring position (arbitrary 6 inches)
        actions.append("INTAKE;0")                   # release

    elif action is Actions.SCORE_IN_CENTER_UPPER.value:
        # Score into the upper center goal
        # Arbitrary waypoints along a diagonal through the center (to be updated)
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[1.5, 1.5], obstacles=[])  # arbitrary waypoint 
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};55")   # approach goal
        actions.append("TURN;45;40")                 # orient toward upper-center goal (arbitrary 45 deg)
        actions.append("DRIVE;4;40")                 # move into scoring position (arbitrary 4 inches)
        actions.append("INTAKE;0")                   # release

    elif action is Actions.SCORE_IN_CENTER_LOWER.value:
        # Score into the lower center goal
        # Arbitrary waypoints along the other diagonal through the center (to be updated)
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-1.5, 1.5], obstacles=[])  # arbitrary waypoint
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};55")   # approach goal
        actions.append("TURN;-45;40")                # orient toward lower-center goal (arbitrary -45 deg)
        actions.append("DRIVE;4;40")                 # move into scoring position (arbitrary 4 inches)
        actions.append("INTAKE;0")                   # release

    elif action is Actions.TAKE_FROM_LOADER_TL.value:
        # Take a block from top-left loader: drive in front, face loader, intake
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, 4.0], obstacles=[])  # TL loader location (arbitrary)
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};50")   # drive to approach point in front of loader
        actions.append("TURN;90;40")                 # orient to face loader (arbitrary 90 deg)
        actions.append("DRIVE;1;30")                 # inch forward to loader
        actions.append("INTAKE;100")                 # intake to grab block
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")                   # stop intake

    elif action is Actions.TAKE_FROM_LOADER_TR.value:
        # Take a block from top-right loader: drive in front, face loader, intake
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, 4.0], obstacles=[])   # TR loader location (arbitrary)
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};50")   # drive to approach point in front of loader
        actions.append("TURN;-90;40")                # orient to face loader (arbitrary -90 deg)
        actions.append("DRIVE;1;30")                 # inch forward to loader
        actions.append("INTAKE;100")                 # intake to grab block
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")                   # stop intake

    elif action is Actions.TAKE_FROM_LOADER_BL.value:
        # Take a block from bottom-left loader: drive in front, face loader, intake
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, -4.0], obstacles=[])  # BL loader location (arbitrary)
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};50")   # drive to approach point in front of loader
        actions.append("TURN;90;40")                 # orient to face loader (arbitrary 90 deg)
        actions.append("DRIVE;1;30")                 # inch forward to loader
        actions.append("INTAKE;100")                 # intake to grab block
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")                   # stop intake

    elif action is Actions.TAKE_FROM_LOADER_BR.value:
        # Take a block from bottom-right loader: drive in front, face loader, intake
        positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, -4.0], obstacles=[])   # BR loader location (arbitrary)
        points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

        actions.append(f"FOLLOW;{points_str};50")   # drive to approach point in front of loader
        actions.append("TURN;-90;40")                # orient to face loader (arbitrary -90 deg)
        actions.append("DRIVE;1;30")                 # inch forward to loader
        actions.append("INTAKE;100")                 # intake to grab block
        actions.append("WAIT;0.5")
        actions.append("INTAKE;0")                   # stop intake

    # Return the constructed sequence of low-level actions
    action = actions

    return action # TODO: implement actual splitting logic
