"""
VEX Push Back - Actions Module

Defines the action space for the VEX Push Back environment.
The PARK action dynamically uses the robot's team assignment.
"""

from enum import Enum


class Actions(Enum):
    """
    Available actions for robots in the VEX Push Back game.
    
    Scoring actions automatically approach the nearest goal end.
    PARK action dynamically routes to the correct park zone based on robot team.
    """
    PICK_UP_NEAREST_BLOCK = 0
    SCORE_IN_LONG_GOAL_TOP = 1    # Approaches nearest end of top long goal
    SCORE_IN_LONG_GOAL_BOTTOM = 2 # Approaches nearest end of bottom long goal
    SCORE_IN_CENTER_GOAL = 3
    DRIVE_TO_LOADER_TL = 4        # Top Left loader
    DRIVE_TO_LOADER_TR = 5        # Top Right loader
    DRIVE_TO_LOADER_BL = 6        # Bottom Left loader
    DRIVE_TO_LOADER_BR = 7        # Bottom Right loader
    CLEAR_LOADER = 8              # Dispense blocks from nearest loader
    PARK = 9                      # Dynamic: parks in team's zone (red/blue)
    IDLE = 10


# Action metadata for validation and UI
ACTION_DESCRIPTIONS = {
    Actions.PICK_UP_NEAREST_BLOCK: "Pick up the nearest block on the field",
    Actions.SCORE_IN_LONG_GOAL_TOP: "Score blocks in the top long goal",
    Actions.SCORE_IN_LONG_GOAL_BOTTOM: "Score blocks in the bottom long goal",
    Actions.SCORE_IN_CENTER_GOAL: "Score blocks in the center goal",
    Actions.DRIVE_TO_LOADER_TL: "Drive to the top-left loader",
    Actions.DRIVE_TO_LOADER_TR: "Drive to the top-right loader",
    Actions.DRIVE_TO_LOADER_BL: "Drive to the bottom-left loader",
    Actions.DRIVE_TO_LOADER_BR: "Drive to the bottom-right loader",
    Actions.CLEAR_LOADER: "Clear blocks from the nearest loader",
    Actions.PARK: "Park in the team's designated zone",
    Actions.IDLE: "Do nothing (wait)",
}


def get_loader_index_from_action(action: Actions) -> int:
    """
    Get the loader index (0-3) from a drive-to-loader action.
    
    Args:
        action: A DRIVE_TO_LOADER_* action
        
    Returns:
        Loader index (0=TL, 1=TR, 2=BL, 3=BR) or -1 if not a loader action
    """
    loader_actions = {
        Actions.DRIVE_TO_LOADER_TL: 0,
        Actions.DRIVE_TO_LOADER_TR: 1,
        Actions.DRIVE_TO_LOADER_BL: 2,
        Actions.DRIVE_TO_LOADER_BR: 3,
    }
    return loader_actions.get(action, -1)


def is_scoring_action(action: Actions) -> bool:
    """Check if an action is a scoring action."""
    return action in [
        Actions.SCORE_IN_LONG_GOAL_TOP,
        Actions.SCORE_IN_LONG_GOAL_BOTTOM,
        Actions.SCORE_IN_CENTER_GOAL,
    ]


def is_loader_action(action: Actions) -> bool:
    """Check if an action involves loaders."""
    return action in [
        Actions.DRIVE_TO_LOADER_TL,
        Actions.DRIVE_TO_LOADER_TR,
        Actions.DRIVE_TO_LOADER_BL,
        Actions.DRIVE_TO_LOADER_BR,
        Actions.CLEAR_LOADER,
    ]


def requires_held_blocks(action: Actions) -> bool:
    """Check if an action requires the robot to be holding blocks."""
    return is_scoring_action(action)
