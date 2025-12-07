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
    SCORE_IN_LONG_GOAL_1 = 1      # Long goal 1 (y=48)
    SCORE_IN_LONG_GOAL_2 = 2      # Long goal 2 (y=-48)
    SCORE_IN_CENTER_UPPER = 3     # Center goal upper (UL to LR diagonal)
    SCORE_IN_CENTER_LOWER = 4     # Center goal lower (LL to UR diagonal)
    TAKE_FROM_LOADER_TL = 5       # Take block from Top Left loader
    TAKE_FROM_LOADER_TR = 6       # Take block from Top Right loader
    TAKE_FROM_LOADER_BL = 7       # Take block from Bottom Left loader
    TAKE_FROM_LOADER_BR = 8       # Take block from Bottom Right loader
    CLEAR_LOADER = 9              # Dispense blocks from nearest loader
    PARK = 10                     # Dynamic: parks in team's zone (red/blue)
    TURN_TOWARD_CENTER = 11       # Turn to face center of field (0, 0)
    IDLE = 12


# Action metadata for validation and UI
ACTION_DESCRIPTIONS = {
    Actions.PICK_UP_NEAREST_BLOCK: "Pick up the nearest block on the field",
    Actions.SCORE_IN_LONG_GOAL_1: "Score blocks in long goal 1 (top)",
    Actions.SCORE_IN_LONG_GOAL_2: "Score blocks in long goal 2 (bottom)",
    Actions.SCORE_IN_CENTER_UPPER: "Score blocks in center goal upper (45°)",
    Actions.SCORE_IN_CENTER_LOWER: "Score blocks in center goal lower (-45°)",
    Actions.TAKE_FROM_LOADER_TL: "Take block from the top-left loader",
    Actions.TAKE_FROM_LOADER_TR: "Take block from the top-right loader",
    Actions.TAKE_FROM_LOADER_BL: "Take block from the bottom-left loader",
    Actions.TAKE_FROM_LOADER_BR: "Take block from the bottom-right loader",
    Actions.CLEAR_LOADER: "Clear blocks from the nearest loader",
    Actions.PARK: "Park in the team's designated zone",
    Actions.TURN_TOWARD_CENTER: "Turn to face the center of the field",
    Actions.IDLE: "Do nothing (wait)",
}


def get_loader_index_from_action(action: Actions) -> int:
    """
    Get the loader index (0-3) from a take-from-loader action.
    
    Args:
        action: A TAKE_FROM_LOADER_* action
        
    Returns:
        Loader index (0=TL, 1=TR, 2=BL, 3=BR) or -1 if not a loader action
    """
    loader_actions = {
        Actions.TAKE_FROM_LOADER_TL: 0,
        Actions.TAKE_FROM_LOADER_TR: 1,
        Actions.TAKE_FROM_LOADER_BL: 2,
        Actions.TAKE_FROM_LOADER_BR: 3,
    }
    return loader_actions.get(action, -1)


def is_scoring_action(action: Actions) -> bool:
    """Check if an action is a scoring action."""
    return action in [
        Actions.SCORE_IN_LONG_GOAL_1,
        Actions.SCORE_IN_LONG_GOAL_2,
        Actions.SCORE_IN_CENTER_UPPER,
        Actions.SCORE_IN_CENTER_LOWER,
    ]


def is_loader_action(action: Actions) -> bool:
    """Check if an action involves loaders."""
    return action in [
        Actions.TAKE_FROM_LOADER_TL,
        Actions.TAKE_FROM_LOADER_TR,
        Actions.TAKE_FROM_LOADER_BL,
        Actions.TAKE_FROM_LOADER_BR,
        Actions.CLEAR_LOADER,
    ]


def requires_held_blocks(action: Actions) -> bool:
    """Check if an action requires the robot to be holding blocks."""
    return is_scoring_action(action)


