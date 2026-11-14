"""
VEX Core - Generic multi-agent VEX robotics environment.

This module provides the base infrastructure for VEX robotics
reinforcement learning environments. Game-specific implementations
(like Push Back) extend this core.

Note: VexMultiAgentEnv is not imported by default to avoid heavy
dependencies (ray, pandas, etc.) on lightweight runtime systems.
Import it explicitly: from vex_core.base_env import VexMultiAgentEnv
"""

from .base_game import VexGame, ActionEvent, ActionStep
from .robot import Robot, RobotSize, Team
from .path_planner import PathPlanner
from .game_adapter import BaseGameAdapter
from .factory import GameAdapterFactory
from .converters import json_to_numpy, numpy_to_json, get_default_observation_dict

__all__ = [
    'VexGame',
    'Robot',
    'RobotSize',
    'Team',
    'ActionEvent',
    'ActionStep',
    'PathPlanner',
    'BaseGameAdapter',
    'GameAdapterFactory',
    'json_to_numpy',
    'numpy_to_json',
    'get_default_observation_dict',
]
