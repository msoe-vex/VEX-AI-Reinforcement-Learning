"""
VEX Core - Generic multi-agent VEX robotics environment.

This module provides the base infrastructure for VEX robotics
reinforcement learning environments. Game-specific implementations
(like Push Back) extend this core.

Note: VexMultiAgentEnv is not imported by default to avoid heavy
dependencies (ray, pandas, etc.) on lightweight runtime systems.
Import it explicitly: from vex_core.base_env import VexMultiAgentEnv
"""

from .base_game import VexGame, Robot, RobotSize, Team, ActionEvent, ActionStep

__all__ = ['VexGame', 'Robot', 'RobotSize', 'Team', 'ActionEvent', 'ActionStep']
