"""
VEX Core - Generic multi-agent VEX robotics environment.

This module provides the base infrastructure for VEX robotics
reinforcement learning environments. Game-specific implementations
(like Push Back) extend this core.
"""

from .base_env import VexMultiAgentEnv
from .base_game import VexGame

__all__ = ['VexMultiAgentEnv', 'VexGame']
