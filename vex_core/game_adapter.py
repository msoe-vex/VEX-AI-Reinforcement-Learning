"""
Abstract Game Adapter Interface
Bridges game simulation semantics with physical runtime execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BaseGameAdapter(ABC):
    """
    Abstract contract that each season's game must implement to run on the Jetson Nano.
    """

    def __init__(self, game: Optional[Any] = None, agent_name: str = "robot_0", team: str = "red"):
        self.game = game
        self.agent_name = agent_name
        self.team = team
        if game is not None and hasattr(game, "get_team_for_agent"):
            try:
                self.team = game.get_team_for_agent(agent_name)
            except Exception:
                pass

    @property
    @abstractmethod
    def game_name(self) -> str:
        """Return the game variant name (e.g. 'override', 'pushback')."""
        pass

    @property
    @abstractmethod
    def observation_schema(self) -> Dict[str, Any]:
        """JSON Schema dictionary describing the game-specific observation space."""
        pass

    # =========================================================================
    # Serialization & State Conversion
    # =========================================================================

    @abstractmethod
    def observation_dict_to_numpy(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert structured, human-readable state dict into flat 1D numpy vector
        for model inference.
        """
        pass

    @abstractmethod
    def numpy_to_observation_dict(self, obs_array: np.ndarray) -> Dict[str, Any]:
        """
        Convert flat 1D numpy observation vector into structured, inspectable state dict.
        """
        pass

    @abstractmethod
    def get_initial_state_dict(self) -> Dict[str, Any]:
        """
        Return a baseline initial observation dictionary with all required schema fields.
        """
        pass

    def write_back_to_dict(self, obs_dict: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write back or overlay updates (e.g. inferred states, model predictions)
        into the managed state dictionary. Mutates obs_dict in-place and returns it.
        """
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(obs_dict.get(k), dict):
                obs_dict[k].update(v)
            else:
                obs_dict[k] = v
        return obs_dict

    # =========================================================================
    # Action Decomposition & Translation
    # =========================================================================

    @abstractmethod
    def decompose_action(
        self,
        action_idx: int,
        current_state: Dict[str, Any]
    ) -> List[str]:
        """
        Translate a discrete high-level action into a list of executable V5 sub-actions.
        Example return:
            ["TURN_TO_POINT;(-18,18);80", "FOLLOW;(-18,18),(-48,47);90", "PICKUP_ITEM;(-48,47);false"]
        """
        pass

    # =========================================================================
    # State Transition Handlers (Feedback Loop)
    # =========================================================================

    @abstractmethod
    def on_action_success(
        self,
        action_idx: int,
        state: Dict[str, Any]
    ) -> None:
        """
        Mutates or updates the internal state tracker following successful action completion.
        (e.g., mark goal scored, clear held game elements).
        """
        pass

    @abstractmethod
    def on_action_error(
        self,
        action_idx: int,
        error_code: int,
        state: Dict[str, Any]
    ) -> None:
        """
        Mutates or corrects internal state tracker following action failure.
        """
        pass

    # =========================================================================
    # Intra-Option Step Verification (Option-Critic Readiness)
    # =========================================================================

    @abstractmethod
    def validate_subaction_precondition(
        self,
        action_idx: int,
        subaction_index: int,
        subaction_cmd: str,
        current_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a sub-action should proceed or terminate early.
        Returns (should_proceed, reason_if_aborted).
        """
        pass
