"""VEX AI Competition variant of Override.

The field geometry and game elements are inherited from :mod:`override`.
The explicit name keeps this variant separate from Pushback's VEX AI game.
"""

from typing import Dict, Optional

import numpy as np

from vex_core.config import CommunicationOption
from vex_core.robot import Robot, RobotSize, Team

from .override import (
    Actions,
    ObjectStatus,
    OverrideGame,
    VEXAI_GOAL_POSITIONS,
    VEXAI_GOAL_TYPES,
    UPPER_RIGHT_CUP_POSITION,
    UPPER_RIGHT_PIN_POSITIONS,
    LOWER_LEFT_CUP_POSITION,
    LOWER_LEFT_PIN_POSITIONS,
)


def limit_field_objects(state: Dict, max_pins: int, max_cups: int) -> Dict:
    """Keep the base Override positions while reducing active field objects."""
    kept = []
    field_counts = {"pin": 0, "cup": 0}
    for obj in state["objects"]:
        kind = obj["kind"]
        if obj["status"] == ObjectStatus.ON_FIELD and kind in field_counts:
            if field_counts[kind] >= {"pin": max_pins, "cup": max_cups}[kind]:
                continue
            field_counts[kind] += 1
        kept.append(obj)
    state["objects"] = kept
    return state


def remove_field_clusters(
    state: Dict,
    pin_positions: tuple,
    cup_positions: tuple,
) -> Dict:
    """Remove designated source clusters after shared setup is complete."""
    pin_positions = set(pin_positions)
    cup_positions = set(cup_positions)
    state["objects"] = [
        obj for obj in state["objects"]
        if not (
            (obj["kind"] == "pin" and obj.get("source_position") in pin_positions)
            or (obj["kind"] == "cup" and obj.get("source_position") in cup_positions)
        )
    ]
    return state


class VexAICompGame(OverrideGame):
    """Two-robot-per-alliance VEX AI Override match."""

    def __init__(
        self,
        robots: Optional[list] = None,
        communication_mode: CommunicationOption = CommunicationOption.NONE,
        deterministic: bool = True,
    ):
        if robots is None:
            robots = [
                Robot("red_robot_0", Team.RED, RobotSize.INCH_24,
                      np.array([-48.0, 24.0], dtype=np.float32)),
                Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                      np.array([-48.0, -24.0], dtype=np.float32)),
                Robot("blue_robot_0", Team.BLUE, RobotSize.INCH_24,
                      np.array([48.0, 24.0], dtype=np.float32)),
                Robot("blue_robot_1", Team.BLUE, RobotSize.INCH_15,
                      np.array([48.0, -24.0], dtype=np.float32)),
            ]
        super().__init__(
            robots, communication_mode=communication_mode, deterministic=deterministic
        )

    @property
    def total_time(self) -> float:
        return 120.0

    @property
    def goal_types(self) -> tuple:
        return VEXAI_GOAL_TYPES

    @property
    def goal_positions(self) -> Dict:
        return VEXAI_GOAL_POSITIONS

    @property
    def excluded_pin_positions(self) -> tuple:
        return tuple(UPPER_RIGHT_PIN_POSITIONS + LOWER_LEFT_PIN_POSITIONS)

    @property
    def excluded_cup_positions(self) -> tuple:
        return (UPPER_RIGHT_CUP_POSITION, LOWER_LEFT_CUP_POSITION)

    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        state = super().get_initial_state(randomize, seed)
        state = remove_field_clusters(
            state,
            self.excluded_pin_positions,
            self.excluded_cup_positions,
        )
        return limit_field_objects(state, 32, 32)

    def is_valid_action(self, agent: str, action: int, observation: np.ndarray) -> bool:
        """Only 24-inch robots may park in their alliance load zone."""
        if not super().is_valid_action(agent, action, observation):
            return False
        if self._decode_action(action) != Actions.PARK_MIDFIELD:
            return True
        return self.state["agents"][agent].get("robot_size") == RobotSize.INCH_24.value

    def compute_score(self) -> Dict[str, int]:
        """Apply the AI parking restriction while retaining Override scoring."""
        scores = super().compute_score()
        for agent in self.state["agents"].values():
            if agent.get("parked_zone") == "midfield" and agent.get("robot_size") != 24:
                scores[agent["team"]] -= 8
        return scores


__all__ = ["VexAICompGame"]
