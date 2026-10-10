"""VEX AI Skills variant of Override."""

from typing import Dict, Optional

import numpy as np

from vex_core.config import CommunicationOption
from vex_core.robot import Robot, RobotSize, Team

from .override import OverrideGame, VEXAI_GOAL_TYPES
from .vexai_comp import limit_field_objects


class VexAISkillsGame(OverrideGame):
    """Two cooperating VEX AI robots playing a 60-second Skills match."""

    def __init__(
        self,
        robots: Optional[list] = None,
        communication_mode: CommunicationOption = CommunicationOption.NONE,
        deterministic: bool = True,
    ):
        if robots is None:
            robots = [
                Robot("red_robot_0", Team.RED, RobotSize.INCH_24,
                      np.array([48.0, 24.0], dtype=np.float32)),
                Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
                      np.array([-48.0, -24.0], dtype=np.float32)),
            ]
        super().__init__(
            robots, communication_mode=communication_mode, deterministic=deterministic
        )

    @property
    def total_time(self) -> float:
        return 60.0

    @property
    def goal_types(self) -> tuple:
        return VEXAI_GOAL_TYPES

    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        return limit_field_objects(super().get_initial_state(randomize, seed), 24, 24)

    def get_team_for_agent(self, _agent: str) -> str:
        return "red"

    def compute_score(self) -> Dict[str, int]:
        """Score the shared red alliance using the base Override elements."""
        score = self._score_goal_pins().get("red", 0)
        score += sum(
            8
            for agent in self.state["agents"].values()
            if agent.get("parked_zone") == "midfield"
        )
        if self.state.get("autonomous_winner") == "red":
            score += 12
        return {"red": score}


__all__ = ["VexAISkillsGame"]
