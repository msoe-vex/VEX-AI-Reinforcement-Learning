"""VEX U Skills variant of Override."""

from typing import Dict, Optional

import numpy as np

from vex_core.config import CommunicationOption
from vex_core.robot import Robot, RobotSize, Team

from .override import OverrideGame


class VexUSkillsGame(OverrideGame):
	"""Two cooperating red robots playing a VEX U skills match."""

	def __init__(self, robots: Optional[list] = None,
				 communication_mode: CommunicationOption = CommunicationOption.NONE,
				 deterministic: bool = True,
				 use_24_inch_robots: bool = True):
		if robots is None:
			primary_size = RobotSize.INCH_24 if use_24_inch_robots else RobotSize.INCH_18
			robots = [
				Robot("red_robot_0", Team.RED, primary_size,
					  np.array([-48.0, 24.0], dtype=np.float32)),
				Robot("red_robot_1", Team.RED, RobotSize.INCH_18,
					  np.array([-48.0, -24.0], dtype=np.float32)),
			]
		super().__init__(robots, communication_mode=communication_mode,
						 deterministic=deterministic,
						 use_24_inch_robots=use_24_inch_robots)

	@property
	def total_time(self) -> float:
		return 60.0

	def get_team_for_agent(self, agent: str) -> str:
		return "red"

	def compute_score(self) -> Dict[str, int]:
		score = self._score_goal_pins().get("red", 0)
		score += sum(8 for agent in self.state["agents"].values()
					 if agent.get("parked_zone") == "midfield")
		if self.state.get("autonomous_winner") == "red":
			score += 12
		return {"red": score}


__all__ = ["VexUSkillsGame"]
