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
				 deterministic: bool = True):
		if robots is None:
			robots = [
				Robot("red_robot_0", Team.RED, RobotSize.INCH_24,
					  np.array([-48.0, 24.0], dtype=np.float32)),
				Robot("red_robot_1", Team.RED, RobotSize.INCH_15,
					  np.array([-48.0, -24.0], dtype=np.float32)),
			]
		super().__init__(robots, communication_mode=communication_mode,
						 deterministic=deterministic)

	@property
	def total_time(self) -> float:
		return 60.0

	def get_team_for_agent(self, agent: str) -> str:
		return "red"

	def compute_score(self) -> Dict[str, int]:
		score = 0
		for obj in self.state["objects"]:
			if obj["status"] == 2 and obj["kind"] == "pin":
				score += 5
		score += sum(8 for agent in self.state["agents"].values()
					 if agent.get("parked_zone") == "midfield")
		if self.state.get("autonomous_winner") == "red":
			score += 12
		return {"red": score}


__all__ = ["VexUSkillsGame"]
