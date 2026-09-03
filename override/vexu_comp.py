"""VEX U Competition variant of Override."""

from typing import Optional

import numpy as np

from vex_core.config import CommunicationOption
from vex_core.robot import Robot, RobotSize, Team

from .override import OverrideGame


class VexUCompGame(OverrideGame):
    """Two-robot-per-alliance VEX U Override match."""

    def __init__(self, robots: Optional[list] = None,
                 communication_mode: CommunicationOption = CommunicationOption.NONE,
                 deterministic: bool = True):
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
        super().__init__(robots, communication_mode=communication_mode,
                         deterministic=deterministic)


__all__ = ["VexUCompGame"]
