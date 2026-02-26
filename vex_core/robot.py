from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

class RobotSize(Enum):
    """Robot size categories."""
    INCH_15 = 15
    INCH_24 = 24

class Team(Enum):
    """Robot teams."""
    RED = "red"
    BLUE = "blue"

@dataclass
class Robot:
    """Robot configuration."""
    name: str  # Agent name, e.g., "red_robot_0"
    team: Team  # 'red' or 'blue'
    size: RobotSize
    start_position: Optional[np.ndarray] = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float32))
    start_orientation: Optional[float] = None  # Radians, None = auto based on team
    length: Optional[float] = None
    width: Optional[float] = None
    max_speed: Optional[float] = 85.0
    max_acceleration: Optional[float] = 85.0
    buffer: Optional[float] = 1.0
    # Camera rotation is interpreted as an offset (radians) relative to the robot body orientation.
    # Positive values rotate the camera counter-clockwise relative to the robot heading.
    camera_rotation: Optional[float] = np.pi / 2  
    
    def __post_init__(self):
        # Default dimensions based on size
        if self.length is None:
            self.length = float(self.size.value)
        if self.width is None:
            self.width = float(self.size.value)
        # Default orientation: face toward center (red=0, blue=π)
        if self.start_orientation is None:
            self.start_orientation = np.float32(0.0) if self.team == Team.RED else np.float32(np.pi)
        # Camera rotation is stored as an offset from the robot body orientation.
        try:
            self.camera_rotation_offset = float(self.camera_rotation)
        except Exception:
            self.camera_rotation_offset = 0.0
