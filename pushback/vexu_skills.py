"""
VEX U Skills - Push Back Variant

Rules:
- 2 red robots working together (VURS3)
- Both robots start same side on red (VURS2)
- 60 second match (RSC2)
- 1pt per block, 5pt filled control zones, 15pt parking
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pushback import PushBackGame, BlockStatus, LOADERS, NUM_BLOCKS_FIELD, GoalType, GOALS, PARK_ZONES
from vex_core.base_game import Robot, RobotSize, Team

class VexUSkillsGame(PushBackGame):
    """VEX U Skills game variant."""
    
    def __init__(self, robots: list = None, enable_communication: bool = False, deterministic: bool = True):
        # Default: both robots red, start on red side (per VURS2)
        if robots is None:
            robots = [
                Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_24, 
                      start_position=np.array([-42.0, 24.0], dtype=np.float32)),
                Robot(name="red_robot_1", team=Team.RED, size=RobotSize.INCH_15, 
                      start_position=np.array([-46.5, -24.0], dtype=np.float32)),
            ]
        super().__init__(robots, enable_communication=enable_communication, deterministic=deterministic)
    
    @property
    def default_total_time(self) -> float:
        return 60.0

    def get_team_for_agent(self, agent: str) -> str:
        """Skills is cooperative: all agents contribute to shared red score."""
        return "red"

    def _can_park_in_zone(self, agent_state: Dict, park_zone_color: str) -> bool:
        """VEX U Skills: only red zone parking is valid."""
        return park_zone_color == "red"
    
    def compute_score(self) -> Dict[str, int]:
        """
        Compute score for VEX U Skills.
        Returns:
            Dict[str, int]: {"red": score}
        """
        
        """
        - Filled Control Zone Long: 5 pts
        - Filled Control Zone Center: 10 pts
        - Cleared Park Zone: 5 pts
        - Cleared Loader: 5 pts
        - Parked Robot: 15 pts
        """
        score = 0
        
        # Scoring values
        BLOCK_POINTS = 1
        FILLED_CONTROL_LONG = 5
        FILLED_CONTROL_CENTER = 10
        CLEARED_PARK_ZONE = 5
        CLEARED_LOADER = 5
        PARK_ROBOT = 15
        
        # Blocks in goals
        for goal_type, goal in self.goal_manager.goals.items():
            score += goal.count * BLOCK_POINTS
        
        # Filled Control Zones (count >= threshold)
        counts = self.goal_manager.get_goal_counts()
        for goal_type in [GoalType.LONG_1, GoalType.LONG_2]:
            if counts[goal_type] >= GOALS[goal_type].control_threshold:
                score += FILLED_CONTROL_LONG
        
        for goal_type in [GoalType.CENTER_UPPER, GoalType.CENTER_LOWER]:
            if counts[goal_type] >= GOALS[goal_type].control_threshold:
                score += FILLED_CONTROL_CENTER
        
        # Cleared loaders
        cleared_loaders = sum(1 for count in self.state["loaders"] if count == 0)
        score += cleared_loaders * CLEARED_LOADER
        
        # Cleared Park Zones
        # Check if any ON_FIELD block is within any park zone
        # Park Zones defined in PARK_ZONES (red/blue). 
        # "Cleared" means NO blocks contacting floor in zone.
        # We check all blocks with status ON_FIELD against all park zones.
        for zone in PARK_ZONES.values():
            is_cleared = True
            for block in self.state["blocks"]:
                if block["status"] == BlockStatus.ON_FIELD:
                    pos = block["position"]
                    # Bounds: (x_min, x_max, y_min, y_max)
                    # ParkZone center/bounds logic:
                    # PARK_ZONES defined in pushback.py as ParkZone(center, bounds)
                    # bounds is tuple.
                    x_min, x_max, y_min, y_max = zone.bounds
                    if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                        is_cleared = False
                        break
            if is_cleared:
                score += CLEARED_PARK_ZONE
        
        # Parked robots: only red-zone parking counts, and only one can score.
        parked_red = sum(
            1
            for a in self.state["agents"].values()
            if a.get("parked", False) and a.get("parked_zone") == "red"
        )
        score += (1 if parked_red > 0 else 0) * PARK_ROBOT
        
        return {"red": score}
    
    def _get_initial_blocks(
        self, randomize: bool, seed: Optional[int]
    ) -> List[Dict]:
        """Generate initial blocks explicit configuration."""
        if seed is not None:
            np.random.seed(seed)
        
        blocks = []
        
        def add_block(x, y, team, status=BlockStatus.ON_FIELD):
            if randomize and status == BlockStatus.ON_FIELD:
                pos = np.random.uniform(-70.0, 70.0, size=2).astype(np.float32)
            else:
                pos = np.array([float(x), float(y)], dtype=np.float32)
            
            blocks.append({
                "position": pos,
                "status": status,
                "team": team,
                "held_by": None,
            })

        # --- VEX U SKILLS BLOCKS ---
        
        # Blue Blocks
        blue_coords = [
            (-70, 4), (-70, 0), (-70, -4),
            (-66, 4), (-66, 0), (-66, -4),
            (22, 26), (26, 22), (22, -22), (26, -26),
            (-22, 22), (-26, 26), (-22, -26), (-26, -22),
            (50, -2), (46, 2), (-50, 2), (-46, -2)
        ]
        for x, y in blue_coords:
            add_block(x, y, "blue")
            
        # Red Blocks
        red_coords = [
            (70, 4), (70, 0), (70, -4),
            (66, 4), (66, 0), (66, -4),
            (22, 22), (26, 26), (22, -26), (26, -22),
            (-22, 26), (-26, 22), (-22, -22), (-26, -26),
            (50, 2), (46, -2), (-50, -2), (-46, 2)
        ]
        for x, y in red_coords:
            add_block(x, y, "red")
            
        # --- LOADERS (Skills) ---
        # TL: Red, Red, Red, Blue, Blue, Blue
        # TR: Blue, Blue, Blue, Red, Red, Red
        # BL: Blue, Blue, Blue, Red, Red, Red
        # BR: Red, Red, Red, Blue, Blue, Blue
        
        loaders_config = [
            (0, ["red"]*3 + ["blue"]*3),  # TL
            (1, ["blue"]*3 + ["red"]*3),  # TR
            (2, ["blue"]*3 + ["red"]*3),  # BL
            (3, ["red"]*3 + ["blue"]*3),  # BR
        ]
        
        for l_idx, colors in loaders_config:
            loader_pos = LOADERS[l_idx].position
            # Add Top-to-Bottom (Standard Order)
            for color in colors:
                add_block(loader_pos[0], loader_pos[1], color, status=BlockStatus.IN_LOADER_TL + l_idx)
        
        # Preloads: 1 per robot (Both Red in skills)
        for robot in self.robots:
            add_block(robot.start_position[0], robot.start_position[1], "red", status=BlockStatus.HELD)
            blocks[-1]["held_by"] = robot.name
            
        return blocks
    
    def _get_loader_counts(self) -> List[int]:
        return [6, 6, 6, 6]
