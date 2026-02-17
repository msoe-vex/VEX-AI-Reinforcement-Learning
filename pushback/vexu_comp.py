"""
VEX U Competition - Push Back Variant

Rules:
- 2v2 alliance competition
- 30s auto + 90s driver (VUT4, VUT5)
- 3pts per block, control zone bonuses, parking bonuses
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pushback import PushBackGame, BlockStatus, LOADERS, NUM_BLOCKS_FIELD, GoalType, GOALS
from vex_core.base_game import Robot, RobotSize, Team

class VexUCompGame(PushBackGame):
    """VEX U Competition game variant."""
    
    def __init__(self, robots: list = None, enable_communication: bool = False):
        # Default: 2v2 (2 per team, 24" and 15")
        if robots is None:
            robots = [
                Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_24, 
                      start_position=np.array([-42.0, 24.0], dtype=np.float32)),
                Robot(name="red_robot_1", team=Team.RED, size=RobotSize.INCH_15, 
                      start_position=np.array([-46.5, -24.0], dtype=np.float32)),
                Robot(name="blue_robot_0", team=Team.BLUE, size=RobotSize.INCH_24, 
                      start_position=np.array([42.0, 24.0], dtype=np.float32)),
                Robot(name="blue_robot_1", team=Team.BLUE, size=RobotSize.INCH_15, 
                      start_position=np.array([46.5, -24.0], dtype=np.float32)),
            ]
        super().__init__(robots, enable_communication=enable_communication)
    
    @property
    def total_time(self) -> float:
        return 120.0
    
    def compute_score(self) -> Dict[str, int]:
        """
        Compute score for VEX U Competition (Red vs Blue).
        Returns:
            Dict[str, int]: {"red": score, "blue": score}
        """
        scores = {"red": 0, "blue": 0}
        
        # Scoring values
        BLOCK_POINTS = 3
        CONTROL_ZONE_LONG = 10
        CONTROL_ZONE_CENTER_UPPER = 8
        CONTROL_ZONE_CENTER_LOWER = 6
        PARK_SINGLE = 8
        PARK_DOUBLE = 30
        
        # 1. Block Points & Counts for Majority
        # counts[goal_type][team] = count
        goal_counts = {
            gt: {"red": 0, "blue": 0} 
            for gt in [GoalType.LONG_1, GoalType.LONG_2, GoalType.CENTER_UPPER, GoalType.CENTER_LOWER]
        }
        
        for block in self.state["blocks"]:
            goal_type = BlockStatus.get_goal_type(block["status"])
            if goal_type:
                team = block.get("team", "red")
                if team in scores:
                    scores[team] += BLOCK_POINTS
                    goal_counts[goal_type][team] += 1
        
        # 2. Control Zones (Majority Rule)
        # Long Goals
        for goal_type in [GoalType.LONG_1, GoalType.LONG_2]:
            r = goal_counts[goal_type]["red"]
            b = goal_counts[goal_type]["blue"]
            if r > b:
                scores["red"] += CONTROL_ZONE_LONG
            elif b > r:
                scores["blue"] += CONTROL_ZONE_LONG
        
        # Center Upper
        r = goal_counts[GoalType.CENTER_UPPER]["red"]
        b = goal_counts[GoalType.CENTER_UPPER]["blue"]
        if r > b:
            scores["red"] += CONTROL_ZONE_CENTER_UPPER
        elif b > r:
            scores["blue"] += CONTROL_ZONE_CENTER_UPPER
            
        # Center Lower
        r = goal_counts[GoalType.CENTER_LOWER]["red"]
        b = goal_counts[GoalType.CENTER_LOWER]["blue"]
        if r > b:
            scores["red"] += CONTROL_ZONE_CENTER_LOWER
        elif b > r:
            scores["blue"] += CONTROL_ZONE_CENTER_LOWER
        
        # 3. Parked Robots
        parked_counts = {"red": 0, "blue": 0}
        for agent_name, agent_state in self.state["agents"].items():
            if agent_state.get("parked", False):
                team = agent_state["team"]
                if team in parked_counts:
                    parked_counts[team] += 1
        
        for team, count in parked_counts.items():
            if count >= 2:
                scores[team] += PARK_DOUBLE
            elif count == 1:
                scores[team] += PARK_SINGLE
        
        return scores
    

    

    
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

        # --- VEX U COMP BLOCKS ---
        
        # Blue Blocks
        blue_coords = [
            (-50, 70), (-46, 70), (-50, -70), (-46, -70),
            (60, 6), (60, 2), (60, -2), (60, -6),
            (2, 48), (6, 48), (2, -48), (6, -48),
            (0, -24), (0, -32), (0, -40), (0, 28), (0, 36), (0, 44)
        ]
        for x, y in blue_coords:
            add_block(x, y, "blue")
            
        # Red Blocks
        red_coords = [
            (50, 70), (46, 70), (50, -70), (46, -70),
            (-60, 6), (-60, 2), (-60, -2), (-60, -6),
            (-2, 48), (-6, 48), (-2, -48), (-6, -48),
            (0, 24), (0, 32), (0, 40), (0, -28), (0, -36), (0, -44)
        ]
        for x, y in red_coords:
            add_block(x, y, "red")
            
        # --- LOADERS ---
        # TL (0): 3 Blue, 3 Red (Top to Bottom -> First consumed is Bottom)
        # Note: In standard queue, first in is first out.
        # If "Top" is listed first in notes, and we take from top (or bottom?), we strictly just put them in the pool.
        # "Top left loader: Blue, Blue, Blue, Red, Red, Red"
        # Since we just match "color" for AI, and U doesn't care, we just add them.
        
        # --- LOADERS ---
        # User: "Take from loader actually takes the bottom block not the top"
        # User: "colors in loaders are listed top to bottom"
        # To make "take from loader" (which picks first found) take the bottom one, 
        # we must add the Bottom blocks first.
        # So we iterate the config (Top->Bottom) in REVERSE.
        
        loaders_config = [
            (0, ["blue"]*3 + ["red"]*3),  # TL
            (1, ["red"]*3 + ["blue"]*3),  # TR
            (2, ["blue"]*3 + ["red"]*3),  # BL
            (3, ["red"]*3 + ["blue"]*3),  # BR
        ]
        
        for l_idx, colors in loaders_config:
            loader_pos = LOADERS[l_idx].position
            # Add Top-to-Bottom (Standard Order)
            for color in colors:
                add_block(loader_pos[0], loader_pos[1], color, status=BlockStatus.IN_LOADER_TL + l_idx)
        
        # Preloads: 1 per robot
        for robot in self.robots:
            add_block(robot.start_position[0], robot.start_position[1], robot.team, status=BlockStatus.HELD)
            blocks[-1]["held_by"] = robot.name
            
        return blocks
    
    def _get_loader_counts(self) -> List[int]:
        return [6, 6, 6, 6]
