"""
VEX AI Skills - Push Back Variant

Rules per VAIRS:
- 24" robot starts in blue Park Zone, 15" in red Park Zone (VAIRS4)
- No preloads (VAIRS4)
- 60 second match
- Majority color scoring, loader matching points (VAIRS7)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pushback import PushBackGame, BlockStatus, LOADERS, NUM_BLOCKS_FIELD, GoalType, GOALS
from vex_core.base_game import Robot, RobotSize, Team

class VexAISkillsGame(PushBackGame):
    """VEX AI Skills game variant."""
    
    def __init__(self, robots: list = None, enable_communication: bool = False, deterministic: bool = True):
        # Default: 24" in blue park zone, 15" in red park zone (per VAIRS4)
        if robots is None:
            robots = [
                Robot(name="blue_robot_0", team=Team.BLUE, size=RobotSize.INCH_24, length=15, width=15,
                      start_position=np.array([60.0, 0.0], dtype=np.float32)),
                Robot(name="red_robot_0", team=Team.RED, size=RobotSize.INCH_15, length=15, width=15,
                      start_position=np.array([-60.0, 0.0], dtype=np.float32)),
            ]
        super().__init__(robots, enable_communication=enable_communication, deterministic=deterministic)
    
    @property
    def total_time(self) -> float:
        return 60.0

    def get_team_for_agent(self, agent: str) -> str:
        """Skills is cooperative: all agents contribute to shared red score."""
        return "red"
    
    def compute_score(self) -> Dict[str, int]:
        """
        Compute score for VEX AI Skills.
        Returns:
            Dict[str, int]: {"red": score}
        """
        
        """
        - Loader Matching: 
            - 3 pts per matching block (Red in Left, Blue in Right).
            - 5 pts bonus if loader full (6) of matching color.
        - Parked Robot: 5 pts.
        """
        score = 0
        
        # Scoring values
        BLOCK_POINTS = 3
        LOADER_BLOCK_POINTS = 3
        LOADER_FULL_BONUS = 5
        PARK_ROBOT = 5
        
        # 1. Goal Majority
        goal_counts = {
            gt: {"red": 0, "blue": 0} 
            for gt in [GoalType.LONG_1, GoalType.LONG_2, GoalType.CENTER_UPPER, GoalType.CENTER_LOWER]
        }
        
        for block in self.state["blocks"]:
            goal_type = BlockStatus.get_goal_type(block["status"])
            if goal_type:
                team = block.get("team", "red")
                goal_counts[goal_type][team] += 1
        
        for gt in goal_counts:
            r = goal_counts[gt]["red"]
            b = goal_counts[gt]["blue"]
            if r > b:
                score += r * BLOCK_POINTS
            elif b > r:
                score += b * BLOCK_POINTS
            # Tie = 0 points
            
        # 2. Loader Matching
        # Loaders 0, 2 are Left (Red). Loaders 1, 3 are Right (Blue).
        loader_stats = {0: {"red": 0, "blue": 0}, 
                        1: {"red": 0, "blue": 0}, 
                        2: {"red": 0, "blue": 0}, 
                        3: {"red": 0, "blue": 0}}
        
        for block in self.state["blocks"]:
            if BlockStatus.IN_LOADER_TL <= block["status"] <= BlockStatus.IN_LOADER_BR:
                loader_idx = block["status"] - BlockStatus.IN_LOADER_TL
                team = block.get("team", "red")
                loader_stats[loader_idx][team] += 1
        
        # Red Loaders (0, 2) check Red blocks
        for idx in [0, 2]:
            count = loader_stats[idx]["red"]
            score += count * LOADER_BLOCK_POINTS
            if count == 6:
                score += LOADER_FULL_BONUS
        
        # Blue Loaders (1, 3) check Blue blocks
        for idx in [1, 3]:
            count = loader_stats[idx]["blue"]
            score += count * LOADER_BLOCK_POINTS
            if count == 6:
                score += LOADER_FULL_BONUS
        
        # 3. Parked Robots
        parked_count = sum(1 for a in self.state["agents"].values() if a.get("parked", False))
        score += parked_count * PARK_ROBOT
        
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

        # --- VEX AI SKILLS BLOCKS ---
        
        # Blue Blocks
        blue_coords = [
            (48, 48), (48, 48), (48, -48), (48, -48),
            (-48, 48), (-48, -48),
            (24, 24), (24, 24), (24, 24),
            (24, 28), (28, 24),
            (24, -24), (24, -24), (24, -24),
            (24, -28), (28, -24),
            (-24, 28), (-24, 28), (-28, 24), (-28, 24),
            (-24, -28), (-24, -28), (-28, -24), (-28, -24)
        ]
        for x, y in blue_coords:
            add_block(x, y, "blue")
            
        # Red Blocks
        red_coords = [
            (48, 48), (48, -48), (-48, 48), (-48, 48),
            (-48, -48), (-48, -48),
            (-24, 24), (-24, 24), (-24, 24),
            (-24, 28), (-28, 24),
            (-24, -24), (-24, -24), (-24, -24),
            (-24, -28), (-28, -24),
            (24, 28), (24, 28), (28, 24), (28, 24),
            (24, -28), (24, -28), (28, -24), (28, -24)
        ]
        for x, y in red_coords:
            add_block(x, y, "red")
            
        # --- LOADERS (AI Skills) ---
        # TL: Red, Blue, Blue, Blue, Blue, Blue
        # TR: Blue, Red, Red, Red, Red, Red
        # BL: Red, Blue, Blue, Blue, Blue, Blue
        # BR: Blue, Red, Red, Red, Red, Red
        
        loaders_config = [
            (0, ["red"] + ["blue"]*5),  # TL
            (1, ["blue"] + ["red"]*5),  # TR
            (2, ["red"] + ["blue"]*5),  # BL
            (3, ["blue"] + ["red"]*5),  # BR
        ]
        
        for l_idx, colors in loaders_config:
            loader_pos = LOADERS[l_idx].position
            # Add Top-to-Bottom (Standard Order)
            for color in colors:
                add_block(loader_pos[0], loader_pos[1], color, status=BlockStatus.IN_LOADER_TL + l_idx)
        
        # NO PRELOADS in VEX AI Skills (VAIRS4)
            
        return blocks
    
    def _get_loader_counts(self) -> List[int]:
        return [6, 6, 6, 6]
