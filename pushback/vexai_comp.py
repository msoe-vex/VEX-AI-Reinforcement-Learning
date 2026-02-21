"""
VEX AI Competition - Push Back Variant

Rules per VAISC:
- 2v2 with AI robots
- 15s isolation + 105s interaction (VAIT1)
- Control bonuses, only 24" can park (VAISC2)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pushback import PushBackGame, BlockStatus, LOADERS, NUM_BLOCKS_FIELD, GoalType, GOALS
from vex_core.base_game import Robot, RobotSize, Team

class VexAICompGame(PushBackGame):
    """VEX AI Competition game variant."""
    
    def __init__(self, robots: list = None, enable_communication: bool = False, deterministic: bool = True):
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
        super().__init__(robots, enable_communication=enable_communication, deterministic=deterministic)
    
    @property
    def total_time(self) -> float:
        return 120.0
    
    def compute_score(self) -> Dict[str, int]:
        """
        Compute score for VEX AI Competition (Red vs Blue).
        - Block: 3 pts
        - Park: 10 pts (24" only)
        - Control Bonus: 10 pts (Outermost Long Goal block matches Nearest Center Goal block)
        """
        scores = {"red": 0, "blue": 0}
        
        # Scoring values
        BLOCK_POINTS = 3
        CONTROL_BONUS = 10
        PARK_24 = 10
        
        # 1. Block Points
        blocks_by_status = {} # Status -> List of blocks
        
        for block in self.state["blocks"]:
            status = block["status"]
            if status not in blocks_by_status:
                blocks_by_status[status] = []
            blocks_by_status[status].append(block)
            
            goal_type = BlockStatus.get_goal_type(status)
            if goal_type:
                team = block.get("team", "red")
                if team in scores:
                    scores[team] += BLOCK_POINTS
        
        # 2. Control Bonuses
        # Define pairings: (Long Goal Type, Side Filter Function, Center Goal Type, Target Point)
        # Tape lines at X = +/- 6.0. Blocks inside -6..6 excluded.
        
        def get_outermost_long(goal_type, side_sign):
            # side_sign: -1 for Left (Min X), 1 for Right (Max X)
            target_blocks = blocks_by_status.get(BlockStatus.get_status_for_goal(goal_type), [])
            candidates = []
            for b in target_blocks:
                x = b["position"][0]
                # Check tape line exclusion (|x| > 6.0) AND side
                if abs(x) > 6.0:
                    if side_sign < 0 and x < 0:
                        candidates.append((x, b))
                    elif side_sign > 0 and x > 0:
                        candidates.append((x, b))
            
            if not candidates:
                return None
            
            # Sort by X. If Left (-1), we want MIN X. If Right (1), we want MAX X.
            candidates.sort(key=lambda item: item[0])
            if side_sign < 0:
                return candidates[0][1] # Min X
            else:
                return candidates[-1][1] # Max X

        def get_nearest_center(goal_type, target_point):
            target_blocks = blocks_by_status.get(BlockStatus.get_status_for_goal(goal_type), [])
            if not target_blocks:
                return None
            
            best_block = None
            min_dist = float('inf')
            
            for b in target_blocks:
                dist = np.linalg.norm(b["position"] - target_point)
                if dist < min_dist:
                    min_dist = dist
                    best_block = b
            
            return best_block

        # Pairings
        pairings = [
            # TL: Long 1 Left <-> Center Upper Left End
            (GoalType.LONG_1, -1, GoalType.CENTER_UPPER, GOALS[GoalType.CENTER_UPPER].left_entry),
            # TR: Long 1 Right <-> Center Lower Right/Top Entry (Entry points are swapped in definition relative to name? Let's use entries directly)
            # Center Lower entries: Left(8.5,8.5), Right(-8.5,-8.5) (Wait, definition Step 358 lines 168-169 check)
            # Center Lower: left_entry=(8.5,8.5), right_entry=(-8.5,-8.5).
            # (8.5, 8.5) is Top-Right quadrant. So "Left Entry" is TR? Naming might be relative.
            # I will use the coordinates.
            (GoalType.LONG_1, 1, GoalType.CENTER_LOWER, np.array([8.5, 8.5])),
            # BL: Long 2 Left <-> Center Lower Bottom-Left (Coordinate -8.5, -8.5)
            (GoalType.LONG_2, -1, GoalType.CENTER_LOWER, np.array([-8.5, -8.5])),
            # BR: Long 2 Right <-> Center Upper Bottom-Right (Coordinate 8.5, -8.5. Wait. Center Upper Right Entry is (8.5, -8.5))
            (GoalType.LONG_2, 1, GoalType.CENTER_UPPER, GOALS[GoalType.CENTER_UPPER].right_entry)
        ]
        
        for lg_type, side, cg_type, cg_point in pairings:
            b_long = get_outermost_long(lg_type, side)
            b_center = get_nearest_center(cg_type, cg_point)
            
            if b_long and b_center:
                team_l = b_long.get("team")
                team_c = b_center.get("team")
                if team_l == team_c and team_l in scores:
                    scores[team_l] += CONTROL_BONUS

        # 3. Parked Robots (24" only)
        for agent_name, agent_state in self.state["agents"].items():
            if agent_state.get("parked", False):
                if agent_state.get("robot_size") == 24:  # RobotSize.INCH_24.value
                    team = agent_state["team"]
                    if team in scores:
                        scores[team] += PARK_24
        
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

        # --- VEX AI COMP BLOCKS ---
        
        # Blue Blocks
        blue_coords = [
            (46, 48), (50, 48), (46, -48), (50, -48),
            (-46, 48), (-50, 48), (-46, 48), (-50, 48),
            (-46, -48), (-50, -48), (-46, -48), (-50, -48),
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
            (46, 48), (50, 48), (46, 48), (50, 48),
            (46, -48), (50, -48), (46, -48), (50, -48),
            (-46, 48), (-50, 48), (-46, -48), (-50, -48),
            (-24, 24), (-24, 24), (-24, 24),
            (-24, 28), (-28, 24),
            (-24, -24), (-24, -24), (-24, -24),
            (-24, -28), (-28, -24),
            (24, 28), (24, 28), (28, 24), (28, 24),
            (24, -28), (24, -28), (28, -24), (28, -24)
        ]
        for x, y in red_coords:
            add_block(x, y, "red")
            
        # --- LOADERS (AI Comp) ---
        # TL: R, B, R, B, R, B
        # TR: B, R, B, R, B, R
        # BL: R, B, R, B, R, B
        # BR: B, R, B, R, B, R
        
        loaders_config = [
            (0, ["red", "blue"] * 3),  # TL
            (1, ["blue", "red"] * 3),  # TR
            (2, ["red", "blue"] * 3),  # BL
            (3, ["blue", "red"] * 3),  # BR
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
