"""
VEX AI Competition - Push Back Variant

Rules per VAISC:
- 2v2 with AI robots
- 15s isolation + 105s interaction (VAIT1)
- Control bonuses, only 24" can park (VAISC2)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .pushback import PushBackGame, ScoringConfig, BlockStatus, LOADERS, NUM_BLOCKS_FIELD


class VexAICompGame(PushBackGame):
    """VEX AI Competition game variant."""
    
    def _get_scoring_config(self) -> ScoringConfig:
        return ScoringConfig(
            total_time=120.0,  # 15s isolation + 105s interaction
            block_points=3,
            control_zone_long=10,  # Control Bonus (VAISC1)
            control_zone_center_upper=10,
            control_zone_center_lower=10,
            park_single=10,  # Only 24" can park (VAISC2)
            park_double=10,  # Only one robot can park
            cleared_loader=0,
            cleared_park_zone=0,
            is_skills=False,
        )
    
    def _get_agents_config(self) -> List[str]:
        # 2v2: red vs blue
        return ["red_robot_0", "red_robot_1", "blue_robot_0", "blue_robot_1"]
    
    def _get_robot_configs(self) -> Dict[str, Tuple[np.ndarray, str, str]]:
        """
        Red alliance on left, blue on right.
        Robot 0 = 24" bot (can park), Robot 1 = 15" bot (cannot park).
        """
        return {
            "red_robot_0": (np.array([-60.0, 24.0], dtype=np.float32), "24", "red"),
            "red_robot_1": (np.array([-60.0, -24.0], dtype=np.float32), "15", "red"),
            "blue_robot_0": (np.array([60.0, 24.0], dtype=np.float32), "24", "blue"),
            "blue_robot_1": (np.array([60.0, -24.0], dtype=np.float32), "15", "blue"),
        }
    
    def _get_initial_blocks(
        self, randomize: bool, seed: Optional[int]
    ) -> List[Dict]:
        """Generate initial blocks with preloads."""
        rng = np.random.default_rng(seed)
        blocks = []
        
        # Field blocks - simple grid pattern
        block_count = 0
        grid_positions = self._generate_grid_positions()
        
        for x, y in grid_positions[:NUM_BLOCKS_FIELD]:
            if randomize:
                x += rng.uniform(-6.0, 6.0)
                y += rng.uniform(-6.0, 6.0)
            
            team = "red" if block_count % 2 == 0 else "blue"
            blocks.append({
                "position": np.array([x, y], dtype=np.float32),
                "status": BlockStatus.ON_FIELD,
                "team": team,
                "held_by": None,
            })
            block_count += 1
        
        # Loader blocks (6 per loader)
        for loader in LOADERS:
            team = "red" if loader.index % 2 == 0 else "blue"
            for _ in range(6):
                blocks.append({
                    "position": loader.position.copy().astype(np.float32),
                    "status": BlockStatus.IN_LOADER_TL + loader.index,
                    "team": team,
                    "held_by": None,
                })
        
        # Preloads - one per robot, matching team color
        for agent in self._get_agents_config():
            team = "red" if "red" in agent else "blue"
            pos_x = -60.0 if team == "red" else 60.0
            blocks.append({
                "position": np.array([pos_x, 0.0], dtype=np.float32),
                "status": BlockStatus.HELD,
                "team": team,
                "held_by": agent,
            })
        
        return blocks
    
    def _get_loader_counts(self) -> List[int]:
        return [6, 6, 6, 6]
    
    def _generate_grid_positions(self) -> List[Tuple[float, float]]:
        """Generate a grid of block positions."""
        positions = []
        
        # Quadrants around center
        for dx in [-24.0, 24.0]:
            for dy in [-24.0, 24.0]:
                for ox in range(3):
                    for oy in range(3):
                        x = dx + ox * 6.0
                        y = dy + oy * 6.0
                        if abs(x) > 12 or abs(y) > 12:
                            positions.append((x, y))
        
        # Near long goals
        for y in [36.0, -36.0]:
            for x in [-36.0, -24.0, -12.0, 12.0, 24.0, 36.0]:
                positions.append((x, y))
        
        # Near park zones
        for x in [-48.0, 48.0]:
            for y in [-24.0, -12.0, 0.0, 12.0, 24.0]:
                positions.append((x, y))
        
        return positions
    
    # TODO: Implement AI Competition-specific rules
    # - Only 24" robots can park (VAISC2)
    # - Control bonus logic (VAISC1)
