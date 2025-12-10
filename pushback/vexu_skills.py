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

from .pushback import PushBackGame, ScoringConfig, BlockStatus, LOADERS, NUM_BLOCKS_FIELD


class VexUSkillsGame(PushBackGame):
    """VEX U Skills game variant."""
    
    def _get_scoring_config(self) -> ScoringConfig:
        return ScoringConfig(
            total_time=60.0,
            block_points=1,
            control_zone_long=5,
            control_zone_center_upper=10,
            control_zone_center_lower=10,
            park_single=15,
            park_double=15,  # Same as single in skills
            cleared_loader=5,
            cleared_park_zone=5,
            is_skills=True,
        )
    
    def _get_agents_config(self) -> List[str]:
        # Both robots are red, starting same side
        return ["red_robot_0", "red_robot_1"]
    
    def _get_robot_configs(self) -> Dict[str, Tuple[np.ndarray, str, str]]:
        """
        Both robots start on red (left) side.
        Robot 0 = 24" bot, Robot 1 = 15" bot
        """
        return {
            "red_robot_0": (np.array([-60.0, 24.0], dtype=np.float32), "24", "red"),
            "red_robot_1": (np.array([-60.0, -24.0], dtype=np.float32), "15", "red"),
        }
    
    def _get_initial_blocks(
        self, randomize: bool, seed: Optional[int]
    ) -> List[Dict]:
        """Generate initial blocks with preloads for each robot."""
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
            blocks.append({
                "position": np.array([-60.0, 0.0], dtype=np.float32),
                "status": BlockStatus.HELD,
                "team": "red",  # Skills mode - red team
                "held_by": agent,
            })
        
        return blocks
    
    def _get_loader_counts(self) -> List[int]:
        return [6, 6, 6, 6]
    
    def _generate_grid_positions(self) -> List[Tuple[float, float]]:
        """Generate a grid of block positions avoiding obstacles."""
        positions = []
        
        # Quadrants around center
        for dx in [-24.0, 24.0]:
            for dy in [-24.0, 24.0]:
                for ox in range(3):
                    for oy in range(3):
                        x = dx + ox * 6.0
                        y = dy + oy * 6.0
                        # Avoid center goal area
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
