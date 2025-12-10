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

from .pushback import PushBackGame, ScoringConfig, BlockStatus, LOADERS, NUM_BLOCKS_FIELD


class VexAISkillsGame(PushBackGame):
    """VEX AI Skills game variant."""
    
    def _get_scoring_config(self) -> ScoringConfig:
        return ScoringConfig(
            total_time=60.0,
            block_points=3,  # Majority color blocks
            control_zone_long=0,  # Not used in AI Skills (VAIRS7e)
            control_zone_center_upper=0,
            control_zone_center_lower=0,
            park_single=5,
            park_double=5,
            cleared_loader=0,  # Different system - loader matching
            cleared_park_zone=0,
            is_skills=True,
        )
    
    def _get_agents_config(self) -> List[str]:
        # One robot from each team, starting opposite sides
        return ["red_robot_0", "blue_robot_0"]
    
    def _get_robot_configs(self) -> Dict[str, Tuple[np.ndarray, str, str]]:
        """
        Per VAIRS4:
        - 24" robot starts in BLUE Park Zone (parked position)
        - 15" robot starts in RED Park Zone (parked position)
        
        We designate blue_robot_0 as 24" and red_robot_0 as 15".
        """
        return {
            # 24" robot in blue park zone
            "blue_robot_0": (np.array([60.0, 0.0], dtype=np.float32), "24", "blue"),
            # 15" robot in red park zone  
            "red_robot_0": (np.array([-60.0, 0.0], dtype=np.float32), "15", "red"),
        }
    
    def _get_initial_blocks(
        self, randomize: bool, seed: Optional[int]
    ) -> List[Dict]:
        """Generate initial blocks - NO PRELOADS per VAIRS4."""
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
        
        # NO PRELOADS in VEX AI Skills (VAIRS4)
        
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
    
    # TODO: Implement AI Skills-specific scoring (VAIRS7)
    # - Majority color scoring per goal
    # - Loader matching points (blocks matching adjacent park zone color)
    # - Filled loader bonus
