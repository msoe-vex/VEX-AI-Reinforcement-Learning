"""
VEX Push Back - Field Layout Configuration

Defines the physical field layout including:
- Field dimensions
- Goal positions and entry points
- Loader positions
- Park zones
- Permanent obstacles
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

# Try import path planner Obstacle, fallback to local definition
try:
    from .path_planner import Obstacle
except ImportError:
    try:
        from path_planner import Obstacle
    except ImportError:
        @dataclass
        class Obstacle:
            x: float
            y: float
            radius: float
            ignore_collision: bool = False


# =============================================================================
# Field Constants (all measurements in inches)
# =============================================================================

FIELD_SIZE_INCHES = 144  # 12 feet = 144 inches
FIELD_HALF = FIELD_SIZE_INCHES / 2  # 72 inches from center to edge

# Robot dimensions
ROBOT_WIDTH = 18.0
ROBOT_LENGTH = 18.0
BUFFER_RADIUS_INCHES = 2.0

# Block dimensions (hexagonal blocks)
BLOCK_RADIUS = 2.4  # Approximate radius for hexagon

# Game element counts
NUM_LOADERS = 4
NUM_BLOCKS_FIELD = 36
NUM_BLOCKS_LOADER = 24  # 6 per loader
TOTAL_BLOCKS = 62  # 36 field + 24 loaders + 2 preloads (1 per robot)

# Goal capacities
LONG_GOAL_CAPACITY = 15
CENTER_GOAL_CAPACITY = 7

# Control zone thresholds
LONG_GOAL_CONTROL_THRESHOLD = 3
CENTER_GOAL_CONTROL_THRESHOLD = 7

# Field of view for robot vision
FOV = np.pi / 2


class GoalType(Enum):
    """Types of goals on the field."""
    LONG_1 = "long_1"           # Long goal at y=48 (top)
    LONG_2 = "long_2"           # Long goal at y=-48 (bottom)
    CENTER_UPPER = "center_upper"  # Upper center: UL to LR diagonal (45°)
    CENTER_LOWER = "center_lower"  # Lower center: LL to UR diagonal (-45°)


@dataclass
class GoalPosition:
    """Defines a goal's position and entry points."""
    center: np.ndarray           # Center position of the goal
    left_entry: np.ndarray       # Left side entry point
    right_entry: np.ndarray      # Right side entry point
    capacity: int                # Maximum blocks the goal can hold
    control_threshold: int       # Blocks needed for control bonus
    goal_type: GoalType
    angle: float = 0.0           # Rotation angle in degrees (for rendering)
    
    def get_nearest_entry(self, robot_position: np.ndarray) -> np.ndarray:
        """Get the entry point nearest to the robot."""
        left_dist = np.linalg.norm(robot_position - self.left_entry)
        right_dist = np.linalg.norm(robot_position - self.right_entry)
        return self.left_entry if left_dist < right_dist else self.right_entry
    
    def get_nearest_entry_side(self, robot_position: np.ndarray) -> str:
        """Get the side of the nearest entry ('left' or 'right')."""
        left_dist = np.linalg.norm(robot_position - self.left_entry)
        right_dist = np.linalg.norm(robot_position - self.right_entry)
        return "left" if left_dist < right_dist else "right"


@dataclass
class ParkZone:
    """Defines a parking zone for a team."""
    center: np.ndarray
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    
    def contains(self, position: np.ndarray) -> bool:
        """Check if a position is within the park zone."""
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max


@dataclass  
class LoaderPosition:
    """Defines a loader's position."""
    position: np.ndarray
    index: int  # 0=TL, 1=TR, 2=BL, 3=BR


# =============================================================================
# Field Layout
# =============================================================================

# Goal positions with entry points (all in inches, centered at origin)
# Center goals are at 45° angles forming an X pattern
GOALS: Dict[GoalType, GoalPosition] = {
    GoalType.LONG_1: GoalPosition(
        center=np.array([0.0, 48.0]),
        left_entry=np.array([-24.0, 48.0]),
        right_entry=np.array([24.0, 48.0]),
        capacity=LONG_GOAL_CAPACITY,
        control_threshold=LONG_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.LONG_1,
        angle=0.0,
    ),
    GoalType.LONG_2: GoalPosition(
        center=np.array([0.0, -48.0]),
        left_entry=np.array([-24.0, -48.0]),
        right_entry=np.array([24.0, -48.0]),
        capacity=LONG_GOAL_CAPACITY,
        control_threshold=LONG_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.LONG_2,
        angle=0.0,
    ),
    # Center Upper: diagonal at -45° rotation (render coordmates)
    # Entry points are at the diagonal ends
    GoalType.CENTER_UPPER: GoalPosition(
        center=np.array([0.0, 0.0]),
        left_entry=np.array([-8.5, 8.5]),   # Top-left end
        right_entry=np.array([8.5, -8.5]),  # Bottom-right end
        capacity=CENTER_GOAL_CAPACITY,
        control_threshold=CENTER_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.CENTER_UPPER,
        angle=-45.0,
    ),
    # Center Lower: diagonal from upper-right to bottom-left (-45° rotation when drawn)
    GoalType.CENTER_LOWER: GoalPosition(
        center=np.array([0.0, 0.0]),
        left_entry=np.array([8.5, 8.5]),    # Upper-right end
        right_entry=np.array([-8.5, -8.5]), # Bottom-left end
        capacity=CENTER_GOAL_CAPACITY,
        control_threshold=CENTER_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.CENTER_LOWER,
        angle=-45.0,
    ),
}

# Loader positions (corners of field)
LOADERS: List[LoaderPosition] = [
    LoaderPosition(position=np.array([-72.0, 48.0]), index=0),   # Top Left
    LoaderPosition(position=np.array([72.0, 48.0]), index=1),    # Top Right
    LoaderPosition(position=np.array([-72.0, -48.0]), index=2),  # Bottom Left
    LoaderPosition(position=np.array([72.0, -48.0]), index=3),   # Bottom Right
]

# Park zones for each team
PARK_ZONES = {
    "red": ParkZone(
        center=np.array([-60.0, 0.0]),
        bounds=(-72.0, -54.0, -12.0, 12.0),  # Left side
    ),
    "blue": ParkZone(
        center=np.array([60.0, 0.0]),
        bounds=(54.0, 72.0, -12.0, 12.0),  # Right side
    ),
}

# Permanent obstacles (goal structures, etc.)
PERMANENT_OBSTACLES: List[Obstacle] = [
    Obstacle(0.0, 0.0, 11.3, False),       # Center Goal Structure
    Obstacle(-21.0, 48.0, 3.0, False),     # Long Goal Top - Left End
    Obstacle(0.0, 48.0, 3.0, False),       # Long Goal Top - Center
    Obstacle(21.0, 48.0, 3.0, False),      # Long Goal Top - Right End
    Obstacle(-21.0, -48.0, 3.0, False),    # Long Goal Bottom - Left End
    Obstacle(0.0, -48.0, 3.0, False),      # Long Goal Bottom - Center
    Obstacle(21.0, -48.0, 3.0, False),     # Long Goal Bottom - Right End
    Obstacle(58.0, -10.0, 0.0, False),     # Blue Park Zone Bottom Corner
    Obstacle(58.0, 10.0, 0.0, False),      # Blue Park Zone Top Corner
    Obstacle(-58.0, -10.0, 0.0, False),    # Red Park Zone Bottom Corner
    Obstacle(-58.0, 10.0, 0.0, False),     # Red Park Zone Top Corner
]


# =============================================================================
# Initial Block Positions
# =============================================================================

def get_initial_field_blocks(randomize: bool = True, seed=None) -> List[Dict]:
    """
    Generate initial block positions on the field.
    
    Args:
        randomize: If True, fully randomize block positions. If False, use fixed pattern.
        seed: Random seed for reproducibility.
        
    Returns:
        List of block dictionaries with position, status, and team.
        Status 0 = on field, 1 = held, 2-5 = in goals, 6-9 = in loaders
        Team: 'red' or 'blue'
    """
    blocks = []
    rng = np.random.default_rng(seed)
    block_count = 0
    
    def add_block(x: float, y: float, status: int = 0, team: str = None):
        nonlocal block_count
        # Alternate team colors if not specified
        if team is None:
            team = "red" if block_count % 2 == 0 else "blue"
        blocks.append({
            "position": np.array([x, y], dtype=np.float32),
            "status": status,
            "team": team
        })
        block_count += 1
    
    def is_valid_position(x: float, y: float) -> bool:
        """Check if position is valid (not in goals, loaders, or park zones)."""
        # Avoid center goal area
        if abs(x) < 20 and abs(y) < 20:
            return False
        # Avoid long goal areas
        if abs(y) > 42 and abs(x) < 28:
            return False
        # Avoid park zones
        if abs(x) > 50 and abs(y) < 15:
            return False
        # Avoid loader areas
        for loader in LOADERS:
            if np.linalg.norm(np.array([x, y]) - loader.position) < 12:
                return False
        return True
    
    if randomize:
        # Fully randomize block positions
        while len(blocks) < NUM_BLOCKS_FIELD:
            x = rng.uniform(-60, 60)
            y = rng.uniform(-60, 60)
            if is_valid_position(x, y):
                add_block(x, y)
    else:
        # Use fixed pattern (original logic)
        for dx in [-18.0, 18.0]:
            for dy in [-18.0, 18.0]:
                add_block(0.0 + dx, 0.0 + dy)
                add_block(0.0 + dx + 6.0, 0.0 + dy)
                add_block(0.0 + dx, 0.0 + dy + 6.0)
                add_block(0.0 + dx + 6.0, 0.0 + dy + 6.0)
        
        for i in range(5):
            add_block(-24.0 + i * 12.0, -48.0)
            add_block(-24.0 + i * 12.0, 48.0)
        
        for i in range(5):
            add_block(-54.0, -24.0 + i * 6.0)
            add_block(54.0, -24.0 + i * 6.0)
        
        while len(blocks) < NUM_BLOCKS_FIELD:
            add_block(-60.0, -60.0)
    
    return blocks


def get_initial_loader_blocks() -> List[Dict]:
    """Generate initial blocks in loaders (6 per loader)."""
    blocks = []
    
    for loader in LOADERS:
        for _ in range(6):
            # Alternate teams for loader blocks
            team = "red" if loader.index % 2 == 0 else "blue"
            blocks.append({
                "position": loader.position.copy().astype(np.float32),
                "status": 6 + loader.index,  # Status 6-9 for loaders 0-3
                "team": team
            })
    
    return blocks


def get_preload_block(team: str) -> Dict:
    """Get a preload block for the starting robot."""
    if team == "red":
        pos = np.array([-60.0, 0.0], dtype=np.float32)
    else:
        pos = np.array([60.0, 0.0], dtype=np.float32)
    
    return {"position": pos, "status": 1, "team": team}  # Status 1 = held


def get_all_initial_blocks(agents: list = None) -> List[Dict]:
    """
    Get all initial blocks for a match.
    
    Args:
        agents: List of agent names to create preloads for. Each agent gets one preload block.
    """
    blocks = get_initial_field_blocks()
    blocks.extend(get_initial_loader_blocks())
    
    # Create preload blocks for each agent
    if agents:
        for agent in agents:
            team = "red" if "red" in agent.lower() else "blue"
            blocks.append(get_preload_block(team))
    else:
        # Default: single red preload
        blocks.append(get_preload_block("red"))
    
    return blocks


# =============================================================================
# Starting Positions
# =============================================================================

def get_robot_start_position(team: str, robot_index: int = 0) -> np.ndarray:
    """
    Get starting position for a robot based on team and index.
    
    Robots start at y=24 (index 0) and y=-24 (index 1).
    """
    if team == "red":
        # Red robots start on left side
        base_x = -60.0
    else:
        # Blue robots start on right side
        base_x = 60.0
    
    # First robot at y=24, second at y=-24
    y_positions = [24.0, -24.0]
    y = y_positions[robot_index] if robot_index < len(y_positions) else 0.0
    
    return np.array([base_x, y], dtype=np.float32)
