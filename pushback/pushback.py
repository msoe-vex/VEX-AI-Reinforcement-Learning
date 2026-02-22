"""
Push Back - VEX 2024-2025 Game Implementation

This module consolidates:
- Actions (robot actions enum)
- Field layout (goals, loaders, park zones, obstacles)
- Goal mechanics (queue-based block storage)
- Base game logic

Specific game variants (VEX U Skills, VEX U Comp, VEX AI Skills, VEX AI Comp)
subclass PushBackGame to define scoring rules and initial layouts.
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vex_core.base_game import Robot, VexGame, ActionEvent, ActionStep
from path_planner import PathPlanner

# Forward declarations for get_game method
def _get_game_class(game_name: str):
    """Helper to get game class by name (avoids circular imports)."""
    from .vexu_skills import VexUSkillsGame
    from .vexu_comp import VexUCompGame
    from .vexai_skills import VexAISkillsGame
    from .vexai_comp import VexAICompGame
    
    GAME_MAP = {
        'vexu_skills': VexUSkillsGame,
        'vexu_comp': VexUCompGame,
        'vexai_skills': VexAISkillsGame,
        'vexai_comp': VexAICompGame,
    }
    
    if game_name not in GAME_MAP:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAME_MAP.keys())}")
    
    return GAME_MAP[game_name]

NUM_BLOCKS_FIELD = 36
NUM_BLOCKS_LOADER = 24


# =============================================================================
# Constants
# =============================================================================

FIELD_SIZE_INCHES = 144  # 12 feet = 144 inches
FIELD_HALF = FIELD_SIZE_INCHES / 2  # 72 inches

# Block dimensions
BLOCK_RADIUS = 2.4

# Game element counts
NUM_LOADERS = 4
NUM_BLOCKS_FIELD = 36
NUM_BLOCKS_LOADER = 24  # 6 per loader

# Goal capacities
LONG_GOAL_CAPACITY = 15
CENTER_GOAL_CAPACITY = 7

# Control zone thresholds
LONG_GOAL_CONTROL_THRESHOLD = 3
CENTER_GOAL_CONTROL_THRESHOLD = 7

# Field of view for robot vision
FOV = np.pi / 2

# Default penalty for invalid actions
DEFAULT_PENALTY = 1.0

DEFAULT_DURATION = 0.5

# Maximum blocks a robot can hold
MAX_HELD_BLOCKS = 10


# =============================================================================
# Actions
# =============================================================================

class Actions(Enum):
    """Available actions for robots in the Push Back game."""
    PICK_UP_BLOCK = 0             # Pick up nearest block (assumes success, doesn't know color)
    SCORE_IN_LONG_GOAL_1 = 1      # Long goal 1 (y=48)
    SCORE_IN_LONG_GOAL_2 = 2      # Long goal 2 (y=-48)
    SCORE_IN_CENTER_UPPER = 3     # Center goal upper
    SCORE_IN_CENTER_LOWER = 4     # Center goal lower
    TAKE_FROM_LOADER_TL = 5       # Clear Top Left loader (once only, gets all 6 blocks)
    TAKE_FROM_LOADER_TR = 6       # Clear Top Right loader (once only, gets all 6 blocks)
    TAKE_FROM_LOADER_BL = 7       # Clear Bottom Left loader (once only, gets all 6 blocks)
    TAKE_FROM_LOADER_BR = 8       # Clear Bottom Right loader (once only, gets all 6 blocks)
    PARK = 9                      # Park in team's zone
    TURN_TOWARD_CENTER = 10       # Turn to face center of field
    IDLE = 11


def is_scoring_action(action: Actions) -> bool:
    """Check if an action is a scoring action."""
    return action in [
        Actions.SCORE_IN_LONG_GOAL_1,
        Actions.SCORE_IN_LONG_GOAL_2,
        Actions.SCORE_IN_CENTER_UPPER,
        Actions.SCORE_IN_CENTER_LOWER,
    ]


# =============================================================================
# Goal Types and Positions
# =============================================================================

class GoalType(Enum):
    """Types of goals on the field."""
    LONG_1 = "long_1"           # Long goal at y=48 (top)
    LONG_2 = "long_2"           # Long goal at y=-48 (bottom)
    CENTER_UPPER = "center_upper"  # Upper center diagonal
    CENTER_LOWER = "center_lower"  # Lower center diagonal


@dataclass
class GoalPosition:
    """Defines a goal's position and entry points."""
    center: np.ndarray
    left_entry: np.ndarray
    right_entry: np.ndarray
    capacity: int
    control_threshold: int
    goal_type: GoalType
    angle: float = 0.0
    
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


# Goal definitions
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
    GoalType.CENTER_UPPER: GoalPosition(
        center=np.array([0.0, 0.0]),
        left_entry=np.array([-8.5, 8.5]),
        right_entry=np.array([8.5, -8.5]),
        capacity=CENTER_GOAL_CAPACITY,
        control_threshold=CENTER_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.CENTER_UPPER,
        angle=-45.0,
    ),
    GoalType.CENTER_LOWER: GoalPosition(
        center=np.array([0.0, 0.0]),
        left_entry=np.array([8.5, 8.5]),
        right_entry=np.array([-8.5, -8.5]),
        capacity=CENTER_GOAL_CAPACITY,
        control_threshold=CENTER_GOAL_CONTROL_THRESHOLD,
        goal_type=GoalType.CENTER_LOWER,
        angle=-45.0,
    ),
}


# =============================================================================
# Other Field Elements
# =============================================================================

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


LOADERS: List[LoaderPosition] = [
    LoaderPosition(position=np.array([-72.0, 48.0]), index=0),   # Top Left
    LoaderPosition(position=np.array([72.0, 48.0]), index=1),    # Top Right
    LoaderPosition(position=np.array([-72.0, -48.0]), index=2),  # Bottom Left
    LoaderPosition(position=np.array([72.0, -48.0]), index=3),   # Bottom Right
]

PARK_ZONES = {
    "red": ParkZone(
        center=np.array([-60.0, 0.0]),
        bounds=(-72.0, -54.0, -12.0, 12.0),
    ),
    "blue": ParkZone(
        center=np.array([60.0, 0.0]),
        bounds=(54.0, 72.0, -12.0, 12.0),
    ),
}


@dataclass
class Obstacle:
    """Obstacle for path planning."""
    x: float
    y: float
    radius: float
    ignore_collision: bool = False


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
# Block Status
# =============================================================================

class BlockStatus:
    """Block status codes for tracking location."""
    ON_FIELD = 0
    HELD = 1
    IN_LONG_1 = 2
    IN_LONG_2 = 3
    IN_CENTER_UPPER = 4
    IN_CENTER_LOWER = 5
    IN_LOADER_TL = 6
    IN_LOADER_TR = 7
    IN_LOADER_BL = 8
    IN_LOADER_BR = 9
    
    @staticmethod
    def get_goal_type(status: int) -> Optional[GoalType]:
        """Convert block status to goal type if applicable."""
        mapping = {
            2: GoalType.LONG_1,
            3: GoalType.LONG_2,
            4: GoalType.CENTER_UPPER,
            5: GoalType.CENTER_LOWER,
        }
        return mapping.get(status)
    
    @staticmethod
    def get_status_for_goal(goal_type: GoalType) -> int:
        """Get the status code for a goal type."""
        mapping = {
            GoalType.LONG_1: 2,
            GoalType.LONG_2: 3,
            GoalType.CENTER_UPPER: 4,
            GoalType.CENTER_LOWER: 5,
        }
        return mapping[goal_type]


# =============================================================================
# Observation Index Constants
# =============================================================================

class ObsIndex:
    """Observation vector indices for Push Back game.
    
    Layout (90 total):
    - 0-2: Self position (x, y) and orientation
    - 3-5: Teammate position and orientation
    - 6-7: Held blocks (friendly, opponent)
    - 8: Parked status
    - 9: Time remaining
    - 10-11: Block counts on field (friendly, opponent)
    - 12-41: Friendly block positions (15 * 2)
    - 42-71: Opponent block positions (15 * 2)
    - 72-75: Goals added by this agent (4 goals)
    - 76-79: Loaders cleared by this agent (4 loaders)
    - 80-87: Received messages (8 dims)
    - 88: Is executing an action (1.0 = busy, 0.0 = idle)
    - 89: Current action ID (normalized to [0, 1])
    """
    SELF_POS_X = 0
    SELF_POS_Y = 1
    SELF_ORIENT = 2
    TEAMMATE_START = 3
    HELD_FRIENDLY = 6
    HELD_OPPONENT = 7
    PARKED = 8
    TIME_REMAINING = 9
    FRIENDLY_BLOCK_COUNT = 10
    OPPONENT_BLOCK_COUNT = 11
    FRIENDLY_BLOCKS_START = 12
    OPPONENT_BLOCKS_START = 42
    GOALS_ADDED_START = 72
    LOADERS_CLEARED_START = 76
    RECEIVED_MSG_START = 80
    IS_EXECUTING = 88
    CURRENT_ACTION_ID = 89
    TOTAL = 90


# =============================================================================
# Goal Queue
# =============================================================================

class GoalQueue:
    """Queue-based goal that manages blocks with FIFO behavior."""
    
    def __init__(self, goal_type: GoalType, capacity: int):
        self.goal_type = goal_type
        self.capacity = capacity
        self.slots: List[Optional[int]] = [None] * capacity
        self.goal_position = GOALS[goal_type]
    
    @property
    def count(self) -> int:
        """Number of blocks currently in the goal."""
        return sum(1 for slot in self.slots if slot is not None)
    
    @property
    def is_full(self) -> bool:
        """Check if the goal is at capacity."""
        return all(slot is not None for slot in self.slots)
    
    @property
    def blocks(self) -> List[int]:
        """Get list of block IDs (non-None slots)."""
        return [slot for slot in self.slots if slot is not None]
    
    @property
    def left_entry(self) -> np.ndarray:
        return self.goal_position.left_entry
    
    @property
    def right_entry(self) -> np.ndarray:
        return self.goal_position.right_entry
    
    def get_nearest_entry(self, robot_position: np.ndarray) -> np.ndarray:
        return self.goal_position.get_nearest_entry(robot_position)
    
    def get_nearest_side(self, robot_position: np.ndarray) -> str:
        return self.goal_position.get_nearest_entry_side(robot_position)
    
    def add_block(self, block_id: int, from_side: str) -> Optional[int]:
        """
        Add a block from the specified side. Returns ejected block ID if overflow.
        
        Cascading push mechanics:
        - Insert new block at entry side
        - If that slot was occupied, push that block to the next slot
        - Continue pushing until an empty slot absorbs the chain
        - If chain reaches the far end, that block is ejected
        
        Example: Goal is BB__________RRR (15 slots, B at 0,1; R at 12,13,14)
        Adding blue block from right (slot 14):
        - Slot 14 has R, so push it to slot 13
        - Slot 13 has R, so push it to slot 12  
        - Slot 12 has R, so push it to slot 11
        - Slot 11 is empty, R settles there
        - Place new B at slot 14
        - Result: BB________RRRB (B at 0,1; R at 11,12,13; B at 14)
        """
        ejected = None
        
        if from_side == "left":
            # Adding from left (index 0), push toward right (higher indices)
            entry_idx = 0
            push_direction = 1  # toward higher indices
            exit_idx = self.capacity - 1
        else:  # from_side == "right"
            # Adding from right (last index), push toward left (lower indices)
            entry_idx = self.capacity - 1
            push_direction = -1  # toward lower indices
            exit_idx = 0
        
        # Check if entry slot is occupied
        if self.slots[entry_idx] is not None:
            # Need to push blocks - find the end of the continuous chain
            # Start from entry, walk in push direction until we find empty or exit
            chain_end = entry_idx
            
            while True:
                next_idx = chain_end + push_direction
                
                # Check if we've reached the exit boundary
                if (push_direction > 0 and next_idx > exit_idx) or \
                   (push_direction < 0 and next_idx < exit_idx):
                    # Chain reaches exit - eject the block at chain_end
                    ejected = self.slots[chain_end]
                    # Shift all blocks from chain_end back to entry
                    current = chain_end
                    while current != entry_idx:
                        prev = current - push_direction
                        self.slots[current] = self.slots[prev]
                        current = prev
                    break
                
                # Check if next slot is empty - chain can expand into it
                if self.slots[next_idx] is None:
                    # Shift all blocks one step in push_direction
                    # Start from next_idx (the empty slot) and work back to entry
                    current = next_idx
                    while current != entry_idx:
                        prev = current - push_direction
                        self.slots[current] = self.slots[prev]
                        current = prev
                    break
                
                # Next slot is occupied, continue the chain
                chain_end = next_idx
        
        # Place the new block at entry
        self.slots[entry_idx] = block_id
        
        return ejected
    
    def add_block_from_nearest(
        self, block_id: int, robot_position: np.ndarray
    ) -> Tuple[Optional[int], str]:
        """Add block from nearest side. Returns (ejected_id, side)."""
        side = self.get_nearest_side(robot_position)
        ejected = self.add_block(block_id, side)
        return ejected, side
    
    def get_block_positions(self, goal_center: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Calculate display positions for blocks in the goal."""
        positions = []
        direction = self.right_entry - self.left_entry
        
        for i, block_id in enumerate(self.slots):
            if block_id is not None:
                t = (i + 0.5) / self.capacity
                pos = self.left_entry + t * direction
                positions.append((block_id, pos.copy()))
        
        return positions
    
    def clear(self) -> List[int]:
        """Clear all blocks. Returns list of ejected IDs."""
        ejected = [b for b in self.slots if b is not None]
        self.slots = [None] * self.capacity
        return ejected


class GoalManager:
    """Manages all goals on the field."""
    
    def __init__(self):
        self.goals = {
            GoalType.LONG_1: GoalQueue(GoalType.LONG_1, LONG_GOAL_CAPACITY),
            GoalType.LONG_2: GoalQueue(GoalType.LONG_2, LONG_GOAL_CAPACITY),
            GoalType.CENTER_UPPER: GoalQueue(GoalType.CENTER_UPPER, CENTER_GOAL_CAPACITY),
            GoalType.CENTER_LOWER: GoalQueue(GoalType.CENTER_LOWER, CENTER_GOAL_CAPACITY),
        }
    
    def get_goal(self, goal_type: GoalType) -> GoalQueue:
        return self.goals[goal_type]
    
    def get_goal_counts(self) -> Dict[GoalType, int]:
        return {goal_type: goal.count for goal_type, goal in self.goals.items()}
    
    def reset(self):
        for goal in self.goals.values():
            goal.clear()

# =============================================================================
# Push Back Game Base Class
# =============================================================================

class PushBackGame(VexGame):
    """
    Base Push Back game implementation.
    
    Subclasses must implement:
    - _get_scoring_config() -> ScoringConfig
    - _get_agents_config() -> List of agent names
    - _get_robot_configs() -> Dict mapping agent to (position, size, team)
    - _get_initial_blocks() -> List of block dicts
    - _get_loader_counts() -> List of initial loader counts
    """
    
    
    def __init__(self, robots: list = None, enable_communication: bool = False, deterministic: bool = True):
        super().__init__(robots, enable_communication=enable_communication)
        self.goal_manager = GoalManager()
        self._agents: Optional[List[str]] = None
        self.deterministic = bool(deterministic)
        
        # Initialize state automatically
        self.get_initial_state()
    
    # =========================================================================
    # Abstract methods for variants
    # =========================================================================
    

    
    @abstractmethod
    def _get_initial_blocks(self, randomize: bool, seed: Optional[int]) -> List[Dict]:
        """Get initial block positions for this variant."""
        pass
    
    @abstractmethod
    def _get_loader_counts(self) -> List[int]:
        """Get initial block counts per loader [TL, TR, BL, BR]."""
        pass
    
    # =========================================================================
    # VexGame Properties
    # =========================================================================
    
    @property
    def field_size_inches(self) -> float:
        return FIELD_SIZE_INCHES
    
    @staticmethod
    def get_game(game_name: str, enable_communication: bool = False, deterministic: bool = True) -> VexGame:
        """
        Factory method to create a game instance from a string identifier.
        
        Args:
            game_name: String identifier (e.g., 'vexu_skills', 'vexai_comp')
            enable_communication: Whether to enable agent communication
            deterministic: Whether environment mechanics are deterministic
            
        Returns:
            VexGame instance
        """
        game_class = _get_game_class(game_name)
        return game_class(enable_communication=enable_communication, deterministic=deterministic)
    
    @property
    @abstractmethod
    def total_time(self) -> float:
        """Total game time in seconds."""
        pass
    
    @property
    def num_actions(self) -> int:
        return len(Actions)
    
    @property
    def fallback_action(self) -> int:
        return Actions.IDLE.value
        
    def get_action_name(self, action: int) -> str:
        """Get the human-readable string name for an action index."""
        try:
            return Actions(action).name
        except Exception:
            return str(action)
    
    # =========================================================================
    # VexGame State Management
    # =========================================================================
    
    def reset(self) -> None:
        """Reset game state."""
        self.goal_manager.reset()
        self._agents = None
    
    def get_initial_state(
        self, 
        randomize: bool = False, 
        seed: Optional[int] = None
    ) -> Dict:
        """Create initial game state and store in self.state."""
        # Initialize agents from Robot objects
        agents_dict = {}
        for robot in self.robots:
            agents_dict[robot.name] = {
                "position": robot.start_position.copy().astype(np.float32),
                "orientation": np.array([robot.start_orientation], dtype=np.float32),
                # camera_rotation_offset is a scalar (radians) relative to robot body orientation
                "camera_rotation_offset": float(getattr(robot, "camera_rotation_offset", 0.0)),
                "team": robot.team.value,
                "robot_size": robot.size.value,
                "held_blocks": 0,
                "parked": False,
                # "gameTime": REMOVED - Managed by Enviroment
                "active": True,
                "agent_name": robot.name,
            "current_action": None,
                "goals_added": [0, 0, 0, 0],  # [LONG_1, LONG_2, CENTER_UPPER, CENTER_LOWER]
                "loaders_taken": [0, 0, 0, 0],  # [TL, TR, BL, BR]
            }
        
        # Get blocks
        blocks = self._get_initial_blocks(randomize, seed)
        
        # Count preloads (held blocks)
        for block in blocks:
            if block.get("held_by"):
                agent_name = block["held_by"]
                if agent_name in agents_dict:
                    agents_dict[agent_name]["held_blocks"] += 1
        
        self.state = {
            "agents": agents_dict,
            "blocks": blocks,
            "loaders": self._get_loader_counts(),
        }
        return self.state
    
    def get_observation(self, agent: str, game_time: float = 0.0) -> np.ndarray:
        """Build observation vector for an agent.
        
        The agent knows the color of blocks it holds (from loaders and field).
        It knows positions of all blocks on field, separated by friendly/opponent.
        """
        agent_state = self.state["agents"][agent]
        
        obs_parts = []
        
        # 1. Self position (2) and orientation (1)
        obs_parts.append(float(agent_state["position"][0]))
        obs_parts.append(float(agent_state["position"][1]))
        obs_parts.append(float(agent_state["orientation"][0]))
        
        # 2. Teammate robots only: position (2) + orientation (1) each, max 1 teammate
        MAX_TEAMMATES = 1  # Support up to 2 robots per team
        my_team = agent_state["team"]
        teammate_data = []
        for other_agent, other_state in self.state["agents"].items():
            if other_agent != agent and other_state["team"] == my_team:
                teammate_data.extend([
                    float(other_state["position"][0]),
                    float(other_state["position"][1]),
                    float(other_state["orientation"][0])
                ])
        # Pad with zeros if no teammate
        while len(teammate_data) < MAX_TEAMMATES * 3:
            teammate_data.extend([0.0, 0.0, 0.0])
        obs_parts.extend(teammate_data[:MAX_TEAMMATES * 3])
        
        # 3. Held blocks by color - count from actual blocks
        agent_name = agent_state["agent_name"]
        held_friendly = 0
        held_opponent = 0
        for block in self.state["blocks"]:
            if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                if block.get("team") == my_team:
                    held_friendly += 1
                else:
                    held_opponent += 1
        obs_parts.append(float(held_friendly))
        obs_parts.append(float(held_opponent))
        
        # 4. Self parked status (1)
        obs_parts.append(1.0 if agent_state["parked"] else 0.0)
        
        # 5. Time remaining (1)
        obs_parts.append(float(self.total_time - game_time))
        
        # 6. Count and Position of blocks on field by color (friendly/opponent)
        MAX_TRACKED = 15
        
        friendly_blocks = []
        opponent_blocks = []
        
        robot_pos = agent_state["position"]
        for block in self.state["blocks"]:
            if block["status"] == BlockStatus.ON_FIELD:
                dist = np.linalg.norm(block["position"] - robot_pos)
                block_info = (dist, block["position"][0], block["position"][1])
                
                if block.get("team") == my_team:
                    friendly_blocks.append(block_info)
                else:
                    opponent_blocks.append(block_info)
        
        # Sort by distance
        friendly_blocks.sort(key=lambda x: x[0])
        opponent_blocks.sort(key=lambda x: x[0])
        
        # Add counts
        obs_parts.append(float(len(friendly_blocks)))
        obs_parts.append(float(len(opponent_blocks)))
        
        # Add friendly block positions
        f_positions = []
        for i in range(min(len(friendly_blocks), MAX_TRACKED)):
            f_positions.extend([friendly_blocks[i][1], friendly_blocks[i][2]])
        # Pad with -999 sentinel for empty slots (can't use -inf, breaks NN)
        while len(f_positions) < MAX_TRACKED * 2:
            f_positions.extend([-999.0, -999.0])
        obs_parts.extend(f_positions)
        
        # Add opponent block positions
        o_positions = []
        for i in range(min(len(opponent_blocks), MAX_TRACKED)):
            o_positions.extend([opponent_blocks[i][1], opponent_blocks[i][2]])
        # Pad with -999 sentinel for empty slots (can't use -inf, breaks NN)
        while len(o_positions) < MAX_TRACKED * 2:
            o_positions.extend([-999.0, -999.0])
        obs_parts.extend(o_positions)
        
        # 7. Blocks added to each goal BY THIS AGENT (4)
        obs_parts.extend([float(x) for x in agent_state["goals_added"]])
        
        # 8. Loaders cleared by this agent (4) - 0 or 1 each
        obs_parts.extend([float(min(x, 1)) for x in agent_state["loaders_taken"]])
        
        # 9. Received messages (8 dims)
        # Read from agent_state["received_messages"] if present, else zeros
        msgs = agent_state.get("received_messages", np.zeros(8, dtype=np.float32))
        if len(msgs) != 8:
            msgs = np.zeros(8, dtype=np.float32)
        obs_parts.extend([float(x) for x in msgs])
        
        # 10. Is executing (1) - populated by base_env when agent is busy
        obs_parts.append(1.0 if agent_state.get("is_executing", False) else 0.0)
        
        # 11. Current action ID (1) - normalized to [0, 1] range
        current_action = agent_state.get("current_action", None)
        if current_action is not None:
            obs_parts.append(float(current_action) / max(1.0, float(len(Actions) - 1)))
        else:
            obs_parts.append(0.0)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        shape = self.get_observation_space_shape()
        
        # Create explicit float32 arrays to avoid precision warnings
        low = np.full(shape, -1e10, dtype=np.float32)
        high = np.full(shape, 1e10, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    @staticmethod
    def get_observation_space_shape() -> Tuple[int]:
        """Get the shape of the observation space."""
        # Self: pos(2) + orient(1) = 3
        # Teammate: 1 * 3 = 3
        # Held blocks: friendly(1) + opponent(1) = 2
        # Parked: 1
        # Time remaining: 1
        # Friendly blocks count on field: 1 (Index 10)
        # Opponent blocks count on field: 1 (Index 11)
        # Friendly block positions: 15 * 2 = 30 (Indices 12-41)
        # Opponent block positions: 15 * 2 = 30 (Indices 42-71)
        # Goals added by this agent: 4 (Indices 72-75)
        # Loaders cleared by this agent: 4 (Indices 76-79)
        # Received messages: 8 (Indices 80-87)
        # Is executing: 1 (Index 88)
        # Current action ID: 1 (Index 89)
        # Total: 3 + 3 + 2 + 1 + 1 + 1 + 1 + 30 + 30 + 4 + 4 + 8 + 1 + 1 = 90
        
        return (ObsIndex.TOTAL,)
    
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent.

        Action space: Tuple(Discrete(action), Box(message_8d), Box(beta_1d))
        - Discrete: which action to take
        - Box(8): communication message vector
        - Box(1): termination signal β (option-critic)
          When idle: ignored. When busy: σ(β) > 0.5 cancels current action.
        """
        return spaces.Tuple((
            spaces.Discrete(self.num_actions),
            spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        ))

    @staticmethod
    def get_action_space_shape() -> Tuple[int]:
        """Get the shape of the action space (Not used directly for Tuple, but helpful)."""
        return ()
    
    @staticmethod
    def get_num_actions() -> int:
        """Get number of available actions."""
        return len(Actions)
    
    # =========================================================================
    # VexGame Game Logic
    # =========================================================================
    
    def execute_action(
        self, 
        agent: str, 
        action: int
    ) -> Tuple[List[ActionStep], float]:
        """Execute an action for an agent.
        
        Returns a list of ActionSteps and a penalty value.
        Does NOT mutate agent position/orientation - the env handles that.
        """
        agent_state = self.state["agents"][agent]
        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()
        
        # Any active non-park action unparks the robot
        if action not in (Actions.IDLE.value, Actions.PARK.value):
            agent_state["parked"] = False
        
        # Default values
        penalty = 0.0
        steps = [ActionStep(
            duration=DEFAULT_DURATION,
            target_pos=start_pos.copy(),
            target_orient=start_orient.copy(),
        )]
        
        if action == Actions.PICK_UP_BLOCK.value:
            steps, penalty = self._action_pickup_block(agent_state, target_team=None)
        
        elif action == Actions.SCORE_IN_LONG_GOAL_1.value:
            steps, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_1
            )
        elif action == Actions.SCORE_IN_LONG_GOAL_2.value:
            steps, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_2
            )
        elif action == Actions.SCORE_IN_CENTER_UPPER.value:
            steps, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_UPPER
            )
        elif action == Actions.SCORE_IN_CENTER_LOWER.value:
            steps, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_LOWER
            )
        
        elif action in [
            Actions.TAKE_FROM_LOADER_TL.value,
            Actions.TAKE_FROM_LOADER_TR.value,
            Actions.TAKE_FROM_LOADER_BL.value,
            Actions.TAKE_FROM_LOADER_BR.value,
        ]:
            idx = action - Actions.TAKE_FROM_LOADER_TL.value
            steps, penalty = self._action_clear_loader(
                agent_state, idx
            )
        
        elif action == Actions.PARK.value:
            steps, penalty = self._action_park(agent_state)
        
        elif action == Actions.TURN_TOWARD_CENTER.value:
            steps, penalty = self._action_turn_toward_center(agent_state)
        
        elif action == Actions.IDLE.value:
            penalty = DEFAULT_PENALTY  # Small penalty for idle
            steps = [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )]
        
        # Update tracker (for training, this keeps tracker in sync with simulation)
        self.update_tracker(agent, action)

        return steps, penalty
    
    def update_tracker(self, agent: str, action: int) -> None:
        """Update agent state field based on action.
        
        Called by execute_action() in training and directly by run_action() in inference.
        Note: Game-state updates are now embedded as ActionEvents in ActionSteps and applied by apply_events().
        This method is kept for compatibility but most updates are now handled via pending events.
        """
        agent_state = self.state["agents"][agent]
    
    def update_observation_from_tracker(self, agent: str, observation: np.ndarray) -> np.ndarray:
        """Update observation array with tracker fields from game state.
        
        Merges internal tracker fields into the observation for inference.
        
        Args:
            agent: Agent name
            observation: Observation array to update (modified in-place and returned)
            
        Returns:
            Updated observation array
        """
        agent_state = self.state["agents"][agent]
        
        # Held blocks (assumes all held blocks are friendly)
        observation[ObsIndex.HELD_FRIENDLY] = float(agent_state.get("held_blocks", 0))
        observation[ObsIndex.HELD_OPPONENT] = 0.0
        
        # Parked status
        observation[ObsIndex.PARKED] = 1.0 if agent_state.get("parked", False) else 0.0
        
        # Goals added by this agent (4 goals)
        goals_added = agent_state.get("goals_added", [0, 0, 0, 0])
        for i, count in enumerate(goals_added):
            observation[ObsIndex.GOALS_ADDED_START + i] = float(count)
        
        # Loaders cleared by this agent (4 loaders)
        loaders_taken = agent_state.get("loaders_taken", [0, 0, 0, 0])
        for i, taken in enumerate(loaders_taken):
            observation[ObsIndex.LOADERS_CLEARED_START + i] = float(taken)
        
        return observation

    def _build_move_action_plan(
        self,
        agent_name: str,
        start_pos: np.ndarray,
        start_orient: np.ndarray,
        target_pos: np.ndarray,
        final_orient: np.ndarray,
        post_action_duration: float,
        post_action_events: List[ActionEvent] = None,
    ) -> Tuple[List[ActionStep], float]:
        """Build standard move-action plan: turn -> move -> turn -> post-action.
        
        Events are attached to the final step (post_action or last turn).
        Returns (steps, total_duration).
        """
        pre_turn_duration = DEFAULT_DURATION
        post_turn_duration = DEFAULT_DURATION

        target_pos_np = np.array(target_pos, dtype=np.float32).copy()
        final_orient_np = np.array(final_orient, dtype=np.float32).copy()

        movement = target_pos_np - start_pos
        dist = float(np.linalg.norm(movement))
        speed = float(self.get_robot_speed(agent_name))
        move_duration = dist / speed if speed > 0 else 0.0

        travel_orientation = (
            np.array([np.arctan2(movement[1], movement[0])], dtype=np.float32)
            if dist > 0
            else start_orient.copy()
        )

        duration = pre_turn_duration + move_duration + post_turn_duration + max(0.0, float(post_action_duration))

        steps: List[ActionStep] = []
        
        # Step 1: Pre-turn to face travel direction
        steps.append(ActionStep(
            duration=float(pre_turn_duration),
            target_pos=start_pos.copy(),
            target_orient=travel_orientation.copy(),
        ))
        
        # Step 2: Move to target
        if move_duration > 0.0:
            steps.append(ActionStep(
                duration=float(move_duration),
                target_pos=target_pos_np.copy(),
                target_orient=travel_orientation.copy(),
            ))
        
        # Step 3: Post-turn to face final orientation
        steps.append(ActionStep(
            duration=float(post_turn_duration),
            target_pos=target_pos_np.copy(),
            target_orient=final_orient_np.copy(),
        ))
        
        # Step 4: Post-action (e.g., scoring, pickup) — events attached here
        if post_action_duration > 0.0:
            steps.append(ActionStep(
                duration=float(post_action_duration),
                target_pos=target_pos_np.copy(),
                target_orient=final_orient_np.copy(),
                events=post_action_events or [],
            ))
        elif post_action_events:
            # If no post-action duration but events exist, attach to last step
            steps[-1].events = post_action_events

        return steps, duration

    def _is_stochastic(self) -> bool:
        return not self.deterministic

    def _visibility_probability(self, distance_inches: float) -> float:
        x = float(distance_inches)
        probability = 1.0 / (
            (1.0 + np.exp(-5.0 * (x - 12.0)))
            * (1.0 + np.exp(0.1 * (x - 48.0)))
        )
        return float(np.clip(probability, 0.0, 1.0))

    def _clamp_position_to_field(self, position: np.ndarray) -> np.ndarray:
        limit = float(FIELD_HALF - BLOCK_RADIUS)
        pos = np.array(position, dtype=np.float32).copy()
        pos[0] = np.clip(pos[0], -limit, limit)
        pos[1] = np.clip(pos[1], -limit, limit)
        return pos

    def _random_point_within_radius(self, center: np.ndarray, radius: float = 12.0) -> np.ndarray:
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        distance = float(radius) * float(np.sqrt(np.random.uniform(0.0, 1.0)))
        point = np.array([
            float(center[0]) + distance * np.cos(angle),
            float(center[1]) + distance * np.sin(angle),
        ], dtype=np.float32)
        return self._clamp_position_to_field(point)

    def _action_pickup_block(
        self, agent_state: Dict, target_team: Optional[str] = None
    ) -> Tuple[List[ActionStep], float]:
        """Pick up nearest block in FOV.
        
        The agent ALWAYS assumes it picked up a block (increments held_blocks).
        In reality, it picks up the nearest block of its team color if one exists in FOV.
        The agent does NOT know what color block it picked up.
        
        Args:
            agent_state: The agent's state
            target_team: If None, picks up robot's team color. Otherwise "red" or "blue".
        """
        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()
        
        # Check if already at max capacity
        if agent_state["held_blocks"] >= MAX_HELD_BLOCKS:
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY
        
        target_idx = -1
        min_dist = float('inf')
        
        robot_pos = agent_state["position"]
        # Camera angle is robot body orientation plus the stored offset
        camera_offset = float(agent_state.get("camera_rotation_offset", 0.0))
        camera_theta = float(agent_state["orientation"][0]) + camera_offset
        robot_team = agent_state["team"]
        
        # Determine which team's blocks to pick up (actual game logic)
        pickup_team = target_team if target_team is not None else robot_team
        
        # Find nearest block of the target team in FOV
        for i, block in enumerate(self.state["blocks"]):
            if block["status"] == BlockStatus.ON_FIELD:
                if block.get("team") != pickup_team:
                    continue
                
                block_pos = block["position"]
                direction = block_pos - robot_pos
                dist = np.linalg.norm(direction)
                
                if dist > 0:
                    angle_to_block = np.arctan2(direction[1], direction[0])
                    angle_diff = angle_to_block - camera_theta
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    if abs(angle_diff) <= FOV / 2:
                        if self._is_stochastic() and np.random.random() > self._visibility_probability(dist):
                            continue
                        if dist < min_dist:
                            min_dist = dist
                            target_idx = i
        
        penalty = 0.0
        
        if target_idx != -1:
            # Block found - build plan to move to it and pick it up
            target_pos = self.state["blocks"][target_idx]["position"].copy()
            movement = target_pos - start_pos
            dist = np.linalg.norm(movement)
            
            target_orient = start_orient.copy()
            if dist > 0:
                target_orient = np.array([np.arctan2(movement[1], movement[0])], dtype=np.float32)
            
            # Build pickup event with stochastic probability
            pickup_event = ActionEvent(
                type="pickup_block",
                data={
                    "agent": agent_state["agent_name"],
                    "block_idx": target_idx,
                },
            )
            
            steps, _ = self._build_move_action_plan(
                agent_name=agent_state["agent_name"],
                start_pos=start_pos,
                start_orient=start_orient,
                target_pos=target_pos,
                final_orient=target_orient,
                post_action_duration=DEFAULT_DURATION,
                post_action_events=[pickup_event],
            )
        else:
            # No block found
            penalty = DEFAULT_PENALTY
            steps = [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )]

        return steps, penalty
    
    def _action_score_in_goal(
        self, 
        agent_state: Dict, 
        goal_type: GoalType
    ) -> Tuple[List[ActionStep], float]:
        """Score held blocks in a goal."""
        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()
        
        if agent_state["held_blocks"] <= 0:
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY

        goal = self.goal_manager.get_goal(goal_type)
        scoring_side = goal.get_nearest_side(agent_state["position"])
        
        # Calculate robot position based on goal type
        robot_len, _ = self.get_robot_dimensions(agent_state["agent_name"])
        
        if goal_type == GoalType.LONG_1:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - robot_len/2 - 2.0, 48.0])
                orientation = 0.0
            else:
                robot_pos = np.array([24.0 + robot_len/2 + 2.0, 48.0])
                orientation = np.pi
        elif goal_type == GoalType.LONG_2:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - robot_len/2 - 2.0, -48.0])
                orientation = 0.0
            else:
                robot_pos = np.array([24.0 + robot_len/2 + 2.0, -48.0])
                orientation = np.pi
        elif goal_type == GoalType.CENTER_UPPER:
            offset = robot_len/2 + 4.0
            if scoring_side == "left":
                robot_pos = np.array([-8.5 - offset * 0.707, 8.5 + offset * 0.707])
                orientation = -np.pi / 4
            else:
                robot_pos = np.array([8.5 + offset * 0.707, -8.5 - offset * 0.707])
                orientation = 3 * np.pi / 4
        else:  # CENTER_LOWER
            offset = robot_len/2 + 4.0
            if scoring_side == "left":
                robot_pos = np.array([8.5 + offset * 0.707, 8.5 + offset * 0.707])
                orientation = -3 * np.pi / 4
            else:
                robot_pos = np.array([-8.5 - offset * 0.707, -8.5 - offset * 0.707])
                orientation = np.pi / 4
        
        # Build per-block scoring events with stochastic probability
        agent_name = agent_state["agent_name"]
        robot_team = agent_state["team"]
        goal_center = GOALS[goal_type].center
        # Eject blocks near the scoring entry, not the goal center
        scoring_entry = (
            goal.left_entry.copy() if scoring_side == "left" else goal.right_entry.copy()
        )
        
        score_events: List[ActionEvent] = []
        for block_idx, block in enumerate(self.state["blocks"]):
            if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                if block.get("team") == robot_team:
                    # Friendly block: try to score, may fail stochastically
                    score_events.append(ActionEvent(
                        type="score_block",
                        data={
                            "agent": agent_name,
                            "goal_type": goal_type,
                            "block_idx": block_idx,
                            "scoring_side": scoring_side,
                            "is_friendly": True,
                            "scoring_entry": scoring_entry.copy(),
                        },
                    ))
                else:
                    # Wrong color block - always eject
                    score_events.append(ActionEvent(
                        type="eject_block",
                        data={
                            "block_idx": block_idx,
                            "goal_center": scoring_entry.copy(),
                        },
                    ))
        
        scored_count = sum(1 for e in score_events if e.type == "score_block")
        post_duration = DEFAULT_DURATION * max(1, scored_count)
        
        steps, _ = self._build_move_action_plan(
            agent_name=agent_state["agent_name"],
            start_pos=start_pos,
            start_orient=start_orient,
            target_pos=robot_pos.astype(np.float32),
            final_orient=np.array([orientation], dtype=np.float32),
            post_action_duration=post_duration,
            post_action_events=score_events,
        )

        return steps, 0.0
    
    def _update_goal_block_positions(
        self, goal: GoalQueue, target_status: int
    ) -> None:
        """Update positions of blocks in a goal."""
        block_positions = goal.get_block_positions(goal.goal_position.center)
        idx_to_position = {block_idx: pos for block_idx, pos in block_positions}
        
        for block_idx, block in enumerate(self.state["blocks"]):
            if block_idx in idx_to_position:
                block["position"] = idx_to_position[block_idx].copy()
                block["status"] = target_status
    
    def _action_clear_loader(
        self, 
        agent_state: Dict, 
        loader_idx: int
    ) -> Tuple[List[ActionStep], float]:
        """Clear all blocks from a loader (can only be done once per loader per agent).
        
        The agent collects all 6 blocks from the loader at once.
        The agent does NOT know what colors the blocks are.
        """
        BLOCKS_PER_LOADER = 6
        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()
        
        # Check if this loader has already been cleared by this agent
        if agent_state["loaders_taken"][loader_idx] >= 1:
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY

        loader_pos = LOADERS[loader_idx].position
        robot_len, _ = self.get_robot_dimensions(agent_state["agent_name"])
        offset = robot_len / 2 + 8.0
        
        if loader_idx == 0:  # Top Left
            orientation = np.pi
            robot_pos = np.array([loader_pos[0] + offset, loader_pos[1]])
        elif loader_idx == 1:  # Top Right
            orientation = 0.0
            robot_pos = np.array([loader_pos[0] - offset, loader_pos[1]])
        elif loader_idx == 2:  # Bottom Left
            orientation = np.pi
            robot_pos = np.array([loader_pos[0] + offset, loader_pos[1]])
        else:  # Bottom Right
            orientation = 0.0
            robot_pos = np.array([loader_pos[0] - offset, loader_pos[1]])
        
        # Build loader clear event
        loader_event = ActionEvent(
            type="clear_loader",
            data={
                "agent": agent_state["agent_name"],
                "loader_idx": loader_idx,
            },
        )
        
        # Time to collect all assumed 6 blocks
        post_duration = DEFAULT_DURATION * BLOCKS_PER_LOADER
        steps, _ = self._build_move_action_plan(
            agent_name=agent_state["agent_name"],
            start_pos=start_pos,
            start_orient=start_orient,
            target_pos=robot_pos.astype(np.float32),
            final_orient=np.array([orientation], dtype=np.float32),
            post_action_duration=post_duration,
            post_action_events=[loader_event],
        )

        return steps, 0.0
    
    def _action_park(
        self, agent_state: Dict
    ) -> Tuple[List[ActionStep], float]:
        """Park in team's zone."""
        team = agent_state["team"]
        park_zone = PARK_ZONES[team]

        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()

        # Already parked: treat as invalid to prevent repeated park reward farming
        if agent_state.get("parked", False):
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY

        # Prevent parking early: only allow when 15 seconds or less remain.
        current_time = float(agent_state.get("game_time", 0.0))
        time_remaining = float(self.total_time - current_time)
        if time_remaining > 15.0:
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY

        # Only one robot can park per team
        for other_agent, other_state in self.state["agents"].items():
            if other_agent != agent_state["agent_name"]:
                if other_state["team"] == team and other_state.get("parked", False):
                    return [ActionStep(
                        duration=0.1,
                        target_pos=start_pos.copy(),
                        target_orient=start_orient.copy(),
                    )], DEFAULT_PENALTY
        
        final_orient = np.array([np.random.choice([np.pi/2, -np.pi/2])], dtype=np.float32)
        
        # Build park event
        park_event = ActionEvent(
            type="park",
            data={"agent": agent_state["agent_name"]},
        )
        
        post_duration = DEFAULT_DURATION
        steps, _ = self._build_move_action_plan(
            agent_name=agent_state["agent_name"],
            start_pos=start_pos,
            start_orient=start_orient,
            target_pos=park_zone.center.copy(),
            final_orient=final_orient,
            post_action_duration=post_duration,
            post_action_events=[park_event],
        )

        return steps, 0.0
    
    def _action_turn_toward_center(
        self, agent_state: Dict
    ) -> Tuple[List[ActionStep], float]:
        """Turn robot to face center."""
        start_pos = agent_state["position"].copy()
        start_orient = agent_state["orientation"].copy()

        direction = np.array([0.0, 0.0]) - agent_state["position"]
        target_camera_angle = np.arctan2(direction[1], direction[0])
        camera_offset = float(agent_state.get("camera_rotation_offset", 0.0))

        current_body_angle = float(agent_state["orientation"][0])
        current_camera_angle = current_body_angle + camera_offset
        target_body_angle = target_camera_angle - camera_offset

        angle_error = np.arctan2(
            np.sin(target_camera_angle - current_camera_angle),
            np.cos(target_camera_angle - current_camera_angle),
        )

        # If camera is already facing center, give a small penalty
        if abs(angle_error) < 1e-4:
            return [ActionStep(
                duration=0.1,
                target_pos=start_pos.copy(),
                target_orient=start_orient.copy(),
            )], DEFAULT_PENALTY
        
        # Build turn event
        turn_event = ActionEvent(
            type="turn_toward_center",
            data={
                "agent": agent_state["agent_name"],
                "target_angle": target_body_angle,
            },
        )

        steps = [ActionStep(
            duration=DEFAULT_DURATION,
            target_pos=start_pos.copy(),
            target_orient=np.array([target_body_angle], dtype=np.float32),
            events=[turn_event],
        )]

        return steps, 0.0
    
    @abstractmethod
    def compute_score(self) -> Dict[str, int]:
        """Compute the score for the current state.
        Returns:
            Dict[str, int]: Team scores
        """
        pass
    
    def get_team_for_agent(self, agent: str) -> str:
        """Get team for an agent."""
        if "red" in agent.lower():
            return "red"
        elif "blue" in agent.lower():
            return "blue"
        return "red"
    
    def is_agent_terminated(self, agent: str, game_time: float = 0.0) -> bool:
        """
        Check if an agent has terminated.
        
        In Push Back, an agent terminates when:
        - Their game time exceeds the total time limit, OR
        - They have parked
        """
        agent_state = self.state["agents"][agent]
        
        # Time limit exceeded
        if game_time >= self.total_time:
            return True
        
        # Uncomment to enable parking termination
        # # Parked (Push Back specific)
        # if agent_state.get("parked", False):
        #     return True
        
        return False
    
    def is_valid_action(self, action: int, observation: np.ndarray) -> bool:
        """Check if action is valid based on observation.
        
        Uses ObsIndex constants for observation layout.
        """
        held_friendly = observation[ObsIndex.HELD_FRIENDLY]
        held_opponent = observation[ObsIndex.HELD_OPPONENT]
        total_held = held_friendly + held_opponent
        
        # Scoring actions require held blocks (friendly ones to score)
        if is_scoring_action(Actions(action)):
            if held_friendly <= 0:
                return False
        
        # Pickup action - invalid if at max capacity OR no friendly blocks present
        if action == Actions.PICK_UP_BLOCK.value:
            if total_held >= MAX_HELD_BLOCKS or observation[ObsIndex.FRIENDLY_BLOCK_COUNT] <= 0:
                return False
        
        # Clear loader actions - can only clear each loader once
        if action == Actions.TAKE_FROM_LOADER_TL.value:
            if observation[ObsIndex.LOADERS_CLEARED_START + 0] >= 1:
                return False
        elif action == Actions.TAKE_FROM_LOADER_TR.value:
            if observation[ObsIndex.LOADERS_CLEARED_START + 1] >= 1:
                return False
        elif action == Actions.TAKE_FROM_LOADER_BL.value:
            if observation[ObsIndex.LOADERS_CLEARED_START + 2] >= 1:
                return False
        elif action == Actions.TAKE_FROM_LOADER_BR.value:
            if observation[ObsIndex.LOADERS_CLEARED_START + 3] >= 1:
                return False
        
        if action == Actions.PARK.value:
            # Parking is only valid if 15 seconds or less remain
            time_remaining = observation[ObsIndex.TIME_REMAINING]
            if time_remaining > 15.0:
                return False
        
        return True

    def update_robot_position(self, agent_name: str, position: np.ndarray) -> None:
        for block in self.state["blocks"]:
            if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                block["position"] = position.copy()
            
    def apply_events(self, agent: str, events: List[ActionEvent]) -> None:
        """Apply resolved ActionEvents for an agent.
        
        Called by the environment when a step's ticks are exhausted.
        Events have already been resolved for stochasticity (probability rolls
        handled by the env's _resolve_event_probabilities).
        """
        agent_state = self.state["agents"].get(agent)
        if agent_state is None:
            return
        
        for event in events:
            if event.type == "clear_loader":
                loader_idx = event.data.get("loader_idx")
                block_indices = [i for i, b in enumerate(self.state["blocks"]) if b["status"] == BlockStatus.IN_LOADER_TL + loader_idx]

                # Update actual loader count
                if "loaders" in self.state and 0 <= loader_idx < len(self.state["loaders"]):
                    self.state["loaders"][loader_idx] = 0

                # Update agent tracker
                agent_state["loaders_taken"][loader_idx] = 1
                loader_center = LOADERS[loader_idx].position
                success_count = 0

                for idx in block_indices:
                    if 0 <= idx < len(self.state["blocks"]):
                        b = self.state["blocks"][idx]
                        if not self._is_stochastic():
                            outcome = "success"
                        else:
                            roll = np.random.random()
                            if roll < 0.65:
                                outcome = "success"
                            elif roll < 0.75:
                                outcome = "ground"
                            else:
                                outcome = "stays"
                        
                        if outcome == "success":
                            b["status"] = BlockStatus.HELD
                            b["held_by"] = agent
                            b["position"] = agent_state["position"].copy()
                            success_count += 1
                        elif outcome == "ground":
                            b["status"] = BlockStatus.ON_FIELD
                            b["held_by"] = None
                            b["position"] = self._random_point_within_radius(loader_center)
                        # else outcome == "stays", so do nothing and the block stays in the loader

                agent_state["held_blocks"] += success_count
                if agent_state["held_blocks"] > MAX_HELD_BLOCKS:
                    agent_state["held_blocks"] = MAX_HELD_BLOCKS
            
            elif event.type == "pickup_block":
                block_idx = event.data.get("block_idx", -1)
                # Stochastic pickup check (90% success if stochastic)
                pickup_success = (not self._is_stochastic()) or (np.random.random() < 0.90)
                
                if pickup_success and block_idx >= 0 and 0 <= block_idx < len(self.state["blocks"]):
                    agent_state["held_blocks"] += 1
                    if agent_state["held_blocks"] > MAX_HELD_BLOCKS:
                        agent_state["held_blocks"] = MAX_HELD_BLOCKS

                    b = self.state["blocks"][block_idx]
                    b["status"] = BlockStatus.HELD
                    b["held_by"] = agent
                    b["position"] = agent_state["position"].copy()
            
            elif event.type == "score_block":
                goal_type = event.data.get("goal_type")
                block_idx = event.data.get("block_idx")
                scoring_side = event.data.get("scoring_side")
                
                # Stochastic scoring check (90% success if stochastic)
                score_success = (not self._is_stochastic()) or (np.random.random() < 0.90)
                
                if score_success and 0 <= block_idx < len(self.state["blocks"]):
                    target_status = BlockStatus.get_status_for_goal(goal_type)
                    b = self.state["blocks"][block_idx]
                    
                    goal = self.goal_manager.get_goal(goal_type)
                    ejected_idx, _ = goal.add_block_from_nearest(block_idx, agent_state["position"])
                    
                    b["status"] = target_status
                    b["held_by"] = None
                    
                    # Handle ejected block from goal overflow
                    if ejected_idx is not None and 0 <= ejected_idx < len(self.state["blocks"]):
                        eb = self.state["blocks"][ejected_idx]
                        eb["status"] = BlockStatus.ON_FIELD
                        eb["held_by"] = None
                        eb["position"] = self._clamp_position_to_field(
                            goal.right_entry.copy() if scoring_side == "left" else goal.left_entry.copy()
                        )
                    
                    # Update block positions in goal
                    self._update_goal_block_positions(goal, target_status)
                    
                    # Track goals added
                    goal_idx = [GoalType.LONG_1, GoalType.LONG_2, GoalType.CENTER_UPPER, GoalType.CENTER_LOWER].index(goal_type)
                    agent_state["goals_added"][goal_idx] += 1
                    
                    # Decrement held blocks
                    agent_state["held_blocks"] = max(0, agent_state["held_blocks"] - 1)
                elif not score_success and 0 <= block_idx < len(self.state["blocks"]):
                    # Eject block on failure
                    scoring_entry = event.data.get("scoring_entry")
                    if scoring_entry is not None:
                        b = self.state["blocks"][block_idx]
                        b["status"] = BlockStatus.ON_FIELD
                        b["held_by"] = None
                        b["position"] = self._clamp_position_to_field(scoring_entry)
                    
                    # Decrement held blocks since it left the robot
                    agent_state["held_blocks"] = max(0, agent_state["held_blocks"] - 1)
            
            elif event.type == "eject_block":
                block_idx = event.data.get("block_idx")
                goal_center = event.data.get("goal_center")
                if 0 <= block_idx < len(self.state["blocks"]):
                    b = self.state["blocks"][block_idx]
                    b["status"] = BlockStatus.ON_FIELD
                    b["held_by"] = None
                    if goal_center is not None:
                        b["position"] = self._clamp_position_to_field(
                            self._random_point_within_radius(goal_center)
                        )
                    else:
                        b["position"] = self._clamp_position_to_field(agent_state["position"].copy())
                    
                    # Decrement held blocks
                    agent_state["held_blocks"] = max(0, agent_state["held_blocks"] - 1)
            
            elif event.type == "park":
                agent_state["parked"] = True
            
            elif event.type == "turn_toward_center":
                target_angle = event.data.get("target_angle", 0.0)
                agent_state["orientation"] = np.array([target_angle], dtype=np.float32)
            
            # Unknown event types are ignored

    def get_permanent_obstacles(self) -> List[Obstacle]:
        """Get permanent obstacles."""
        return PERMANENT_OBSTACLES
    
    def render_info_panel(
        self, 
        ax_info: Any, 
        agents: List[str] = None,
        actions: Optional[Dict] = None,
        rewards: Optional[Dict] = None,
        num_steps: int = 0,
        agent_times: Optional[Dict[str, float]] = None,
        action_time_remaining: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Render Push Back specific info panel with held blocks by color.
        """
        if agents is None:
            agents = list(self.state["agents"].keys())
        info_y = 0.95
        ax_info.text(0.5, info_y, "Agent Actions", fontsize=12, fontweight='bold',
                    ha='center', va='top')
        info_y -= 0.08
        
        for i, agent in enumerate(agents):
            st = self.state["agents"][agent]
            team = self.get_team_for_agent(agent)
            robot_color = 'red' if team == 'red' else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            
            # Count held blocks by color
            held_red = 0
            held_blue = 0
            for block in self.state["blocks"]:
                if block["status"] == BlockStatus.HELD and block.get("held_by") == agent:
                    if block.get("team") == "red":
                        held_red += 1
                    else:
                        held_blue += 1
            
            ax_info.text(0.05, info_y, f"Robot {i} ({team}):",
                        fontsize=9, color=robot_color, fontweight='bold', va='top')
            info_y -= 0.05

            # Persistent last action name + reward (stored on agent state)
            last_name = st.get("last_action_name", "---")
            last_reward = st.get("last_action_reward", None)
            reward_str = f" (R: {last_reward:.2f})" if last_reward is not None else ""
            ax_info.text(0.1, info_y, f"Last: {last_name}{reward_str}", fontsize=8, va='top')
            info_y -= 0.03

            # Current action being executed (from agent state)
            current_val = st.get("current_action", None)
            try:
                current_text = self.action_to_name(int(current_val)) if current_val is not None else "---"
            except Exception:
                current_text = str(current_val) if current_val is not None else "---"
            ax_info.text(0.1, info_y, f"Current: {current_text}", fontsize=8, va='top')
            info_y -= 0.03

            remaining_action_time = 0.0
            if action_time_remaining and agent in action_time_remaining:
                remaining_action_time = float(action_time_remaining[agent])
            ax_info.text(0.1, info_y,
                        f"Action Time Left: {remaining_action_time:.1f}s",
                        fontsize=7, va='top', color='gray')
            info_y -= 0.03
            
            # Show time
            current_time = 0.0
            if agent_times and agent in agent_times:
                current_time = agent_times[agent]
            
            ax_info.text(0.1, info_y, 
                        f"Time: {current_time:.1f}s / {self.total_time:.0f}s",
                        fontsize=7, va='top', color='gray')
            info_y -= 0.03
            # Show held blocks by color
            held_text = f"Pos: ({x:.0f}, {y:.0f}) | Held: "
            if held_red > 0 or held_blue > 0:
                parts = []
                if held_red > 0:
                    parts.append(f"{held_red}R")
                if held_blue > 0:
                    parts.append(f"{held_blue}B")
                held_text += " ".join(parts)
            else:
                held_text += "0"
            ax_info.text(0.1, info_y, held_text, fontsize=7, va='top', color='gray')
            info_y -= 0.06
        
        # Score section
        info_y -= 0.02
        ax_info.axhline(y=info_y, xmin=0.05, xmax=0.95, color='gray', linewidth=0.5)
        info_y -= 0.05
        
        team_scores = self.compute_score()
        ax_info.text(0.05, info_y, "Scores:", fontsize=10, fontweight='bold', va='top')
        info_y -= 0.04
        ax_info.text(0.1, info_y, f"Red: {team_scores.get('red', 0)}",
                    fontsize=9, va='top', color='red', fontweight='bold')
        info_y -= 0.04
        if 'blue' in team_scores:
            ax_info.text(0.1, info_y, f"Blue: {team_scores.get('blue', 0)}",
                        fontsize=9, va='top', color='blue', fontweight='bold')
            info_y -= 0.04
        info_y -= 0.01
        ax_info.text(0.05, info_y, f"Step: {num_steps}", fontsize=8, va='top')
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def render_game_elements(self, ax: Any) -> None:
        """Render Push Back game elements."""
        import colorsys
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors
        import matplotlib.transforms as mtransforms

        def _point_visibility_probability(point: np.ndarray) -> float:
            agents = self.state.get("agents", {})
            if not agents:
                return 1.0

            max_probability = 0.0
            for _, agent_state in agents.items():
                agent_pos = np.asarray(agent_state["position"], dtype=float)
                camera_offset = float(agent_state.get("camera_rotation_offset", 0.0))
                camera_theta = float(agent_state["orientation"][0]) + camera_offset
                to_point = np.asarray(point, dtype=float) - agent_pos
                distance = float(np.linalg.norm(to_point))
                if distance <= 1e-6 or distance > 72.0:
                    continue
                angle_to_point = float(np.arctan2(to_point[1], to_point[0]))
                angle_delta = float(np.arctan2(np.sin(angle_to_point - camera_theta), np.cos(angle_to_point - camera_theta)))
                if abs(angle_delta) <= float(FOV) / 2.0:
                    max_probability = max(max_probability, float(self._visibility_probability(distance)))

            return float(np.clip(max_probability, 0.0, 1.0))

        def _visibility_style(color: str, visibility_probability: float, min_alpha: float = 0.45) -> tuple:
            probability = float(np.clip(visibility_probability, 0.0, 1.0))
            rgb = np.array(mcolors.to_rgb(color), dtype=float)
            hue, lightness, saturation = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
            adjusted_saturation = saturation * (0.60 + 0.40 * probability)
            adjusted_rgb = np.array(colorsys.hls_to_rgb(hue, lightness, adjusted_saturation), dtype=float)
            gray = np.array([0.74, 0.74, 0.74], dtype=float)
            blend_strength = (1.0 - probability) * 0.18
            output_rgb = np.clip(adjusted_rgb * (1.0 - blend_strength) + gray * blend_strength, 0.0, 1.0)
            alpha = min_alpha + (1.0 - min_alpha) * probability
            return output_rgb, float(np.clip(alpha, 0.0, 1.0))
        
        # Park Zones
        red_park_style, red_park_alpha = _visibility_style('red', _point_visibility_probability(np.array([-63.0, 0.0])))
        rect_park_red = patches.Rectangle(
            (-72, -12), 18, 24,
            linewidth=1, edgecolor=red_park_style, facecolor=red_park_style, hatch='//', alpha=red_park_alpha * 0.22
        )
        ax.add_patch(rect_park_red)
        
        blue_park_style, blue_park_alpha = _visibility_style('blue', _point_visibility_probability(np.array([63.0, 0.0])))
        rect_park_blue = patches.Rectangle(
            (54, -12), 18, 24,
            linewidth=1, edgecolor=blue_park_style, facecolor=blue_park_style, hatch='//', alpha=blue_park_alpha * 0.22
        )
        ax.add_patch(rect_park_blue)
        
        # Long Goal 1 (top)
        long_goal_top_prob = _point_visibility_probability(np.array([0.0, 48.0]))
        long_goal_top_style, long_goal_top_alpha = _visibility_style('orange', long_goal_top_prob)
        rect_lg_1 = patches.Rectangle(
            (-24, 46), 48, 4, 
            facecolor=long_goal_top_style, alpha=long_goal_top_alpha * 0.5, edgecolor=long_goal_top_style
        )
        ax.add_patch(rect_lg_1)
        ax.text(0, 52, 'Long 1', fontsize=8, ha='center', va='bottom', color=long_goal_top_style, alpha=long_goal_top_alpha)
        ax.plot([-6, -6], [46, 50], color='white', linewidth=2, linestyle='--', alpha=long_goal_top_alpha)
        ax.plot([6, 6], [46, 50], color='white', linewidth=2, linestyle='--', alpha=long_goal_top_alpha)
        
        # Long Goal 2 (bottom)
        long_goal_bottom_prob = _point_visibility_probability(np.array([0.0, -48.0]))
        long_goal_bottom_style, long_goal_bottom_alpha = _visibility_style('orange', long_goal_bottom_prob)
        rect_lg_2 = patches.Rectangle(
            (-24, -50), 48, 4, 
            facecolor=long_goal_bottom_style, alpha=long_goal_bottom_alpha * 0.5, edgecolor=long_goal_bottom_style
        )
        ax.add_patch(rect_lg_2)
        ax.text(0, -54, 'Long 2', fontsize=8, ha='center', va='top', color=long_goal_bottom_style, alpha=long_goal_bottom_alpha)
        ax.plot([-6, -6], [-50, -46], color='white', linewidth=2, linestyle='--', alpha=long_goal_bottom_alpha)
        ax.plot([6, 6], [-50, -46], color='white', linewidth=2, linestyle='--', alpha=long_goal_bottom_alpha)
        
        # Center Structure (X shape)
        w, h = 24, 4
        center_upper_style, center_upper_alpha = _visibility_style('green', _point_visibility_probability(np.array([0.0, 0.0])))
        rect_center_upper = patches.Rectangle(
            (-w/2, -h/2), w, h, 
            facecolor=center_upper_style, alpha=center_upper_alpha * 0.55, edgecolor=center_upper_style,
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, -45) + ax.transData
        )
        ax.add_patch(rect_center_upper)
        ax.text(-10, 10, 'Upper', fontsize=6, ha='center', va='center', color=center_upper_style, alpha=center_upper_alpha)
        
        center_lower_style, center_lower_alpha = _visibility_style('purple', _point_visibility_probability(np.array([0.0, 0.0])))
        rect_center_lower = patches.Rectangle(
            (-w/2, -h/2), w, h, 
            facecolor=center_lower_style, alpha=center_lower_alpha * 0.55, edgecolor=center_lower_style,
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, 45) + ax.transData
        )
        ax.add_patch(rect_center_lower)
        ax.text(-10, -10, 'Lower', fontsize=6, ha='center', va='center', color=center_lower_style, alpha=center_lower_alpha)
        
        # Loaders
        for idx, loader in enumerate(LOADERS):
            loader_probability = _point_visibility_probability(np.array([loader.position[0], loader.position[1]]))
            loader_style, loader_alpha = _visibility_style('orange', loader_probability)
            circle = patches.Circle(
                (loader.position[0], loader.position[1]),
                6.0, fill=False, edgecolor=loader_style, linewidth=4, alpha=loader_alpha
            )
            ax.add_patch(circle)
            
            # Find actual blocks in this loader
            loader_status = BlockStatus.IN_LOADER_TL + idx
            loader_blocks = [b for b in self.state["blocks"] if b["status"] == loader_status]
            
            # Blocks are stored Top-to-Bottom in list (index 0 is Top)
            # We want to render them in a stack.
            # Let's say we render Top at positive Y offset (visually "up")?
            # Or just splayed out.
            
            num_blocks = len(loader_blocks)
            if num_blocks > 0:
                block_spacing = 2.5 # Splay them out to be visible
                # Start centered
                
                # Render Bottom-to-Top visually so they stack?
                # Or just list them.
                # Let's render index 0 (Top) at the "top" (Highest Y)
                
                for i, block in enumerate(loader_blocks):
                    # Visual offset: Top block (i=0) should be highest/most visible?
                    # Let's splay them along Y axis.
                    # Top block at offset +Y. Bottom at -Y.
                    # num_blocks=6. i=0 (Top).
                    # y_offset = (num_blocks-1-i) * spacing - (height/2)
                    
                    # Simpler: Just Stack them. Center is loader pos.
                    # spread from y = pos + 3 to pos - 3
                    
                    # Reverse index for Y position so First(Top) is Highest
                    y_offset = ((num_blocks - 1) * 0.5 - i) * block_spacing
                    
                    block_y = loader.position[1] + y_offset
                    block_color = block.get("team", "grey")
                    block_probability = _point_visibility_probability(np.array([loader.position[0], block_y]))
                    block_style, block_alpha = _visibility_style(block_color, block_probability)
                    
                    hexagon = patches.RegularPolygon(
                        (loader.position[0], block_y),
                        numVertices=6, radius=1.8, orientation=0,
                        facecolor=block_style, edgecolor='black', linewidth=1, alpha=block_alpha
                    )
                    ax.add_patch(hexagon)
        
        # Obstacles
        for obs in PERMANENT_OBSTACLES:
            obstacle_style, obstacle_alpha = _visibility_style('black', _point_visibility_probability(np.array([obs.x, obs.y])))
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius,
                fill=False, edgecolor=obstacle_style, linestyle=':', linewidth=2, alpha=obstacle_alpha
            )
            ax.add_patch(circle)
        
        # Blocks (not in loaders)
        for block in self.state["blocks"]:
            if block["status"] < BlockStatus.IN_LOADER_TL:
                try:
                    fill_color = block.get("team", "red").value
                except AttributeError:
                    fill_color = block.get("team", "red")
                
                if block["status"] == BlockStatus.ON_FIELD:
                    edge_color = 'black'
                    edge_width = 1
                elif block["status"] == BlockStatus.HELD:
                    edge_color = 'yellow'
                    edge_width = 3
                else:
                    edge_color = 'white'
                    edge_width = 2

                block_probability = _point_visibility_probability(np.array([block["position"][0], block["position"][1]]))
                face_style, block_alpha = _visibility_style(fill_color, block_probability, min_alpha=0.55)
                edge_style, _ = _visibility_style(edge_color, block_probability)
                
                hexagon = patches.RegularPolygon(
                    (block["position"][0], block["position"][1]),
                    numVertices=6, radius=2.4, orientation=0,
                    facecolor=face_style, edgecolor=edge_style, linewidth=edge_width, alpha=block_alpha
                )
                ax.add_patch(hexagon)
        
        # Robot FOV wedges
        for agent_name, agent_state in self.state["agents"].items():
            x, y = agent_state["position"]
            camera_offset = float(agent_state.get("camera_rotation_offset", 0.0))
            theta = float(agent_state["orientation"][0]) + camera_offset
            
            fov_radius = 72
            fov_start_angle = np.degrees(theta - FOV/2)
            fov_end_angle = np.degrees(theta + FOV/2)
            num_segments = 12
            for seg_idx in range(num_segments):
                inner_radius = (seg_idx / num_segments) * fov_radius
                outer_radius = ((seg_idx + 1) / num_segments) * fov_radius
                mid_radius = 0.5 * (inner_radius + outer_radius)
                radial_alpha = 0.25 * max(0.0, 1.0 - (mid_radius / fov_radius))
                if radial_alpha <= 0.001:
                    continue
                fov_wedge = patches.Wedge(
                    (x, y), outer_radius, fov_start_angle, fov_end_angle,
                    width=(outer_radius - inner_radius),
                    facecolor='yellow',
                    alpha=radial_alpha,
                    edgecolor='none',
                    linewidth=0.0
                )
                ax.add_patch(fov_wedge)
    
    def action_to_name(self, action: int) -> str:
        """
        Convert an action index to a human-readable name.
        
        Args:
            action: Action index
            
        Returns:
            Human-readable action name
        """
        return Actions(action).name

    def split_action(self, action: int, observation: np.ndarray, robot: Robot) -> List[str]:
        """
        Split a high-level action into low-level robot instructions.

        Possible actions:
        INTAKE;speed
        FOLLOW;(x1,y1),(x2,y2),...;speed
        WAIT;time
        DRIVE;inches;speed
        TURN;degrees;speed
        TURN_TO;heading;speed
        TURN_TO_POINT;(x,y);speed
        SCORE_HIGH
        SCORE_LOW
        SCORE_MIDDLE
        STOP_TRANSFER
        CLEAR_LOADER
        """
        
        actions = []
        
        # Get robot position from observation (in inches)
        robot_pos = np.array([observation[ObsIndex.SELF_POS_X], observation[ObsIndex.SELF_POS_Y]])
        start_pos = [robot_pos[0], robot_pos[1]]

        def get_path(start_pos, end_pos):
            positions, velocities, dt = self.path_planner.Solve(start_point=start_pos, end_point=end_pos, obstacles=[], robot=robot)
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])
            return points_str
        
        # Scoring action mappings: action -> (goal_type, score_cmd)
        SCORING_ACTIONS = {
            Actions.SCORE_IN_LONG_GOAL_1.value: (GoalType.LONG_1, "SCORE_HIGH"),
            Actions.SCORE_IN_LONG_GOAL_2.value: (GoalType.LONG_2, "SCORE_HIGH"),
            Actions.SCORE_IN_CENTER_UPPER.value: (GoalType.CENTER_UPPER, "SCORE_MIDDLE"),
            Actions.SCORE_IN_CENTER_LOWER.value: (GoalType.CENTER_LOWER, "SCORE_LOW"),
        }
        
        # Loader action mappings: action -> loader_index
        LOADER_ACTIONS = {
            Actions.TAKE_FROM_LOADER_TL.value: 0,  # Top Left
            Actions.TAKE_FROM_LOADER_TR.value: 1,  # Top Right
            Actions.TAKE_FROM_LOADER_BL.value: 2,  # Bottom Left
            Actions.TAKE_FROM_LOADER_BR.value: 3,  # Bottom Right
        }
        
        if action == Actions.PICK_UP_BLOCK.value:
            nearest_block_x = observation[ObsIndex.FRIENDLY_BLOCKS_START]
            nearest_block_y = observation[ObsIndex.FRIENDLY_BLOCKS_START + 1]
            target_pos = [nearest_block_x, nearest_block_y]
            
            actions.append("INTAKE;100")
            actions.append(f"FOLLOW;{get_path(start_pos, target_pos)};50")
            actions.append("WAIT;0.5")
            actions.append("STOP_TRANSFER")

        elif action in SCORING_ACTIONS:
            goal_type, score_cmd = SCORING_ACTIONS[action]
            goal = GOALS[goal_type]
            nearest_entry = goal.get_nearest_entry(robot_pos)
            target_pos = [nearest_entry[0], nearest_entry[1]]
            
            actions.append(f"FOLLOW;{get_path(start_pos, target_pos)};50")
            actions.append(f"TURN_TO_POINT;({goal.center[0]:.1f},{goal.center[1]:.1f});30")
            actions.append("DRIVE;6;30")
            actions.append(score_cmd)

        elif action in LOADER_ACTIONS:
            loader_idx = LOADER_ACTIONS[action]
            loader = LOADERS[loader_idx]
            # Approach from inside the field (12 inches from wall)
            offset = 12.0 if loader.position[0] < 0 else -12.0
            approach_pos = [loader.position[0] + offset, loader.position[1]]
            
            actions.append(f"FOLLOW;{get_path(start_pos, approach_pos)};50")
            actions.append(f"TURN_TO_POINT;({loader.position[0]:.1f},{loader.position[1]:.1f});40")
            actions.append("DRIVE;6;30")
            actions.append("CLEAR_LOADER")

        elif action == Actions.PARK.value:
            if robot.team.value == "red":
                park_zone = PARK_ZONES["red"]
                target_pos = [park_zone.center[0], park_zone.center[1]]
                approach_pos = [park_zone.center[0] + 24, park_zone.center[1]]
            else:
                park_zone = PARK_ZONES["blue"]
                target_pos = [park_zone.center[0], park_zone.center[1]]
                approach_pos = [park_zone.center[0] - 24, park_zone.center[1]]
            
            actions.append(f"FOLLOW;{get_path(start_pos, approach_pos)};60")
            actions.append(f"TURN_TO_POINT;({target_pos[0]:.1f},{target_pos[1]:.1f});40")
            actions.append("DRIVE;24;30")

        elif action == Actions.TURN_TOWARD_CENTER.value:
            target_camera_angle = np.arctan2(-robot_pos[1], -robot_pos[0])
            camera_offset = float(getattr(robot, "camera_rotation_offset", 0.0))
            target_body_angle_deg = np.degrees(target_camera_angle - camera_offset)
            actions.append(f"TURN_TO;{target_body_angle_deg:.1f};40")

        elif action == Actions.IDLE.value:
            actions.append("WAIT;0.5")

        return actions
