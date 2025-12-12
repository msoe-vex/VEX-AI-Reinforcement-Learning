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
from vex_core.base_game import VexGame
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
DEFAULT_PENALTY = 0.1 # Will be subtracted from reward


# =============================================================================
# Actions
# =============================================================================

class Actions(Enum):
    """Available actions for robots in the Push Back game."""
    PICK_UP_NEAREST_BLOCK = 0
    SCORE_IN_LONG_GOAL_1 = 1      # Long goal 1 (y=48)
    SCORE_IN_LONG_GOAL_2 = 2      # Long goal 2 (y=-48)
    SCORE_IN_CENTER_UPPER = 3     # Center goal upper
    SCORE_IN_CENTER_LOWER = 4     # Center goal lower
    TAKE_FROM_LOADER_TL = 5       # Top Left loader
    TAKE_FROM_LOADER_TR = 6       # Top Right loader
    TAKE_FROM_LOADER_BL = 7       # Bottom Left loader
    TAKE_FROM_LOADER_BR = 8       # Bottom Right loader
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
    
    
    def __init__(self, robots: list = None):
        super().__init__(robots)
        self.goal_manager = GoalManager()
        self._agents: Optional[List[str]] = None
    
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
    def get_game(game_name: str) -> VexGame:
        """
        Factory method to create a game instance from a string identifier.
        
        Args:
            game_name: String identifier (e.g., 'vexu_skills', 'vexai_comp')
            
        Returns:
            VexGame instance
        """
        game_class = _get_game_class(game_name)
        return game_class()
    
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
        """Create initial game state."""
        # Initialize agents from Robot objects
        agents_dict = {}
        for robot in self.robots:
            agents_dict[robot.name] = {
                "position": robot.start_position.copy().astype(np.float32),
                "orientation": np.array([robot.start_orientation], dtype=np.float32),
                "team": robot.team.value,
                "robot_size": robot.size.value,
                "held_blocks": 0,
                "parked": False,
                "gameTime": 0.0,
                "active": True,
                "agent_name": robot.name,
                # Per-agent tracking for observation
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
        
        return {
            "agents": agents_dict,
            "blocks": blocks,
            "loaders": self._get_loader_counts(),
        }
    
    def get_observation(self, agent: str, state: Dict) -> np.ndarray:
        """Build observation vector for an agent."""
        agent_state = state["agents"][agent]
        
        # Constants for observation space
        MAX_AVAILABLE_BLOCKS = 20  # Maximum blocks to track positions for
        
        obs_parts = []
        
        # 1. Self position (2) and orientation (1)
        obs_parts.append(float(agent_state["position"][0]))
        obs_parts.append(float(agent_state["position"][1]))
        obs_parts.append(float(agent_state["orientation"][0]))
        
        # 2. Teammate robots only: position (2) + orientation (1) each, max 1 teammate
        MAX_TEAMMATES = 1  # Support up to 2 robots per team
        my_team = agent_state["team"]
        teammate_data = []
        for other_agent, other_state in state["agents"].items():
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
        
        # 3. Self held blocks (1)
        obs_parts.append(float(agent_state["held_blocks"]))
        
        # 4. Self parked status (1)
        obs_parts.append(1.0 if agent_state["parked"] else 0.0)
        
        # 5. Time remaining (1)
        obs_parts.append(float(self.total_time - agent_state["gameTime"]))
        
        # 6. Count and Position of blocks (Split by Friendly/Opponent)
        # MAX 15 per type
        MAX_TRACKED = 15
        
        friendly_blocks = []
        opponent_blocks = []
        
        robot_pos = agent_state["position"]
        my_team = agent_state["team"]
        for block in state["blocks"]:
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
        
        # Add friendly positions
        f_positions = []
        for i in range(min(len(friendly_blocks), MAX_TRACKED)):
            f_positions.extend([friendly_blocks[i][1], friendly_blocks[i][2]])
        # Pad
        while len(f_positions) < MAX_TRACKED * 2:
            f_positions.extend([0.0, 0.0])
        obs_parts.extend(f_positions)
        
        # Add opponent positions
        o_positions = []
        for i in range(min(len(opponent_blocks), MAX_TRACKED)):
            o_positions.extend([opponent_blocks[i][1], opponent_blocks[i][2]])
        # Pad
        while len(o_positions) < MAX_TRACKED * 2:
            o_positions.extend([0.0, 0.0])
        obs_parts.extend(o_positions)
        
        # 8. Blocks added to each goal BY THIS AGENT (4)
        obs_parts.extend([float(x) for x in agent_state["goals_added"]])
        
        # 9. Blocks taken from each loader BY THIS AGENT (4)
        obs_parts.extend([float(x) for x in agent_state["loaders_taken"]])
        
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
        # Held blocks: 1
        # Parked: 1
        # Time remaining: 1
        # Friendly blocks count: 1 (Index 9)
        # Opponent blocks count: 1 (Index 10)
        # Friendly Block positions: 15 * 2 = 30 (Indices 11-40)
        # Opponent Block positions: 15 * 2 = 30 (Indices 41-70)
        # Goals added by this agent: 4
        # Loaders taken by this agent: 4
        # Total: 3 (self) + 3 (team) + 1 (held) + 1 (park) + 1 (time) + 2 (counts) + 30 (friend) + 30 (opp) + 4 (goals) + 4 (loaders) = 79
        # Wait, previous was 58.
        # New calculation: 3+3+1+1+1 = 9.
        # Counts: 2. Total 11.
        # Blocks: 60. Total 71.
        # Goals/Loaders: 8. Total 79.
        # Let's verify counts.
        # Old: 3+3+1+1+1+1(count)+40(blocks)+8 = 58.
        # New: 3+3+1+1+1+2(counts)+30(friend)+30(opp)+8 = 79.
        
        return (79,)
    
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return spaces.Discrete(self.num_actions)

    @staticmethod
    def get_action_space_shape() -> Tuple[int]:
        """Get the shape of the action space (discrete)."""
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
        action: int, 
        state: Dict
    ) -> Tuple[float, float]:
        """Execute an action for an agent."""
        agent_state = state["agents"][agent]
        
        # Parked robots cannot act
        if agent_state.get("parked", False):
            return 0.5, 0.0
        
        # Default values (if action logic falls through or for simple actions)
        duration = 0.5
        penalty = 0.0
        
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            duration, penalty = self._action_pickup_block(agent_state, state)
        
        elif action == Actions.SCORE_IN_LONG_GOAL_1.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_1, state
            )
        elif action == Actions.SCORE_IN_LONG_GOAL_2.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_2, state
            )
        elif action == Actions.SCORE_IN_CENTER_UPPER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_UPPER, state
            )
        elif action == Actions.SCORE_IN_CENTER_LOWER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_LOWER, state
            )
        
        elif action in [
            Actions.TAKE_FROM_LOADER_TL.value,
            Actions.TAKE_FROM_LOADER_TR.value,
            Actions.TAKE_FROM_LOADER_BL.value,
            Actions.TAKE_FROM_LOADER_BR.value,
        ]:
            idx = action - Actions.TAKE_FROM_LOADER_TL.value
            duration, penalty = self._action_take_from_loader(
                agent_state, idx, state
            )
        
        elif action == Actions.PARK.value:
            duration, penalty = self._action_park(agent_state, state)
        
        elif action == Actions.TURN_TOWARD_CENTER.value:
            duration, penalty = self._action_turn_toward_center(agent_state)
        
        elif action == Actions.IDLE.value:
            duration = 0.1
            penalty = DEFAULT_PENALTY # Small penalty for idle
        
        # Update held block positions (game-specific)
        self._update_held_blocks(state)
        
        return duration, penalty
    
    def _action_pickup_block(
        self, agent_state: Dict, state: Dict
    ) -> Tuple[float, float]:
        """Pick up nearest block in FOV matching robot's team."""
        target_idx = -1
        min_dist = float('inf')
        
        robot_pos = agent_state["position"]
        robot_theta = agent_state["orientation"][0]
        robot_team = agent_state["team"]
        
        for i, block in enumerate(state["blocks"]):
            if block["status"] == BlockStatus.ON_FIELD:
                if block.get("team") != robot_team:
                    continue
                
                block_pos = block["position"]
                direction = block_pos - robot_pos
                dist = np.linalg.norm(direction)
                
                if dist > 0:
                    angle_to_block = np.arctan2(direction[1], direction[0])
                    angle_diff = angle_to_block - robot_theta
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    if abs(angle_diff) <= FOV / 2:
                        if dist < min_dist:
                            min_dist = dist
                            target_idx = i
        
        duration = 0.1 # Base duration
        penalty = 0.0
        
        if target_idx != -1:
            target_pos = state["blocks"][target_idx]["position"]
            movement = target_pos - agent_state["position"]
            dist = np.linalg.norm(movement)
            agent_state["position"] = target_pos.copy()
            
            if dist > 0:
                agent_state["orientation"][0] = np.arctan2(movement[1], movement[0])
            
            # Use dynamic robot speed
            speed = self.get_robot_speed(agent_state["agent_name"], state)
            duration += dist / speed
            
            state["blocks"][target_idx]["status"] = BlockStatus.HELD
            state["blocks"][target_idx]["held_by"] = agent_state["agent_name"]
            agent_state["held_blocks"] += 1
        else:
            penalty = DEFAULT_PENALTY
        
        return duration, penalty
    
    def _action_score_in_goal(
        self, 
        agent_state: Dict, 
        goal_type: GoalType, 
        state: Dict
    ) -> Tuple[float, float]:
        """Score held blocks in a goal."""
        if agent_state["held_blocks"] <= 0:
            return 0.5, DEFAULT_PENALTY
        
        goal = self.goal_manager.get_goal(goal_type)
        scoring_side = goal.get_nearest_side(agent_state["position"])
        
        # Calculate robot position based on goal type
        robot_len, _ = self.get_robot_dimensions(agent_state["agent_name"], state)
        
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
        
        movement = robot_pos - agent_state["position"]
        dist = np.linalg.norm(movement)
        agent_state["position"] = robot_pos.astype(np.float32)
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        
        # Dynamic speed
        speed = self.get_robot_speed(agent_state["agent_name"], state)
        duration = 0.1 + (dist / speed)
        
        # Score blocks
        scored_count = 0
        target_status = BlockStatus.get_status_for_goal(goal_type)
        agent_name = agent_state["agent_name"]
        robot_team = agent_state["team"]
        
        for block in state["blocks"]:
            if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                if block.get("team") == robot_team:
                    ejected_id, _ = goal.add_block_from_nearest(id(block), agent_state["position"])
                    block["status"] = target_status
                    block["held_by"] = None
                    scored_count += 1
                    
                    if ejected_id is not None:
                        for b in state["blocks"]:
                            if id(b) == ejected_id:
                                b["status"] = BlockStatus.ON_FIELD
                                if scoring_side == "left":
                                    b["position"] = goal.right_entry.copy()
                                else:
                                    b["position"] = goal.left_entry.copy()
                                break
                else:
                    block["status"] = BlockStatus.ON_FIELD
                    block["held_by"] = None
                    block["position"] = agent_state["position"].copy()
        
        # Update block positions in goal
        self._update_goal_block_positions(goal, target_status, state)
        
        # Track per-agent goals added
        goal_idx = [GoalType.LONG_1, GoalType.LONG_2, GoalType.CENTER_UPPER, GoalType.CENTER_LOWER].index(goal_type)
        agent_state["goals_added"][goal_idx] += scored_count
        
        agent_state["held_blocks"] = 0
        duration += 0.1 * scored_count
        
        return duration, 0.0
    
    def _update_goal_block_positions(
        self, goal: GoalQueue, target_status: int, state: Dict
    ) -> None:
        """Update positions of blocks in a goal."""
        block_positions = goal.get_block_positions(goal.goal_position.center)
        id_to_position = {block_id: pos for block_id, pos in block_positions}
        
        for block in state["blocks"]:
            block_id = id(block)
            if block_id in id_to_position:
                block["position"] = id_to_position[block_id].copy()
                block["status"] = target_status
    
    def _action_take_from_loader(
        self, 
        agent_state: Dict, 
        loader_idx: int, 
        state: Dict
    ) -> Tuple[float, float]:
        """Take a block from a loader."""
        loader_pos = LOADERS[loader_idx].position
        robot_len, _ = self.get_robot_dimensions(agent_state["agent_name"], state)
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
        
        movement = robot_pos - agent_state["position"]
        dist = np.linalg.norm(movement)
        agent_state["position"] = robot_pos.astype(np.float32)
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        # Dynamic speed
        speed = self.get_robot_speed(agent_state["agent_name"], state)
        duration = 0.1 + (dist / speed)
        
        if state["loaders"][loader_idx] > 0:
            state["loaders"][loader_idx] -= 1
            agent_state["held_blocks"] += 1
            duration += 0.2
            
            # Take from Bottom (Last added block for this loader)
            for block in reversed(state["blocks"]):
                if block["status"] == BlockStatus.IN_LOADER_TL + loader_idx:
                    block["status"] = BlockStatus.HELD
                    block["held_by"] = agent_state["agent_name"]
                    block["position"] = agent_state["position"].copy()
                    break
            
            # Track per-agent loaders taken
            agent_state["loaders_taken"][loader_idx] += 1
        
        return duration, 0.0
    
    def _action_park(
        self, agent_state: Dict, state: Dict
    ) -> Tuple[float, float]:
        """Park in team's zone."""
        team = agent_state["team"]
        park_zone = PARK_ZONES[team]

        duration = 0.1
        penalty = 0.0

        # If parking before 10 seconds left, apply penalty
        if agent_state["gameTime"] < self.total_time - 10.0:
            penalty += 1000

        # Only one robot can park per team
        for other_agent, other_state in state["agents"].items():
            if other_agent != agent_state["agent_name"]:
                if other_state["team"] == team and other_state.get("parked", False):
                    return duration, penalty + DEFAULT_PENALTY
        
        movement = park_zone.center - agent_state["position"]
        dist = np.linalg.norm(movement)
        agent_state["position"] = park_zone.center.copy()
        # Dynamic speed     
        speed = self.get_robot_speed(agent_state["agent_name"], {})
        duration += 2.0 + dist / speed
        agent_state["parked"] = True
        agent_state["orientation"] = np.array([np.random.choice([np.pi/2, -np.pi/2])], dtype=np.float32)
        
        return duration, penalty
    
    def _action_turn_toward_center(
        self, agent_state: Dict
    ) -> Tuple[float, float]:
        """Turn robot to face center."""

        direction = np.array([0.0, 0.0]) - agent_state["position"]
        target_angle = np.arctan2(direction[1], direction[0])

        # If already facing center, give a small penalty
        if abs(target_angle - agent_state["orientation"][0]) < 1e-4:
            return 0.1, DEFAULT_PENALTY
        
        agent_state["orientation"] = np.array([target_angle], dtype=np.float32)
        
        return 0.25, 0.0
    
    @abstractmethod
    def compute_score(self, state: Dict) -> Dict[str, int]:
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
    
    def is_agent_terminated(self, agent: str, state: Dict) -> bool:
        """
        Check if an agent has terminated.
        
        In Push Back, an agent terminates when:
        - Their game time exceeds the total time limit, OR
        - They have parked
        """
        agent_state = state["agents"][agent]
        
        # Time limit exceeded
        if agent_state["gameTime"] >= self.total_time:
            return True
        
        # Parked (Push Back specific)
        if agent_state.get("parked", False):
            return True
        
        return False
    
    def is_valid_action(self, action: int, observation: np.ndarray) -> bool:
        """Check if action is valid."""
        if is_scoring_action(Actions(action)):
            # Observation layout: 0-2: self, 3-5: teammate, 6: held_blocks
            held_blocks = observation[6]
            if held_blocks <= 0:
                return False
        
        # Mask Pick Up if no FRIENDLY blocks are actionable
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            # Reconstruct geometric check from observation data
            robot_pos = observation[0:2]
            robot_theta = observation[2]
            
            # New Observation Layout:
            # 0-2: Self (3)
            # 3-5: Teammate (3)
            # 6: Held (1)
            # 7: Parked (1)
            # 8: Time (1)
            # 9: Friendly Count (1)
            # 10: Opponent Count (1)
            # 11..40: Friendly positions (15*2 = 30)
            # 41..70: Opponent positions (15*2 = 30)
            
            friendly_count = int(observation[9])
            
            # Start of friendly blocks
            base_idx = 11
            MAX_TRACKED = 15
            
            found_valid_block = False
            for i in range(min(friendly_count, MAX_TRACKED)):
                bx = observation[base_idx + i*2]
                by = observation[base_idx + i*2 + 1]
                
                # Simple distance check first
                dist = np.linalg.norm([bx - robot_pos[0], by - robot_pos[1]])
                if dist > 0: # Avoid self if bug
                    angle_to_block = np.arctan2(by - robot_pos[1], bx - robot_pos[0])
                    angle_diff = angle_to_block - robot_theta
                    # Normalize
                    while angle_diff > np.pi: angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi: angle_diff += 2 * np.pi
                    
                    if abs(angle_diff) <= FOV / 2:
                        found_valid_block = True
                        break
            
            if not found_valid_block:
                return False
        
        return True
    
    def _update_held_blocks(self, state: Dict) -> None:
        """Update held block positions internally."""
        for agent_name, agent_state in state["agents"].items():
            if agent_state["held_blocks"] > 0:
                for block in state["blocks"]:
                    if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                        block["position"] = agent_state["position"].copy()
    

    
    def get_permanent_obstacles(self) -> List[Obstacle]:
        """Get permanent obstacles."""
        return PERMANENT_OBSTACLES
    
    def render_info_panel(
        self, 
        ax_info: Any, 
        state: Dict, 
        agents: List[str],
        actions: Optional[Dict],
        rewards: Optional[Dict],
        num_moves: int
    ) -> None:
        """
        Render Push Back specific info panel with held blocks by color.
        """
        info_y = 0.95
        ax_info.text(0.5, info_y, "Agent Actions", fontsize=12, fontweight='bold',
                    ha='center', va='top')
        info_y -= 0.08
        
        for i, agent in enumerate(agents):
            st = state["agents"][agent]
            team = self.get_team_for_agent(agent)
            robot_color = 'red' if team == 'red' else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            
            # Count held blocks by color
            held_red = 0
            held_blue = 0
            for block in state["blocks"]:
                if block["status"] == BlockStatus.HELD and block.get("held_by") == agent:
                    if block.get("team") == "red":
                        held_red += 1
                    else:
                        held_blue += 1
            
            # Info panel text
            action_text = "---"
            if actions and agent in actions:
                if st.get("action_skipped", False):
                    action_text = "--"
                else:
                    try:
                        action_text = self.action_to_name(actions[agent])
                    except Exception:
                        action_text = str(actions[agent])
            
            reward_text = ""
            if rewards and agent in rewards:
                reward_text = f" (R: {rewards[agent]:.2f})"
            
            ax_info.text(0.05, info_y, f"Robot {i} ({team}):",
                        fontsize=9, color=robot_color, fontweight='bold', va='top')
            info_y -= 0.05
            ax_info.text(0.1, info_y, f"{action_text}{reward_text}", fontsize=8, va='top')
            info_y -= 0.03
            ax_info.text(0.1, info_y, 
                        f"Time: {st['gameTime']:.1f}s / {self.total_time:.0f}s",
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
        
        team_scores = self.compute_score(state)
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
        ax_info.text(0.05, info_y, f"Step: {num_moves}", fontsize=8, va='top')
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def render_game_elements(self, ax: Any, state: Dict) -> None:
        """Render Push Back game elements."""
        import matplotlib.patches as patches
        import matplotlib.transforms as mtransforms
        
        # Park Zones
        rect_park_red = patches.Rectangle(
            (-72, -12), 18, 24,
            linewidth=1, edgecolor='red', facecolor='none', hatch='//'
        )
        ax.add_patch(rect_park_red)
        
        rect_park_blue = patches.Rectangle(
            (54, -12), 18, 24,
            linewidth=1, edgecolor='blue', facecolor='none', hatch='//'
        )
        ax.add_patch(rect_park_blue)
        
        # Long Goal 1 (top)
        rect_lg_1 = patches.Rectangle(
            (-24, 46), 48, 4, 
            facecolor='orange', alpha=0.3, edgecolor='orange'
        )
        ax.add_patch(rect_lg_1)
        ax.text(0, 52, 'Long 1', fontsize=8, ha='center', va='bottom', color='orange')
        ax.plot([-6, -6], [46, 50], color='white', linewidth=2, linestyle='--')
        ax.plot([6, 6], [46, 50], color='white', linewidth=2, linestyle='--')
        
        # Long Goal 2 (bottom)
        rect_lg_2 = patches.Rectangle(
            (-24, -50), 48, 4, 
            facecolor='orange', alpha=0.3, edgecolor='orange'
        )
        ax.add_patch(rect_lg_2)
        ax.text(0, -54, 'Long 2', fontsize=8, ha='center', va='top', color='orange')
        ax.plot([-6, -6], [-50, -46], color='white', linewidth=2, linestyle='--')
        ax.plot([6, 6], [-50, -46], color='white', linewidth=2, linestyle='--')
        
        # Center Structure (X shape)
        w, h = 24, 4
        rect_center_upper = patches.Rectangle(
            (-w/2, -h/2), w, h, 
            facecolor='green', alpha=0.4, edgecolor='green',
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, -45) + ax.transData
        )
        ax.add_patch(rect_center_upper)
        ax.text(-10, 10, 'Upper', fontsize=6, ha='center', va='center', color='green')
        
        rect_center_lower = patches.Rectangle(
            (-w/2, -h/2), w, h, 
            facecolor='purple', alpha=0.4, edgecolor='purple',
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, 45) + ax.transData
        )
        ax.add_patch(rect_center_lower)
        ax.text(-10, -10, 'Lower', fontsize=6, ha='center', va='center', color='purple')
        
        # Loaders
        for idx, loader in enumerate(LOADERS):
            circle = patches.Circle(
                (loader.position[0], loader.position[1]),
                6.0, fill=False, edgecolor='orange', linewidth=4
            )
            ax.add_patch(circle)
            
            # Find actual blocks in this loader
            loader_status = BlockStatus.IN_LOADER_TL + idx
            loader_blocks = [b for b in state["blocks"] if b["status"] == loader_status]
            
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
                    
                    hexagon = patches.RegularPolygon(
                        (loader.position[0], block_y),
                        numVertices=6, radius=1.8, orientation=0,
                        facecolor=block_color, edgecolor='black', linewidth=1
                    )
                    ax.add_patch(hexagon)
        
        # Obstacles
        for obs in PERMANENT_OBSTACLES:
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius,
                fill=False, edgecolor='black', linestyle=':', linewidth=2
            )
            ax.add_patch(circle)
        
        # Blocks (not in loaders)
        for block in state["blocks"]:
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
                
                hexagon = patches.RegularPolygon(
                    (block["position"][0], block["position"][1]),
                    numVertices=6, radius=2.4, orientation=0,
                    facecolor=fill_color, edgecolor=edge_color, linewidth=edge_width
                )
                ax.add_patch(hexagon)
        
        # Robot FOV wedges
        for agent_name, agent_state in state["agents"].items():
            x, y = agent_state["position"]
            theta = agent_state["orientation"][0]
            
            fov_radius = 72
            fov_start_angle = np.degrees(theta - FOV/2)
            fov_end_angle = np.degrees(theta + FOV/2)
            fov_wedge = patches.Wedge(
                (x, y), fov_radius, fov_start_angle, fov_end_angle,
                facecolor='yellow', alpha=0.15, edgecolor='yellow', linewidth=0.5
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

    @staticmethod
    def split_action(action: int, observation: np.ndarray) -> List[str]:
        """
        Split a high-level action into low-level robot instructions.

        Possible actions:
        INTAKE;speed
        FOLLOW;(x1,y1),(x2,y2),...;speed
        WAIT;time
        DRIVE;inches;speed
        TURN;degrees;speed
        TURN_TO;(x,y);speed
        """
        # Initialize path planner (can be optimized to be a class member)
        path_planner = PathPlanner(15, 15, 2, 70, 70)
        
        actions = []
        
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[0.5,0.5], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append("INTAKE;100")
            actions.append(f"FOLLOW;{points_str};50")
            actions.append("WAIT;0.5")
            actions.append("INTAKE;0")

        elif action == Actions.SCORE_IN_LONG_GOAL_1.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, 1.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};60")
            actions.append("TURN;30;40")
            actions.append("DRIVE;6;40")
            actions.append("INTAKE;0")

        elif action == Actions.SCORE_IN_LONG_GOAL_2.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, 1.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};60")
            actions.append("TURN;-30;40")
            actions.append("DRIVE;6;40")
            actions.append("INTAKE;0")

        elif action == Actions.SCORE_IN_CENTER_UPPER.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[1.5, 1.5], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};55")
            actions.append("TURN;45;40")
            actions.append("DRIVE;4;40")
            actions.append("INTAKE;0")

        elif action == Actions.SCORE_IN_CENTER_LOWER.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-1.5, 1.5], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};55")
            actions.append("TURN;-45;40")
            actions.append("DRIVE;4;40")
            actions.append("INTAKE;0")

        elif action == Actions.TAKE_FROM_LOADER_TL.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, 4.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};50")
            actions.append("TURN;90;40")
            actions.append("DRIVE;1;30")
            actions.append("INTAKE;100")
            actions.append("WAIT;0.5")
            actions.append("INTAKE;0")

        elif action == Actions.TAKE_FROM_LOADER_TR.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, 4.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};50")
            actions.append("TURN;-90;40")
            actions.append("DRIVE;1;30")
            actions.append("INTAKE;100")
            actions.append("WAIT;0.5")
            actions.append("INTAKE;0")

        elif action == Actions.TAKE_FROM_LOADER_BL.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[-3.0, -4.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};50")
            actions.append("TURN;90;40")
            actions.append("DRIVE;1;30")
            actions.append("INTAKE;100")
            actions.append("WAIT;0.5")
            actions.append("INTAKE;0")

        elif action == Actions.TAKE_FROM_LOADER_BR.value:
            positions, velocities, dt = path_planner.Solve(start_point=[0,0], end_point=[3.0, -4.0], obstacles=[])
            points_str = ",".join([f"({pos[0]:.3f}, {pos[1]:.3f})" for pos in positions])

            actions.append(f"FOLLOW;{points_str};50")
            actions.append("TURN;-90;40")
            actions.append("DRIVE;1;30")
            actions.append("INTAKE;100")
            actions.append("WAIT;0.5")
            actions.append("INTAKE;0")

        return actions
