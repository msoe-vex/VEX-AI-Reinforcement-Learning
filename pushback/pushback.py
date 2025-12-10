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
from vex_core.base_game import VexGame
from path_planner import PathPlanner

NUM_BLOCKS_FIELD = 36
NUM_BLOCKS_LOADER = 24


# =============================================================================
# Constants
# =============================================================================

FIELD_SIZE_INCHES = 144  # 12 feet = 144 inches
FIELD_HALF = FIELD_SIZE_INCHES / 2  # 72 inches

# Robot dimensions
ROBOT_24_SIZE = 24.0  # 24" robot
ROBOT_15_SIZE = 15.0  # 15" robot
DEFAULT_ROBOT_SIZE = 18.0  # Default/generic size

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

# Robot speed
ROBOT_SPEED = 60.0  # inches per second

# Default penalty for invalid actions
DEFAULT_PENALTY = -0.1


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
    CLEAR_LOADER = 9              # Dispense blocks from nearest loader
    PARK = 10                     # Park in team's zone
    TURN_TOWARD_CENTER = 11       # Turn to face center of field
    IDLE = 12


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
        """Add a block from the specified side. Returns ejected block ID if overflow."""
        ejected = None
        
        if from_side == "left":
            for i in range(self.capacity):
                if self.slots[i] is None:
                    self.slots[i] = block_id
                    return None
            ejected = self.slots[-1]
            self.slots = [block_id] + self.slots[:-1]
        else:
            for i in range(self.capacity - 1, -1, -1):
                if self.slots[i] is None:
                    self.slots[i] = block_id
                    return None
            ejected = self.slots[0]
            self.slots = self.slots[1:] + [block_id]
        
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
# Scoring Configuration
# =============================================================================




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
    
    
    def __init__(self):
        self.goal_manager = GoalManager()
        self._agents: Optional[List[str]] = None
    
    # =========================================================================
    # Abstract methods for variants
    # =========================================================================
    
    @abstractmethod
    def _get_agents_config(self) -> List[str]:
        """Get list of agent names for this variant."""
        pass
    
    @abstractmethod
    def _get_robot_configs(self) -> Dict[str, Tuple[np.ndarray, str, str]]:
        """
        Get robot configurations.
        
        Returns:
            Dict mapping agent_name -> (start_position, robot_size, team)
            robot_size is '24' or '15'
            team is 'red' or 'blue'
        """
        pass
    
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
    
    @property
    @abstractmethod
    def total_time(self) -> float:
        """Total game time in seconds."""
        pass
    
    @property
    def possible_agents(self) -> List[str]:
        if self._agents is None:
            self._agents = self._get_agents_config()
        return self._agents
    
    @property
    def num_actions(self) -> int:
        return len(Actions)
    
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
        robot_configs = self._get_robot_configs()
        
        # Initialize agents
        agents_dict = {}
        for agent_name, (position, robot_size, team) in robot_configs.items():
            agents_dict[agent_name] = {
                "position": position.copy().astype(np.float32),
                "orientation": np.array([0.0], dtype=np.float32),
                "team": team,
                "robot_size": robot_size,
                "held_blocks": 0,  # Will be set based on preloads
                "parked": False,
                "gameTime": 0.0,
                "active": True,
                "agent_name": agent_name,
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
        
        # Block data
        block_data = []
        for b in state["blocks"]:
            block_data.extend([b["position"][0], b["position"][1], float(b["status"])])
        
        # Goal counts
        goal_counts = self.goal_manager.get_goal_counts()
        goal_data = [
            float(goal_counts[GoalType.LONG_1]),
            float(goal_counts[GoalType.LONG_2]),
            float(goal_counts[GoalType.CENTER_UPPER]),
            float(goal_counts[GoalType.CENTER_LOWER]),
        ]
        
        total_blocks = len(state["blocks"])
        obs = np.concatenate([
            agent_state["position"],
            agent_state["orientation"],
            [float(agent_state["held_blocks"])],
            [1.0 if agent_state["parked"] else 0.0],
            [float(self.total_time - agent_state["gameTime"])],
            np.array(block_data, dtype=np.float32),
            np.array(state["loaders"], dtype=np.float32),
            np.array(goal_data, dtype=np.float32),
        ])
        
        return obs.astype(np.float32)
    
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        shape = self.get_observation_space_shape(len(self.possible_agents))
        
        low = np.full(shape, -float('inf'), dtype=np.float32)
        high = np.full(shape, float('inf'), dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    @staticmethod
    def get_observation_space_shape(num_agents: int) -> Tuple[int]:
        """Get the shape of the observation space."""
        # Calculate observation size based on expected blocks
        total_blocks = NUM_BLOCKS_FIELD + NUM_BLOCKS_LOADER + num_agents
        obs_size = 6 + (total_blocks * 3) + 4 + 4  # agent state + blocks + loaders + goals
        return (obs_size,)
    
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
        
        duration = 0.5
        penalty = 0.0
        
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            duration, penalty = self._action_pickup_block(agent_state, state, duration, penalty)
        
        elif action == Actions.SCORE_IN_LONG_GOAL_1.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_1, state, duration, penalty
            )
        elif action == Actions.SCORE_IN_LONG_GOAL_2.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_2, state, duration, penalty
            )
        elif action == Actions.SCORE_IN_CENTER_UPPER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_UPPER, state, duration, penalty
            )
        elif action == Actions.SCORE_IN_CENTER_LOWER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_LOWER, state, duration, penalty
            )
        
        elif action in [
            Actions.TAKE_FROM_LOADER_TL.value,
            Actions.TAKE_FROM_LOADER_TR.value,
            Actions.TAKE_FROM_LOADER_BL.value,
            Actions.TAKE_FROM_LOADER_BR.value,
        ]:
            idx = action - Actions.TAKE_FROM_LOADER_TL.value
            duration, penalty = self._action_take_from_loader(
                agent_state, idx, state, duration, penalty
            )
        
        elif action == Actions.CLEAR_LOADER.value:
            duration, penalty = self._action_clear_loader(agent_state, state, duration, penalty)
        
        elif action == Actions.PARK.value:
            duration, penalty = self._action_park(agent_state, duration, penalty)
        
        elif action == Actions.TURN_TOWARD_CENTER.value:
            duration, penalty = self._action_turn_toward_center(agent_state, duration, penalty)
        
        elif action == Actions.IDLE.value:
            duration = 0.1
        
        # Update held block positions (game-specific)
        self._update_held_blocks(state)
        
        return duration, penalty
    
    def _action_pickup_block(
        self, agent_state: Dict, state: Dict, duration: float, penalty: float
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
        
        if target_idx != -1:
            target_pos = state["blocks"][target_idx]["position"]
            movement = target_pos - agent_state["position"]
            dist_travelled = np.linalg.norm(movement)
            agent_state["position"] = target_pos.copy()
            
            if dist_travelled > 0:
                agent_state["orientation"][0] = np.arctan2(movement[1], movement[0])
            duration += dist_travelled / ROBOT_SPEED
            
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
        state: Dict, 
        duration: float, 
        penalty: float
    ) -> Tuple[float, float]:
        """Score held blocks in a goal."""
        if agent_state["held_blocks"] <= 0:
            return duration, penalty + DEFAULT_PENALTY
        
        goal = self.goal_manager.get_goal(goal_type)
        scoring_side = goal.get_nearest_side(agent_state["position"])
        
        # Calculate robot position based on goal type
        robot_len = ROBOT_24_SIZE if agent_state.get("robot_size") == "24" else ROBOT_15_SIZE
        
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
        duration += dist / ROBOT_SPEED
        
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
        
        agent_state["held_blocks"] = 0
        duration += 0.5 * scored_count
        
        return duration, penalty
    
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
        state: Dict, 
        duration: float, 
        penalty: float
    ) -> Tuple[float, float]:
        """Take a block from a loader."""
        loader_pos = LOADERS[loader_idx].position
        robot_len = ROBOT_24_SIZE if agent_state.get("robot_size") == "24" else ROBOT_15_SIZE
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
        duration += dist / ROBOT_SPEED
        
        if state["loaders"][loader_idx] > 0:
            state["loaders"][loader_idx] -= 1
            agent_state["held_blocks"] += 1
            duration += 0.5
            
            # Take from Bottom (Last added block for this loader)
            for block in reversed(state["blocks"]):
                if block["status"] == BlockStatus.IN_LOADER_TL + loader_idx:
                    block["status"] = BlockStatus.HELD
                    block["held_by"] = agent_state["agent_name"]
                    block["position"] = agent_state["position"].copy()
                    break
        
        return duration, penalty
    
    def _action_clear_loader(
        self, agent_state: Dict, state: Dict, duration: float, penalty: float
    ) -> Tuple[float, float]:
        """Clear blocks from nearest loader."""
        closest_loader = -1
        
        for loader in LOADERS:
            if np.linalg.norm(agent_state["position"] - loader.position) < 18.0:
                closest_loader = loader.index
                break
        
        if closest_loader != -1 and state["loaders"][closest_loader] > 0:
            loader_status = BlockStatus.IN_LOADER_TL + closest_loader
            # Clear from Bottom (Last added)
            for block in reversed(state["blocks"]):
                if block["status"] == loader_status:
                    block["status"] = BlockStatus.ON_FIELD
                    block["position"] = agent_state["position"] + np.random.uniform(-6.0, 6.0, 2).astype(np.float32)
                    state["loaders"][closest_loader] -= 1
                    break
            duration += 1.0
        else:
            penalty = DEFAULT_PENALTY
        
        return duration, penalty
    
    def _action_park(
        self, agent_state: Dict, duration: float, penalty: float
    ) -> Tuple[float, float]:
        """Park in team's zone."""
        team = agent_state["team"]
        park_zone = PARK_ZONES[team]
        
        movement = park_zone.center - agent_state["position"]
        dist = np.linalg.norm(movement)
        agent_state["position"] = park_zone.center.copy()
        duration += dist / ROBOT_SPEED
        agent_state["parked"] = True
        agent_state["orientation"] = np.array([np.random.choice([np.pi/2, -np.pi/2])], dtype=np.float32)
        
        return duration, penalty
    
    def _action_turn_toward_center(
        self, agent_state: Dict, duration: float, penalty: float
    ) -> Tuple[float, float]:
        """Turn robot to face center."""
        direction = np.array([0.0, 0.0]) - agent_state["position"]
        target_angle = np.arctan2(direction[1], direction[0])
        agent_state["orientation"] = np.array([target_angle], dtype=np.float32)
        duration += 0.3
        
        return duration, penalty
    
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
            held_blocks = observation[3]
            if held_blocks <= 0:
                return False
        return True
    
    def _update_held_blocks(self, state: Dict) -> None:
        """Update held block positions internally."""
        for agent_name, agent_state in state["agents"].items():
            if agent_state["held_blocks"] > 0:
                for block in state["blocks"]:
                    if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                        block["position"] = agent_state["position"].copy()
    
    def get_robot_dimensions(self, agent: str, state: Dict) -> Tuple[float, float]:
        """Get robot dimensions."""
        if state and "agents" in state and agent in state["agents"]:
            size = state["agents"][agent].get("robot_size", "18")
            if size == "24":
                return (ROBOT_24_SIZE, ROBOT_24_SIZE)
            elif size == "15":
                return (ROBOT_15_SIZE, ROBOT_15_SIZE)
        return (DEFAULT_ROBOT_SIZE, DEFAULT_ROBOT_SIZE)
    
    def get_permanent_obstacles(self) -> List[Obstacle]:
        """Get permanent obstacles."""
        return PERMANENT_OBSTACLES
    
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

    @staticmethod
    def split_action(action: int, observation: np.ndarray) -> List[str]:
        """
        Split a high-level action into low-level robot instructions.
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
