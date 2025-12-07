"""
VEX Push Back - Goal Queue Mechanics

Implements goal structures with queue-based block storage:
- Blocks added from nearest end
- Overflow ejects blocks from opposite end
- Blocks shift as they're added
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, List, Tuple, Dict
from enum import Enum

try:
    from .field import GoalType, GOALS, LONG_GOAL_CAPACITY, CENTER_GOAL_CAPACITY
except ImportError:
    from field import GoalType, GOALS, LONG_GOAL_CAPACITY, CENTER_GOAL_CAPACITY


@dataclass
class Block:
    """Represents a single block in the game."""
    id: int
    position: np.ndarray
    status: int  # 0=field, 1=held, 2-4=scored, 5-8=loader
    team: Optional[str] = None  # Which team this block belongs to (for colored blocks)
    
    def copy(self) -> 'Block':
        """Create a copy of this block."""
        return Block(
            id=self.id,
            position=self.position.copy(),
            status=self.status,
            team=self.team
        )


class GoalQueue:
    """
    Queue-based goal that manages blocks with FIFO behavior from both sides.
    
    Uses a fixed-size array with None for empty slots. Adding from left fills
    from index 0 rightward, adding from right fills from the end leftward.
    """
    
    def __init__(self, goal_type: GoalType, capacity: int):
        """
        Initialize a goal queue.
        
        Args:
            goal_type: Type of goal (LONG_1, LONG_2, CENTER_UPPER, CENTER_LOWER)
            capacity: Maximum number of blocks the goal can hold
        """
        self.goal_type = goal_type
        self.capacity = capacity
        self.slots: List[Optional[int]] = [None] * capacity  # Fixed-size array
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
        """Get the left entry point."""
        return self.goal_position.left_entry
    
    @property
    def right_entry(self) -> np.ndarray:
        """Get the right entry point."""
        return self.goal_position.right_entry
        
    def get_nearest_entry(self, robot_position: np.ndarray) -> np.ndarray:
        """Get the entry point nearest to the robot."""
        return self.goal_position.get_nearest_entry(robot_position)
    
    def get_nearest_side(self, robot_position: np.ndarray) -> str:
        """Get which side ('left' or 'right') is nearest to the robot."""
        return self.goal_position.get_nearest_entry_side(robot_position)
    
    def add_block(self, block_id: int, from_side: str) -> Optional[int]:
        """
        Add a block to the goal from the specified side.
        
        Args:
            block_id: ID of the block to add
            from_side: 'left' or 'right'
            
        Returns:
            ID of ejected block if overflow occurred, None otherwise
        """
        ejected = None
        
        if from_side == "left":
            # Add from left - find first empty slot from left
            for i in range(self.capacity):
                if self.slots[i] is None:
                    self.slots[i] = block_id
                    return None
            # Full - eject from right, shift left, add at left
            ejected = self.slots[-1]
            self.slots = [block_id] + self.slots[:-1]
        else:  # right
            # Add from right - find first empty slot from right
            for i in range(self.capacity - 1, -1, -1):
                if self.slots[i] is None:
                    self.slots[i] = block_id
                    return None
            # Full - eject from left, shift right, add at right
            ejected = self.slots[0]
            self.slots = self.slots[1:] + [block_id]
            
        return ejected
    
    def add_block_from_nearest(self, block_id: int, robot_position: np.ndarray) -> Tuple[Optional[int], str]:
        """
        Add a block from the side nearest to the robot.
        
        Args:
            block_id: ID of the block to add
            robot_position: Current position of the robot
            
        Returns:
            Tuple of (ejected block ID or None, side used)
        """
        side = self.get_nearest_side(robot_position)
        ejected = self.add_block(block_id, side)
        return ejected, side
    
    def remove_block(self, block_id: int) -> bool:
        """
        Remove a specific block from the goal.
        
        Args:
            block_id: ID of the block to remove
            
        Returns:
            True if block was found and removed, False otherwise
        """
        for i in range(self.capacity):
            if self.slots[i] == block_id:
                self.slots[i] = None
                return True
        return False
    
    def remove_from_side(self, side: str) -> Optional[int]:
        """
        Remove a block from the specified side (for descoring).
        
        Args:
            side: 'left' or 'right'
            
        Returns:
            ID of removed block or None if empty on that side
        """
        if side == "left":
            for i in range(self.capacity):
                if self.slots[i] is not None:
                    block_id = self.slots[i]
                    self.slots[i] = None
                    return block_id
        else:
            for i in range(self.capacity - 1, -1, -1):
                if self.slots[i] is not None:
                    block_id = self.slots[i]
                    self.slots[i] = None
                    return block_id
        return None
    
    def get_block_positions(self, goal_center: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Calculate display positions for all blocks in the goal.
        
        Each slot has a fixed position. Only returns positions for non-empty slots.
        
        Args:
            goal_center: Center position of the goal (unused, kept for compatibility)
            
        Returns:
            List of (block_id, position) tuples
        """
        positions = []
        direction = self.right_entry - self.left_entry
        
        for i, block_id in enumerate(self.slots):
            if block_id is not None:
                # Position at center of slot
                t = (i + 0.5) / self.capacity
                pos = self.left_entry + t * direction
                positions.append((block_id, pos.copy()))
            
        return positions
    
    def clear(self) -> List[int]:
        """Clear all blocks from the goal. Returns list of ejected block IDs."""
        ejected = [b for b in self.slots if b is not None]
        self.slots = [None] * self.capacity
        return ejected


class GoalManager:
    """Manages all goals on the field."""
    
    def __init__(self):
        """Initialize goal manager with all field goals."""
        self.goals = {
            GoalType.LONG_1: GoalQueue(GoalType.LONG_1, LONG_GOAL_CAPACITY),
            GoalType.LONG_2: GoalQueue(GoalType.LONG_2, LONG_GOAL_CAPACITY),
            GoalType.CENTER_UPPER: GoalQueue(GoalType.CENTER_UPPER, CENTER_GOAL_CAPACITY),
            GoalType.CENTER_LOWER: GoalQueue(GoalType.CENTER_LOWER, CENTER_GOAL_CAPACITY),
        }
    
    def get_goal(self, goal_type: GoalType) -> GoalQueue:
        """Get a specific goal by type."""
        return self.goals[goal_type]
    
    def score_block(self, goal_type: GoalType, block_id: int, 
                    robot_position: np.ndarray) -> Tuple[Optional[int], str]:
        """
        Score a block in a goal from the nearest side.
        
        Args:
            goal_type: Type of goal to score in
            block_id: ID of block to score
            robot_position: Position of the scoring robot
            
        Returns:
            Tuple of (ejected block ID or None, side scored from)
        """
        goal = self.goals[goal_type]
        return goal.add_block_from_nearest(block_id, robot_position)
    
    def get_goal_counts(self) -> dict:
        """Get block counts for all goals."""
        return {
            goal_type: goal.count 
            for goal_type, goal in self.goals.items()
        }
    
    def reset(self):
        """Reset all goals to empty state."""
        for goal in self.goals.values():
            goal.clear()


# Status codes for block locations
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

