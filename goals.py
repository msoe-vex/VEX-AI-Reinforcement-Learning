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
from typing import Optional, List, Tuple
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
    Represents a goal with queue-based block storage.
    
    Blocks are stored in a list representing positions from left to right.
    When a block is added from the left, it's inserted at the beginning.
    When a block is added from the right, it's appended to the end.
    If the goal is full, the block on the opposite end is ejected.
    """
    
    def __init__(self, goal_type: GoalType, capacity: int):
        """
        Initialize a goal queue.
        
        Args:
            goal_type: Type of goal (LONG_TOP, LONG_BOTTOM, CENTER)
            capacity: Maximum number of blocks the goal can hold
        """
        self.goal_type = goal_type
        self.capacity = capacity
        self.blocks: List[int] = []  # List of block IDs
        self.goal_position = GOALS[goal_type]
        
    @property
    def count(self) -> int:
        """Number of blocks currently in the goal."""
        return len(self.blocks)
    
    @property
    def is_full(self) -> bool:
        """Check if the goal is at capacity."""
        return len(self.blocks) >= self.capacity
    
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
            # Add to left side (beginning of list)
            if self.is_full:
                ejected = self.blocks.pop()  # Eject from right
            self.blocks.insert(0, block_id)
        else:  # right
            # Add to right side (end of list)
            if self.is_full:
                ejected = self.blocks.pop(0)  # Eject from left
            self.blocks.append(block_id)
            
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
        if block_id in self.blocks:
            self.blocks.remove(block_id)
            return True
        return False
    
    def remove_from_side(self, side: str) -> Optional[int]:
        """
        Remove a block from the specified side (for descoring).
        
        Args:
            side: 'left' or 'right'
            
        Returns:
            ID of removed block or None if empty
        """
        if not self.blocks:
            return None
            
        if side == "left":
            return self.blocks.pop(0)
        else:
            return self.blocks.pop()
    
    def get_block_positions(self, goal_center: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Calculate display positions for all blocks in the goal.
        
        Args:
            goal_center: Center position of the goal
            
        Returns:
            List of (block_id, position) tuples
        """
        if not self.blocks:
            return []
        
        # Spread blocks evenly across the goal
        positions = []
        goal_width = np.linalg.norm(self.right_entry - self.left_entry)
        spacing = goal_width / (self.capacity + 1)
        
        for i, block_id in enumerate(self.blocks):
            # Calculate position from left to right
            t = (i + 1) / (len(self.blocks) + 1)
            pos = self.left_entry + t * (self.right_entry - self.left_entry)
            positions.append((block_id, pos))
            
        return positions
    
    def clear(self) -> List[int]:
        """Clear all blocks from the goal. Returns list of ejected block IDs."""
        ejected = self.blocks.copy()
        self.blocks = []
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

