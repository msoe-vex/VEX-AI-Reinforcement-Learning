"""
VEX Core - Abstract Game Interface

Defines the interface that each year's VEX game must implement.
The VexMultiAgentEnv delegates game-specific logic to a VexGame instance.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from gymnasium import spaces


class VexGame(ABC):
    """
    Abstract base class for VEX game implementations.
    
    Each year's game (e.g., Push Back 2024-2025) implements this interface
    to define game-specific mechanics, scoring, and rendering.
    """
    
    # =========================================================================
    # Configuration Properties
    # =========================================================================
    
    @property
    @abstractmethod
    def field_size_inches(self) -> float:
        """Field size in inches (typically 144 for 12ft x 12ft field)."""
        pass
    
    @property
    @abstractmethod
    def total_time(self) -> float:
        """Total match time in seconds."""
        pass
    
    @property
    @abstractmethod
    def possible_agents(self) -> List[str]:
        """List of agent names for this game mode."""
        pass
    
    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of possible actions in the action space."""
        pass
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    @abstractmethod
    def get_initial_state(self, randomize: bool = False, seed: Optional[int] = None) -> Dict:
        """
        Create the initial game state.
        
        Args:
            randomize: If True, randomize initial positions for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - 'agents': Dict of agent states
            - 'blocks': List of block dictionaries
            - 'loaders': List of loader block counts
            - Any other game-specific state
        """
        pass
    
    @abstractmethod
    def get_observation(self, agent: str, state: Dict) -> np.ndarray:
        """
        Build the observation vector for an agent.
        
        Args:
            agent: Agent name
            state: Current game state
            
        Returns:
            Observation array for the agent
        """
        pass
    
    @abstractmethod
    def observation_space(self, agent: str) -> spaces.Space:
        """Get the observation space for an agent."""
        pass
    
    @abstractmethod
    def action_space(self, agent: str) -> spaces.Space:
        """Get the action space for an agent."""
        pass
    
    # =========================================================================
    # Game Logic
    # =========================================================================
    
    @abstractmethod
    def execute_action(
        self, 
        agent: str, 
        action: int, 
        state: Dict
    ) -> Tuple[float, float]:
        """
        Execute an action for an agent.
        
        Args:
            agent: Agent name
            action: Action index
            state: Current game state (will be modified in-place)
            
        Returns:
            Tuple of (duration in seconds, penalty value)
        """
        pass
    
    @abstractmethod
    def compute_score(self, state: Dict) -> int:
        """
        Compute the total score for the current state.
        
        Args:
            state: Current game state
            
        Returns:
            Total score
        """
        pass
    
    @abstractmethod
    def compute_team_scores(self, state: Dict) -> Dict[str, int]:
        """
        Compute scores per team.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping team names to scores
        """
        pass
    
    @abstractmethod
    def get_team_for_agent(self, agent: str) -> str:
        """
        Get the team name for an agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Team name ('red' or 'blue')
        """
        pass
    
    @abstractmethod
    def is_agent_terminated(self, agent: str, state: Dict) -> bool:
        """
        Check if an agent has terminated (game-specific logic).
        
        Args:
            agent: Agent name
            state: Current game state
            
        Returns:
            True if the agent is terminated and should stop acting
        """
        pass
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    @abstractmethod
    def render_game_elements(self, ax: Any, state: Dict) -> None:
        """
        Render game-specific field elements (goals, blocks, etc.).
        
        Args:
            ax: Matplotlib axes object
            state: Current game state
        """
        pass
    
    @abstractmethod
    def get_permanent_obstacles(self) -> List[Any]:
        """
        Get list of permanent obstacles for path planning.
        
        Returns:
            List of obstacle objects
        """
        pass
    
    # =========================================================================
    # Optional Methods (with default implementations)
    # =========================================================================
    
    def is_valid_action(self, action: int, observation: np.ndarray) -> bool:
        """
        Check if an action is valid in the current state.
        
        Override this to implement action masking.
        
        Args:
            action: Action index
            observation: Current observation
            
        Returns:
            True if action is valid
        """
        return True
    
    def get_robot_dimensions(self, agent: str, state: Dict) -> Tuple[float, float]:
        """
        Get robot dimensions for an agent.
        
        Args:
            agent: Agent name
            state: Current game state
            
        Returns:
            Tuple of (length, width) in inches
        """
        return (18.0, 18.0)  # Default 18x18 inch robot
    
    def reset(self) -> None:
        """
        Reset any internal game state.
        
        Override if the game has internal state that needs resetting
        (e.g., goal managers).
        """
        pass
