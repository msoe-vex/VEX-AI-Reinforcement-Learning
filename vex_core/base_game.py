"""
VEX Core - Abstract Game Interface

Defines the interface that each year's VEX game must implement.
The VexMultiAgentEnv delegates game-specific logic to a VexGame instance.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from enum import Enum

class RobotSize(Enum):
    """Robot size categories."""
    INCH_15 = 15
    INCH_24 = 24

class Team(Enum):
    """Robot teams."""
    RED = "red"
    BLUE = "blue"

@dataclass
class Robot:
    """Robot configuration."""
    name: str  # Agent name, e.g., "red_robot_0"
    team: Team  # 'red' or 'blue'
    size: RobotSize
    start_position: Optional[np.ndarray] = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float32))
    start_orientation: Optional[float] = None  # Radians, None = auto based on team
    length: Optional[float] = None
    width: Optional[float] = None
    max_speed: Optional[float] = 85.0
    max_acceleration: Optional[float] = 85.0
    buffer: Optional[float] = 1.0
    
    def __post_init__(self):
        # Default dimensions based on size
        if self.length is None:
            self.length = float(self.size.value)
        if self.width is None:
            self.width = float(self.size.value)
        # Default orientation: face toward center (red=0, blue=π)
        if self.start_orientation is None:
            self.start_orientation = np.float32(0.0) if self.team == Team.RED else np.float32(np.pi)


class VexGame(ABC):
    """
    Abstract base class for VEX game implementations.
    
    Each year's game (e.g., Push Back 2024-2025) implements this interface
    to define game-specific mechanics, scoring, and rendering.
    """
    
    def __init__(self, robots: list[Robot] = None):
        """Initialize with robot configurations."""
        from path_planner import PathPlanner  # Lazy import to avoid circular dependency
        self.robots = robots or []
        self._robot_map = {r.name: r for r in self.robots}
        self.path_planner = PathPlanner()
    
    @staticmethod
    @abstractmethod
    def get_game(game_name: str) -> 'VexGame':
        """
        Factory method to create a game instance from a string identifier.
        
        Args:
            game_name: String identifier for the game variant
            
        Returns:
            VexGame instance
        """
        pass
    
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
    def possible_agents(self) -> List[str]:
        """List of agent names derived from robots."""
        return [r.name for r in self.robots]
    
    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of possible actions in the action space."""
        pass
    
    @abstractmethod
    def fallback_action(self) -> int:
        """Default action to take if no valid action is available."""
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
    def compute_score(self, state: Dict) -> Dict[str, int]:
        """Compute the score for the current state.
        Returns:
            Dict[str, int]: Team scores (e.g., {"red": 10, "blue": 5})
        """
        pass
    
    def compute_reward(
        self, 
        agent: str, 
        initial_scores: Dict[str, int], 
        new_scores: Dict[str, int],
        penalty: float
    ) -> float:
        """
        Compute reward for an agent based on score changes.
        
        Override this for game modes where the reward logic differs from
        standard team-based scoring (e.g., skills where all robots
        contribute to the same score regardless of their team).
        
        Args:
            agent: Agent name
            initial_scores: Scores before action
            new_scores: Scores after action
            penalty: Penalty from action execution
            
        Returns:
            Reward value for the agent
        """
        # Default: competitive scoring - my delta minus opponent delta
        agent_team = self.get_team_for_agent(agent)
        opposing_team = "blue" if agent_team == "red" else "red"
        
        own_delta = new_scores.get(agent_team, 0) - initial_scores.get(agent_team, 0)
        opp_delta = new_scores.get(opposing_team, 0) - initial_scores.get(opposing_team, 0)
        
        return own_delta - opp_delta - penalty

    
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
    def split_action(self, action: int, observation: np.ndarray, robot: Robot) -> List[str]:
        """Convert a high-level action into a list of low-level commands."""
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

    @abstractmethod
    def action_to_name(self, action: int) -> str:
        """
        Convert an action index to a human-readable name.
        
        Args:
            action: Action index
            
        Returns:
            Human-readable action name
        """
        pass
    
    @abstractmethod
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
        Render game-specific info panel.
        
        Args:
            ax_info: Matplotlib axes for info panel
            state: Current game state
            agents: List of active agent names
            actions: Dict of actions taken (or None)
            rewards: Dict of rewards received (or None)
            num_moves: Current step number
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
    
    def get_robot_for_agent(self, agent: str) -> Robot:
        """Get Robot object for an agent by name."""
        return self._robot_map.get(agent)

    def get_robot_dimensions(self, agent: str, state: dict) -> tuple:
        """Get robot dimensions for an agent."""
        robot = self.get_robot_for_agent(agent)
        if robot:
            return (robot.length, robot.width)
        return (18.0, 18.0)
    
    def get_robot_speed(self, agent: str, state: dict) -> float:
        """Get robot speed for an agent."""
        robot = self.get_robot_for_agent(agent)
        if robot:
            return robot.max_speed
        return 60.0
    
    def reset(self) -> None:
        """
        Reset any internal game state.
        
        Override if the game has internal state that needs resetting
        (e.g., goal managers).
        """
        pass
