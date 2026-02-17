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
    # Camera rotation is interpreted as an offset (radians) relative to the robot body orientation.
    # Positive values rotate the camera counter-clockwise relative to the robot heading.
    camera_rotation: Optional[float] = np.pi / 2  
    
    def __post_init__(self):
        # Default dimensions based on size
        if self.length is None:
            self.length = float(self.size.value)
        if self.width is None:
            self.width = float(self.size.value)
        # Default orientation: face toward center (red=0, blue=π)
        if self.start_orientation is None:
            self.start_orientation = np.float32(0.0) if self.team == Team.RED else np.float32(np.pi)
        # Camera rotation is stored as an offset from the robot body orientation.
        try:
            self.camera_rotation_offset = float(self.camera_rotation)
        except Exception:
            self.camera_rotation_offset = 0.0


class VexGame(ABC):
    """
    Abstract base class for VEX game implementations.
    
    Each year's game (e.g., Push Back 2024-2025) implements this interface
    to define game-specific mechanics, scoring, and rendering.
    
    State is stored internally in self.state. Each training environment
    should create its own game instance.
    """
    
    def __init__(self, robots: list[Robot] = None, enable_communication: bool = False):
        """Initialize with robot configurations."""
        from path_planner import PathPlanner  # Lazy import to avoid circular dependency
        self.robots = robots or []
        self._robot_map = {r.name: r for r in self.robots}
        self.path_planner = PathPlanner()
        self.enable_communication = enable_communication
        self.state: Dict = None  # Game state, initialized by get_initial_state()
    
    @property
    def agents(self) -> Dict:
        """Shortcut to access agents dict from state."""
        return self.state["agents"] if self.state else {}
    
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
        Create the initial game state and store it in self.state.
        
        Args:
            randomize: If True, randomize initial positions for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - 'agents': Dict of agent states
            - 'blocks': List of block dictionaries
            - 'loaders': List of loader block counts
            - Any other game-specific state
            
        Note:
            Implementations should store the state in self.state before returning.
        """
        pass
    
    @abstractmethod
    def get_observation(self, agent: str, game_time: float = 0.0) -> np.ndarray:
        """
        Build the observation vector for an agent.
        
        Args:
            agent: Agent name
            game_time: Current game time for the agent
            
        Returns:
            Observation array for the agent
        """
        pass

    @abstractmethod
    def is_agent_terminated(self, agent: str, game_time: float = 0.0) -> bool:
        """
        Check if an agent has terminated (game-specific logic).
        
        Args:
            agent: Agent name
            game_time: Current game time for the agent
            
        Returns:
            True if the agent is terminated and should stop acting
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
        action: int
    ) -> Tuple[float, float]:
        """
        Execute an action for an agent.
        
        Args:
            agent: Agent name
            action: Action index
            
        Returns:
            Tuple of (duration in seconds, penalty value)
        """
        pass
    
    @abstractmethod
    def update_tracker(self, agent: str, action: int) -> None:
        """
        Update agent tracker fields based on action.
        
        Called by execute_action() in training and directly in inference.
        Updates: held_blocks, loaders_taken, goals_added.
        
        Args:
            agent: Agent name
            action: Action index
        """
        pass
    
    @abstractmethod
    def update_observation_from_tracker(self, agent: str, observation: np.ndarray) -> np.ndarray:
        """
        Update observation array with tracker fields from game state.
        
        Used during inference to merge external observation data (position, blocks from camera)
        with internal tracker fields (held_blocks, goals_added, loaders_cleared, parked).
        
        Args:
            agent: Agent name
            observation: Observation array to update (modified in-place and returned)
            
        Returns:
            Updated observation array
        """
        pass
    
    @abstractmethod
    def compute_score(self) -> Dict[str, int]:
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
    def is_agent_terminated(self, agent: str, game_time: float = 0.0) -> bool:
        """
        Check if an agent has terminated (game-specific logic).
        
        Args:
            agent: Agent name
            game_time: Current game time for the agent
            
        Returns:
            True if the agent is terminated and should stop acting
        """
        pass
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    @abstractmethod
    def render_game_elements(self, ax: Any) -> None:
        """
        Render game-specific field elements (goals, blocks, etc.).
        
        Args:
            ax: Matplotlib axes object
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
        agents: List[str] = None,
        actions: Optional[Dict] = None,
        rewards: Optional[Dict] = None,
        num_moves: int = 0,
        agent_times: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Render game-specific info panel.
        
        Args:
            ax_info: Matplotlib axes for info panel
            agents: List of active agent names
            actions: Dict of actions taken (or None)
            rewards: Dict of rewards received (or None)
            num_moves: Current step number
            agent_times: Dict mapping agent names to current game time
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

    def get_robot_dimensions(self, agent: str) -> tuple:
        """Get robot dimensions for an agent."""
        robot = self.get_robot_for_agent(agent)
        if robot:
            return (robot.length, robot.width)
        return (18.0, 18.0)
    
    def get_robot_speed(self, agent: str) -> float:
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
    
    def check_robot_collision(self, agent: str) -> bool:
        """
        Check if an agent has collided with another robot.
        
        Override this to implement custom collision detection logic.
        
        Args:
            agent: Agent name to check for collisions
            
        Returns:
            True if agent is colliding with another robot, False otherwise
        """
        if not self.state or "agents" not in self.state:
            return False
        
        agent_state = self.state["agents"].get(agent)
        if not agent_state:
            return False
        
        # Prefer a projection (when present) — fall back to actual `position`.
        agent_pos = agent_state.get("projected_position", agent_state.get("position"))
        if agent_pos is None:
            return False
        
        agent_robot = self.get_robot_for_agent(agent)
        if not agent_robot:
            return False
        
        # Use robot size as collision radius
        agent_size = agent_robot.size.value / 2.0  # Convert to radius
        
        # Check against all other agents (use their projected_position when available)
        for other_agent, other_state in self.state["agents"].items():
            if other_agent == agent:
                continue
            
            other_pos = other_state.get("projected_position", other_state.get("position"))
            if other_pos is None:
                continue
            
            other_robot = self.get_robot_for_agent(other_agent)
            if not other_robot:
                continue
            
            other_size = other_robot.size.value / 2.0  # Convert to radius
            
            # Calculate distance between robot projections (or positions)
            distance = np.linalg.norm(agent_pos - other_pos)
            
            # Collision if distance is less than sum of radii
            min_distance = agent_size + other_size
            
            if distance < min_distance:
                return True
        
        return False
    
    def get_collision_penalty(self) -> float:
        """
        Get the collision penalty value.
        
        Override this to customize the penalty for collisions.
        
        Returns:
            Penalty value (default: 5.0)
        """
        return 10.0
