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

# Default reward decomposition weights
# reward = a * team_delta + b * opp_delta + c * individual_delta + d * individual_penalty + e * team_penalty
DEFAULT_REWARD_WEIGHT_TEAM_DELTA = 0.1 # (Points scored for my team by teammates - Points scored for opp team by teammates)
DEFAULT_REWARD_WEIGHT_OPP_DELTA = -0.0 # (Points scored for opp team by opponents - Points scored for my team by opponents)
DEFAULT_REWARD_WEIGHT_INDIVIDUAL_DELTA = 1.0 # (Points scored for my team by me - Points scored for opp team by me)
DEFAULT_REWARD_WEIGHT_INDIVIDUAL_PENALTY = -0.1 # (Penalty for actions performed by me)
DEFAULT_REWARD_WEIGHT_TEAM_PENALTY = -0.001 # (Penalty for actions performed by teammates)
from vex_core.robot import Robot, Team, RobotSize
from vex_core.config import CommunicationOption


@dataclass
class ActionEvent:
    """A single game-state change that occurs when an action completes.
    
    All stochasticity should be handled by the game logic inside `apply_events`.
    """
    type: str
    data: Dict = field(default_factory=dict)


@dataclass
class ActionStep:
    """A single sub-step in an action's execution timeline.
    
    An action is composed of one or more ActionSteps executed sequentially.
    Each step defines where the robot moves, how long it takes, and what
    game-state changes (events) to apply when the step completes.
    """
    duration: float
    target_pos: np.ndarray
    target_orient: np.ndarray
    events: List[ActionEvent] = field(default_factory=list)


class VexGame(ABC):
    """
    Abstract base class for VEX game implementations.
    
    Each year's game (e.g., Push Back 2024-2025) implements this interface
    to define game-specific mechanics, scoring, and rendering.
    
    State is stored internally in self.state. Each training environment
    should create its own game instance.
    """
    
    def __init__(self, robots: list[Robot] = None, communication_mode: CommunicationOption = CommunicationOption.NONE):
        """Initialize with robot configurations."""
        self.robots = robots or []
        
        # A team may use multiple 15-inch robots, but only one 24-inch robot.
        from vex_core.robot import RobotSize
        team_sizes = {}
        for r in self.robots:
            if r.team not in team_sizes:
                team_sizes[r.team] = set()
            if r.size == RobotSize.INCH_24 and r.size in team_sizes[r.team]:
                raise ValueError(f"Team {r.team} may have only one 24-inch robot.")
            team_sizes[r.team].add(r.size)
            
        self._robot_map = {r.name: r for r in self.robots}
        self.communication_mode = communication_mode
        self.state: Dict = None  # Game state, initialized by get_initial_state()
        self.reward_weight_team_delta = DEFAULT_REWARD_WEIGHT_TEAM_DELTA
        self.reward_weight_opp_delta = DEFAULT_REWARD_WEIGHT_OPP_DELTA
        self.reward_weight_individual_delta = DEFAULT_REWARD_WEIGHT_INDIVIDUAL_DELTA
        self.reward_weight_individual_penalty = DEFAULT_REWARD_WEIGHT_INDIVIDUAL_PENALTY
        self.reward_weight_team_penalty = DEFAULT_REWARD_WEIGHT_TEAM_PENALTY
    
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

    def get_game_name(self) -> str:
        return self.__class__.__name__.lower()
    
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
    
    @property
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
    def get_game_observation(self, agent: str, game_time: float = 0.0) -> spaces.Space:
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
    def get_game_observation_space(self, agent: str) -> spaces.Space:
        """Get the observation space for an agent."""
        pass
    
    @abstractmethod
    def get_game_action_space(self, agent: str) -> spaces.Space:
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
    ) -> Tuple[List[ActionStep], float]:
        """
        Execute an action for an agent.
        
        Returns a list of ActionSteps defining the action's timeline, and a
        penalty value. The environment handles interpolation and applies each
        step's events when the step's ticks are exhausted.
        
        IMPORTANT: This method should NOT mutate agent position/orientation.
        Position changes are handled by the environment via interpolation.
        
        Args:
            agent: Agent name
            action: Action index
            
        Returns:
            Tuple of (list of ActionSteps, penalty value)
        """
        pass
    
    @abstractmethod
    def get_action_name(self, action: int) -> str:
        """
        Get the human-readable string name for an action index.
        
        Args:
            action: Action integer index
            
        Returns:
            String name of the action
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
    
    def combine_reward_components(
        self,
        agent: str,
        team_delta: float,
        opp_delta: float,
        individual_delta: float,
        individual_penalty: float,
        team_penalty: float,
    ) -> float:
        """Combine reward components using configurable weights.

        reward = a * team_delta + b * opp_delta + c * individual_delta
               + d * individual_penalty + e * team_penalty
        """
        a = float(getattr(self, "reward_weight_team_delta", 1.0))
        b = float(getattr(self, "reward_weight_opp_delta", -1.0))
        c = float(getattr(self, "reward_weight_individual_delta", 1.0))
        d = float(getattr(self, "reward_weight_individual_penalty", -1.0))
        e = float(getattr(self, "reward_weight_team_penalty", 0.0))

        return (
            a * float(team_delta)
            + b * float(opp_delta)
            + c * float(individual_delta)
            + d * float(individual_penalty)
            + e * float(team_penalty)
        )

    
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

    def render_field_markings(self, ax: Any) -> None:
        """Render field markings shared by the game's PNG output."""
        field_half = self.field_size_inches / 2
        ax.plot([-field_half, field_half], [0, 0], color='#d0d0d0', linewidth=2)

    def camera_fov_degrees(self) -> float:
        """Return the camera field of view used by the renderer."""
        return 90.0

    def camera_range_inches(self) -> float:
        """Return the camera range used by the renderer."""
        return 72.0
    
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
        num_steps: int = 0,
        agent_times: Optional[Dict[str, float]] = None,
        action_time_remaining: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Render game-specific info panel.
        
        Args:
            ax_info: Matplotlib axes for info panel
            agents: List of active agent names
            actions: Dict of actions taken (or None)
            rewards: Dict of rewards received (or None)
            num_steps: Current environment step number
            agent_times: Dict mapping agent names to current game time
            action_time_remaining: Dict mapping agent names to remaining time
                (seconds) in the currently executing action
        """
        pass
    
    # =========================================================================
    # Optional Methods (with default implementations)
    # =========================================================================
    
    def is_valid_action(self, agent: str, action: int, observation: np.ndarray) -> bool:
        """
        Check if an action is valid in the current state.
        
        Override this to implement action masking.
        
        Args:
            agent: Agent name
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
    
    @staticmethod
    def _get_robot_edges(center: np.ndarray, length: float, width: float, orientation_deg: float) -> List[np.ndarray]:
        """Return the four rectangle corners for a robot at the given pose."""
        half_l = length / 2.0
        half_w = width / 2.0
        theta = np.deg2rad(90.0 - float(orientation_deg))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w],
        ], dtype=float)
        return (rot @ corners.T).T + center

    @staticmethod
    def _project_onto_axis(points: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        projections = points @ axis
        return float(np.min(projections)), float(np.max(projections))

    @classmethod
    def _oriented_rectangles_overlap(
        cls,
        center_a: np.ndarray,
        length_a: float,
        width_a: float,
        orient_a_deg: float,
        center_b: np.ndarray,
        length_b: float,
        width_b: float,
        orient_b_deg: float,
    ) -> bool:
        """Check whether two rectangular robots overlap using their edges."""
        corners_a = cls._get_robot_edges(center_a, length_a, width_a, orient_a_deg)
        corners_b = cls._get_robot_edges(center_b, length_b, width_b, orient_b_deg)

        axes = []
        for corners in (corners_a, corners_b):
            edge_vectors = np.diff(np.vstack([corners, corners[0]]), axis=0)
            for edge in edge_vectors:
                axis = np.array([edge[1], -edge[0]], dtype=float)
                norm = np.linalg.norm(axis)
                if norm > 0:
                    axes.append(axis / norm)

        for axis in axes:
            min_a, max_a = cls._project_onto_axis(corners_a, axis)
            min_b, max_b = cls._project_onto_axis(corners_b, axis)
            if max_a < min_b or max_b < min_a:
                return False

        return True

    @staticmethod
    def _rectangle_circle_collision(
        center: np.ndarray,
        length: float,
        width: float,
        orientation_deg: float,
        obstacle_center: np.ndarray,
        obstacle_radius: float,
    ) -> bool:
        """Check whether a robot rectangle intersects a circular obstacle boundary."""
        half_l = length / 2.0
        half_w = width / 2.0
        theta = np.deg2rad(90.0 - float(orientation_deg))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        rel = obstacle_center - center
        rel_local = rot.T @ rel
        nearest_x = np.clip(rel_local[0], -half_l, half_l)
        nearest_y = np.clip(rel_local[1], -half_w, half_w)
        dx = rel_local[0] - nearest_x
        dy = rel_local[1] - nearest_y
        return (dx * dx + dy * dy) <= (obstacle_radius ** 2)

    @classmethod
    def _rectangle_hits_field_bounds(
        cls,
        center: np.ndarray,
        length: float,
        width: float,
        orientation_deg: float,
        field_half: float,
    ) -> bool:
        """Check whether any edge of the robot rectangle crosses the field wall."""
        corners = cls._get_robot_edges(center, length, width, orientation_deg)
        return (
            np.any(corners[:, 0] > field_half)
            or np.any(corners[:, 0] < -field_half)
            or np.any(corners[:, 1] > field_half)
            or np.any(corners[:, 1] < -field_half)
        )

    def check_robot_collision(self, agent: str) -> bool:
        """
        Check if an agent has collided with another robot or obstacle boundary.

        Uses the robot's actual rectangle edges instead of a center-radius approximation.
        """
        if not self.state or "agents" not in self.state:
            return False

        agent_state = self.state["agents"].get(agent)
        if not agent_state:
            return False

        agent_pos = agent_state.get("projected_position", agent_state.get("position"))
        if agent_pos is None:
            return False

        agent_robot = self.get_robot_for_agent(agent)
        if not agent_robot:
            return False

        agent_length = float(agent_robot.length or agent_robot.size.value)
        agent_width = float(agent_robot.width or agent_robot.size.value)
        agent_orientation = float(agent_state.get("orientation", [0.0])[0])

        for other_agent, other_state in self.state["agents"].items():
            if other_agent == agent:
                continue

            other_pos = other_state.get("projected_position", other_state.get("position"))
            if other_pos is None:
                continue

            other_robot = self.get_robot_for_agent(other_agent)
            if not other_robot:
                continue

            other_length = float(other_robot.length or other_robot.size.value)
            other_width = float(other_robot.width or other_robot.size.value)
            other_orientation = float(other_state.get("orientation", [0.0])[0])

            if self._oriented_rectangles_overlap(
                np.asarray(agent_pos, dtype=float),
                agent_length,
                agent_width,
                agent_orientation,
                np.asarray(other_pos, dtype=float),
                other_length,
                other_width,
                other_orientation,
            ):
                return True

        field_half = float(self.field_size_inches) / 2.0
        if self._rectangle_hits_field_bounds(
            np.asarray(agent_pos, dtype=float),
            agent_length,
            agent_width,
            agent_orientation,
            field_half,
        ):
            return True

        obstacles = self.get_permanent_obstacles() if hasattr(self, "get_permanent_obstacles") else []
        for obstacle in obstacles:
            if getattr(obstacle, "ignore_collision", False):
                continue
            obstacle_radius = float(getattr(obstacle, "radius", 0.0))
            obstacle_center = np.array([float(obstacle.x), float(obstacle.y)], dtype=float)
            if self._rectangle_circle_collision(
                np.asarray(agent_pos, dtype=float),
                agent_length,
                agent_width,
                agent_orientation,
                obstacle_center,
                obstacle_radius,
            ):
                return True

        return False
    
    def get_collision_penalty(self) -> float:
        """
        Get the collision penalty value.
        
        Override this to customize the penalty for collisions.
        
        Returns:
            Penalty value (default: 3.0)
        """
        return 3.0

    def update_robot_position(self, agent: str, position: np.ndarray) -> None:
        """
        Hook for game-specific side effects when a robot's position is updated.

        The environment owns authoritative robot position updates. Games can
        override this hook to keep related game state synchronized (e.g.,
        held objects following the robot center).

        Args:
            agent: Agent name
            position: Updated robot center position
        """
        pass

    def apply_events(self, agent: str, events: List[ActionEvent]) -> None:
        """Apply a list of ActionEvents for an agent.
        
        Called by the environment when a step's ticks are exhausted.
        Probability resolution is handled by the environment before calling
        this method — only events that succeeded (or their on_failure
        alternatives) are passed here.
        
        Override this in game subclasses to implement game-specific event
        processing logic.
        
        Args:
            agent: Agent name
            events: List of ActionEvents to apply (already resolved)
        """
        pass
