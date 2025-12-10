"""
VEX Push Back Multi-Agent Environment

A reinforcement learning environment for VEX V5RC/VU Push Back competition.
Supports all four competition modes:
- VEX U Competition (2v2)
- VEX U Skills (single robot)
- VEX AI Skills (single AI robot)
- VEX AI Competition (2v2 AI)

COORDINATE SYSTEM:
- All positions are in INCHES (-72 to 72)
- Field size is 144 inches x 144 inches (12 feet x 12 feet), centered at origin
- Path planner uses normalized coordinates (0.0 to 1.0)
"""

import functools
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from ray.rllib.env import MultiAgentEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
import os
import imageio.v2 as imageio

# Import modular components
try:
    from .game_modes import (
        CompetitionMode, Team, ModeConfig, 
        get_mode_config, get_agents_for_mode, get_team_from_agent
    )
    from .field import (
        FIELD_SIZE_INCHES, ROBOT_WIDTH, ROBOT_LENGTH, BUFFER_RADIUS_INCHES,
        NUM_LOADERS, NUM_BLOCKS_FIELD, TOTAL_BLOCKS, FOV,
        GoalType, GOALS, LOADERS, PARK_ZONES, PERMANENT_OBSTACLES,
        get_all_initial_blocks, get_robot_start_position
    )
    from .goals import GoalManager, GoalQueue, BlockStatus
    from .scoring import ScoreCalculator, compute_instant_reward
    from .actions import Actions, is_scoring_action, get_loader_index_from_action
    from .path_planner import Obstacle, PathPlanner
except ImportError:
    from game_modes import (
        CompetitionMode, Team, ModeConfig, 
        get_mode_config, get_agents_for_mode, get_team_from_agent
    )
    from field import (
        FIELD_SIZE_INCHES, ROBOT_WIDTH, ROBOT_LENGTH, BUFFER_RADIUS_INCHES,
        NUM_LOADERS, NUM_BLOCKS_FIELD, TOTAL_BLOCKS, FOV,
        GoalType, GOALS, LOADERS, PARK_ZONES, PERMANENT_OBSTACLES,
        get_all_initial_blocks, get_robot_start_position
    )
    from goals import GoalManager, GoalQueue, BlockStatus
    from scoring import ScoreCalculator, compute_instant_reward
    from actions import Actions, is_scoring_action, get_loader_index_from_action
    try:
        from path_planner import Obstacle, PathPlanner
    except ImportError:
        PathPlanner = None
        Obstacle = None


# Default penalty for invalid actions
DEFAULT_PENALTY = -0.1

# Robot speed constants
ROBOT_SPEED = 60.0  # inches per second


def env_creator(config=None):
    """Create environment instance for RLlib registration."""
    config = config or {}
    return Push_Back_Multi_Agent_Env(
        render_mode=None, 
        randomize=config.get("randomize", True),
        competition_mode=config.get("competition_mode", CompetitionMode.VEX_U_SKILLS),
        agents_config=config.get("agents_config", None)
    )


class Push_Back_Multi_Agent_Env(MultiAgentEnv, ParallelEnv):
    """
    VEX V5RC Push Back multi-agent reinforcement learning environment.
    
    Supports:
    - Dual robot operation (both alliance robots)
    - All four competition modes
    - Goal queue mechanics with overflow
    - Dynamic PARK action based on robot team
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "all"], "name": "vex_push_back"}

    def __init__(
        self, 
        render_mode=None, 
        output_directory="", 
        randomize=True, 
        competition_mode=CompetitionMode.VEX_U_SKILLS, 
        agents_config=None
    ):
        """
        Initialize the Push Back environment.
        
        Args:
            render_mode: 'human' for display, 'rgb_array' or 'all' to save frames
            output_directory: Directory for saving renders
            randomize: Whether to randomize initial block positions
            competition_mode: Which competition mode to use
            agents_config: Optional custom list of agent names
        """
        super().__init__()
        
        self.competition_mode = competition_mode
        self.mode_config = get_mode_config(competition_mode)
        
        # Configure agents based on mode or custom config
        if agents_config:
            self.possible_agents = agents_config
        else:
            self.possible_agents = get_agents_for_mode(competition_mode)
        
        self._agent_ids = self.possible_agents
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        
        self.render_mode = render_mode
        self.agents = []
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.output_directory = output_directory
        self.randomize = randomize
        self.invalid_actions = []

        # Initialize goal manager
        self.goal_manager = GoalManager()
        
        # Initialize score calculator for this mode
        self.score_calculator = ScoreCalculator(competition_mode)
        
        self.score = 0
        
        # Path planner setup
        if PathPlanner is not None:
            try:
                self.path_planner = PathPlanner(
                    robot_length=ROBOT_LENGTH / FIELD_SIZE_INCHES,
                    robot_width=ROBOT_WIDTH / FIELD_SIZE_INCHES,
                    buffer_radius=BUFFER_RADIUS_INCHES / FIELD_SIZE_INCHES,
                    max_velocity=80.0 / FIELD_SIZE_INCHES,
                    max_accel=100.0 / FIELD_SIZE_INCHES,
                    field_size_inches=FIELD_SIZE_INCHES,
                    field_center=(0, 0)
                )
            except Exception:
                self.path_planner = None
        else:
            self.path_planner = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Get observation space for an agent."""
        # Observation vector:
        # Pos(2) + Orient(1) + HeldBlocks(1) + Parked(1) + Time(1) = 6
        # Blocks (x,y,status) * TOTAL_BLOCKS (61 * 3 = 183)
        # Loaders Count (4)
        # Goal block counts (3)
        # Total = 196
        obs_size = 6 + (TOTAL_BLOCKS * 3) + 4 + 3
        
        low = np.full((obs_size,), -float('inf'), dtype=np.float32)
        high = np.full((obs_size,), float('inf'), dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Get action space for an agent."""
        return spaces.Discrete(len(Actions))

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        # Reset movement history
        self.agent_movements = {agent: None for agent in self.possible_agents}
        
        # Reset goal manager
        self.goal_manager.reset()
        
        if self.randomize:
            self.environment_state = self._get_random_environment_state(seed)
        else:
            self.environment_state = self._get_initial_environment_state()
        self._update_held_blocks()
        
        # Compute initial score
        self.score = self._compute_score()
            
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_initial_environment_state(self):
        """Create the initial environment state."""
        # Get initial blocks with preloads for each agent
        blocks = get_all_initial_blocks(self.possible_agents)
        
        # Initialize agents
        agents_dict = {}
        for i, agent in enumerate(self.possible_agents):
            team = get_team_from_agent(agent)
            team_str = team.value
            robot_index = int(agent.split("_")[-1])
            
            start_pos = get_robot_start_position(team_str, robot_index)
            
            agents_dict[agent] = {
                "position": start_pos.copy(),
                "orientation": np.array([0.0], dtype=np.float32),
                "team": team,
                "held_blocks": 1,  # Each robot starts with a preload
                "parked": False,
                "gameTime": 0.0,
                "active": True,
                "agent_name": agent,  # Track agent name for block ownership
            }

        # Set held_by for preload blocks (they are at the end of the blocks list)
        # Also ensure preload block color matches the agent's team
        preload_start_idx = NUM_BLOCKS_FIELD + 24  # Field blocks + loader blocks
        for i, agent in enumerate(self.possible_agents):
            if preload_start_idx + i < len(blocks):
                blocks[preload_start_idx + i]["held_by"] = agent
                # Set block color to match agent's team
                team = get_team_from_agent(agent)
                blocks[preload_start_idx + i]["team"] = team.value

        return {
            "agents": agents_dict,
            "blocks": blocks,
            "loaders": [6, 6, 6, 6]  # 6 blocks per loader
        }

    def _get_random_environment_state(self, seed=None):
        """Create randomized initial state for training."""
        state = self._get_initial_environment_state()
        rng = np.random.default_rng(seed)
        
        # Jitter field block positions
        for i in range(NUM_BLOCKS_FIELD):
            state["blocks"][i]["position"] += rng.uniform(-6.0, 6.0, size=2).astype(np.float32)
            
        return state

    def _get_observation(self, agent):
        """Build observation vector for an agent."""
        agent_state = self.environment_state["agents"][agent]
        
        # Flatten block data
        block_data = []
        for b in self.environment_state["blocks"]:
            block_data.extend([b["position"][0], b["position"][1], float(b["status"])])
        
        # Goal block counts
        goal_counts = self.goal_manager.get_goal_counts()
        goal_data = [
            float(goal_counts[GoalType.LONG_1]),
            float(goal_counts[GoalType.LONG_2]),
            float(goal_counts[GoalType.CENTER_UPPER]),
            float(goal_counts[GoalType.CENTER_LOWER]),
        ]
            
        obs = np.concatenate([
            agent_state["position"],
            agent_state["orientation"],
            [float(agent_state["held_blocks"])],
            [1.0 if agent_state["parked"] else 0.0],
            [float(self.mode_config.total_time - agent_state["gameTime"])],
            np.array(block_data, dtype=np.float32),
            np.array(self.environment_state["loaders"], dtype=np.float32),
            np.array(goal_data, dtype=np.float32),
        ])
        
        return obs.astype(np.float32)

    def step(self, actions):
        """Execute one environment step."""
        if not actions:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
            
        self.num_moves += 1
        rewards = {agent: 0.0 for agent in self.agents}
        
        # Determine current game time (for action skipping)
        min_game_time = min([
            self.environment_state["agents"][agent]["gameTime"] 
            for agent in self.agents
        ])
        
        # Check terminations - each agent terminates when THEIR time is up OR they are parked
        terminations = {
            agent: (self.environment_state["agents"][agent]["gameTime"] >= self.mode_config.total_time or
                    self.environment_state["agents"][agent].get("parked", False))
            for agent in self.agents
        }
        # Only end when ALL agents have reached the time limit or are parked
        terminations["__all__"] = all(terminations.values())
        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = False

        for agent, action in actions.items():
            agent_state = self.environment_state["agents"][agent]
            
            # Skip agent if their time is ahead of min_game_time
            if agent_state["gameTime"] > min_game_time or not agent_state["active"]:
                agent_state["action_skipped"] = True
                continue
            
            agent_state["action_skipped"] = False

            initial_score = self._compute_score()
            agent_team = "red" if agent_state.get("team") == Team.RED else "blue"
            opposing_team = "blue" if agent_team == "red" else "red"
            
            # Get team scores before action
            initial_team_scores = self._compute_team_scores()
            
            penalty = 0.0
            duration = 0.5  # Default action duration

            # Execute action
            duration, penalty = self._execute_action(agent, action, agent_state, duration, penalty)

            agent_state["gameTime"] += duration
            
            # Get team scores after action
            new_team_scores = self._compute_team_scores()
            
            # Calculate reward using scoring module
            reward = compute_instant_reward(
                initial_team_scores[agent_team],
                new_team_scores[agent_team],
                initial_team_scores[opposing_team],
                new_team_scores[opposing_team],
                penalty
            )
            
            # Update total score
            new_score = self._compute_score()
            rewards[agent] = reward
            self.score = new_score

        # Update held block positions to follow robots
        self._update_held_blocks()

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        if terminations["__all__"]:
            self.agents = []

        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def _execute_action(self, agent, action, agent_state, duration, penalty):
        """Execute a single action for an agent."""
        
        # Parked robots cannot perform any actions
        if agent_state.get("parked", False):
            return duration, penalty
        
        # Store initial position for path tracking
        initial_pos = agent_state["position"].copy()
        
        # Normalize action to integer value (handle both enum and int)
        if hasattr(action, 'value'):
            action = action.value
        action = int(action)
        
        # PICK UP NEAREST BLOCK
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            duration, penalty = self._action_pickup_block(agent_state, duration, penalty)
        
        # SCORE IN GOALS
        elif action == Actions.SCORE_IN_LONG_GOAL_1.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_1, duration, penalty
            )
        elif action == Actions.SCORE_IN_LONG_GOAL_2.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_2, duration, penalty
            )
        elif action == Actions.SCORE_IN_CENTER_UPPER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_UPPER, duration, penalty
            )
        elif action == Actions.SCORE_IN_CENTER_LOWER.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER_LOWER, duration, penalty
            )
        
        # TAKE FROM LOADERS
        elif action in [
            Actions.TAKE_FROM_LOADER_TL.value, 
            Actions.TAKE_FROM_LOADER_TR.value,
            Actions.TAKE_FROM_LOADER_BL.value, 
            Actions.TAKE_FROM_LOADER_BR.value
        ]:
            idx = action - Actions.TAKE_FROM_LOADER_TL.value
            duration, penalty = self._action_take_from_loader(agent_state, idx, duration, penalty)
        
        # CLEAR LOADER
        elif action == Actions.CLEAR_LOADER.value:
            duration, penalty = self._action_clear_loader(agent_state, duration, penalty)
        
        # PARK (dynamic based on team)
        elif action == Actions.PARK.value:
            duration, penalty = self._action_park(agent_state, duration, penalty)
        
        # TURN TOWARD CENTER
        elif action == Actions.TURN_TOWARD_CENTER.value:
            duration, penalty = self._action_turn_toward_center(agent_state, duration, penalty)
        
        # IDLE
        elif action == Actions.IDLE.value:
            duration = 0.1
        
        # After executing action, check if position changed and store movement
        final_pos = agent_state["position"]
        if not np.array_equal(initial_pos, final_pos): # Moved
            self.agent_movements[agent] = (initial_pos.copy(), final_pos.copy())
        else:
            self.agent_movements[agent] = None
        
        return duration, penalty

    def _action_pickup_block(self, agent_state, duration, penalty):
        """Pick up the nearest block on the field within robot's field of view that matches robot's team."""
        target_block_idx = -1
        min_dist = float('inf')
        
        robot_pos = agent_state["position"]
        robot_theta = agent_state["orientation"][0]
        robot_team = "red" if agent_state.get("team") == Team.RED else "blue"
        
        for i, block in enumerate(self.environment_state["blocks"]):
            if block["status"] == BlockStatus.ON_FIELD:
                # Only pick blocks matching robot's team color
                if block.get("team") != robot_team:
                    continue
                    
                # Check if block is within FOV (90 degrees = pi/2 radians)
                block_pos = block["position"]
                direction_to_block = block_pos - robot_pos
                dist = np.linalg.norm(direction_to_block)
                
                if dist > 0:
                    # Calculate angle to block
                    angle_to_block = np.arctan2(direction_to_block[1], direction_to_block[0])
                    # Normalize angle difference to [-pi, pi]
                    angle_diff = angle_to_block - robot_theta
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    # Check if within FOV (half angle on each side)
                    if abs(angle_diff) <= FOV / 2:
                        if dist < min_dist:
                            min_dist = dist
                            target_block_idx = i
                    
        if target_block_idx != -1:
            # Move to block
            target_pos = self.environment_state["blocks"][target_block_idx]["position"]
            movement_vector = target_pos - agent_state["position"]
            dist_travelled = np.linalg.norm(movement_vector)
            agent_state["position"] = target_pos.copy()
            # Update orientation to face the direction of movement
            if dist_travelled > 0:
                agent_state["orientation"][0] = np.arctan2(movement_vector[1], movement_vector[0])
            duration += dist_travelled / ROBOT_SPEED
            
            # Pickup - track which agent holds this block
            self.environment_state["blocks"][target_block_idx]["status"] = BlockStatus.HELD
            self.environment_state["blocks"][target_block_idx]["held_by"] = agent_state.get("agent_name")
            agent_state["held_blocks"] += 1
        else:
            penalty = DEFAULT_PENALTY
            
        return duration, penalty

    def _action_score_in_goal(self, agent_state, goal_type, duration, penalty):
        """Score held blocks in a goal, approaching from nearest end."""
        if agent_state["held_blocks"] <= 0:
            return duration, penalty + DEFAULT_PENALTY
        
        goal = self.goal_manager.get_goal(goal_type)
        
        # Get nearest entry point
        nearest_entry = goal.get_nearest_entry(agent_state["position"])
        scoring_side = goal.get_nearest_side(agent_state["position"])
        
        # Calculate position flush with goal and set exact angle
        # Long goals: robots face left (0°) or right (180°)
        # Center goals: robots face at 45° angles matching the diagonal
        if goal_type == GoalType.LONG_1:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - ROBOT_LENGTH/2 - 2.0, 48.0])
                orientation = 0.0  # Face right toward goal
            else:
                robot_pos = np.array([24.0 + ROBOT_LENGTH/2 + 2.0, 48.0])
                orientation = np.pi  # Face left toward goal
        elif goal_type == GoalType.LONG_2:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - ROBOT_LENGTH/2 - 2.0, -48.0])
                orientation = 0.0  # Face right
            else:
                robot_pos = np.array([24.0 + ROBOT_LENGTH/2 + 2.0, -48.0])
                orientation = np.pi  # Face left
        elif goal_type == GoalType.CENTER_UPPER:
            # Upper center: diagonal at -45° rotation
            # left_entry is (-8.5, 8.5) = TL, right_entry is (8.5, -8.5) = BR
            offset = ROBOT_LENGTH/2 + 4.0
            if scoring_side == "left":
                # Approach from top-left
                robot_pos = np.array([-8.5 - offset * 0.707, 8.5 + offset * 0.707])
                orientation = -np.pi / 4  # Face toward bottom-right
            else:
                # Approach from bottom-right
                robot_pos = np.array([8.5 + offset * 0.707, -8.5 - offset * 0.707])
                orientation = 3 * np.pi / 4  # Face toward top-left
        else:  # CENTER_LOWER
            # Lower center: diagonal from upper-right to bottom-left (-45° rotation)
            # left_entry is (8.5, 8.5) = UR, right_entry is (-8.5, -8.5) = BL
            offset = ROBOT_LENGTH/2 + 4.0
            if scoring_side == "left":
                # Approach from upper-right
                robot_pos = np.array([8.5 + offset * 0.707, 8.5 + offset * 0.707])
                orientation = -3 * np.pi / 4  # Face toward bottom-left
            else:
                # Approach from bottom-left
                robot_pos = np.array([-8.5 - offset * 0.707, -8.5 - offset * 0.707])
                orientation = np.pi / 4  # Face toward upper-right
        
        movement_vector = robot_pos - agent_state["position"]
        dist = np.linalg.norm(movement_vector)
        agent_state["position"] = robot_pos.astype(np.float32)
        # Face movement direction by default, then override with goal-specific orientation if specified
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        duration += dist / ROBOT_SPEED
        
        # Score all blocks held by this agent
        scored_count = 0
        dropped_count = 0
        target_status = BlockStatus.get_status_for_goal(goal_type)
        agent_name = agent_state.get("agent_name")
        robot_team = "red" if agent_state.get("team") == Team.RED else "blue"
        
        for block in self.environment_state["blocks"]:
            # Only score blocks held by THIS agent
            if block["status"] == BlockStatus.HELD and block.get("held_by") == agent_name:
                # Check if block color matches robot's team
                block_team = block.get("team")
                
                if block_team == robot_team:
                    # Correct color - score it in the goal
                    # Add block to goal queue
                    ejected_id, _ = goal.add_block_from_nearest(
                        id(block), 
                        agent_state["position"]
                    )
                    
                    block["status"] = target_status
                    block["held_by"] = None  # No longer held
                    scored_count += 1
                    
                    # Handle overflow - ejected block goes back to field
                    if ejected_id is not None:
                        # Find the ejected block and put it on the field
                        for b in self.environment_state["blocks"]:
                            if id(b) == ejected_id:
                                b["status"] = BlockStatus.ON_FIELD
                                # Place on opposite side of goal
                                if scoring_side == "left":
                                    b["position"] = goal.right_entry.copy()
                                else:
                                    b["position"] = goal.left_entry.copy()
                                break
                else:
                    # Wrong color - drop it on the field
                    block["status"] = BlockStatus.ON_FIELD
                    block["held_by"] = None
                    # Drop at robot's current position
                    block["position"] = agent_state["position"].copy()
                    dropped_count += 1
        
        # Update all block positions in this goal to shift them along the tube
        self._update_goal_block_positions(goal, target_status)
        
        agent_state["held_blocks"] = 0
        duration += 0.5 * scored_count
        
        return duration, penalty
    
    def _update_goal_block_positions(self, goal, target_status):
        """Update positions of all blocks in a goal based on their queue positions."""
        # Get the queue's calculated positions (only blocks actually in the goal queue)
        block_positions = goal.get_block_positions(goal.goal_position.center)
        
        # Create a set of block IDs that are in the goal queue
        goal_block_ids = {block_id for block_id, pos in block_positions}
        
        # Create a mapping from block ID to position
        id_to_position = {block_id: pos for block_id, pos in block_positions}
        
        # Update each block's position if it's in the goal queue
        for block in self.environment_state["blocks"]:
            block_id = id(block)
            if block_id in goal_block_ids:
                block["position"] = id_to_position[block_id].copy()
                block["status"] = target_status  # Ensure status is correct

    def _action_take_from_loader(self, agent_state, loader_idx, duration, penalty):
        """Take a block from a specific loader, positioning perpendicular to the wall."""
        loader_pos = LOADERS[loader_idx].position
        
        # Position robot perpendicular to the wall the loader is on
        # Loaders are in corners - approach perpendicular to the X-axis walls (left/right)
        # TL (0): loader at (-72, 48) on left wall, robot to the right facing left (180°)
        # TR (1): loader at (72, 48) on right wall, robot to the left facing right (0°)
        # BL (2): loader at (-72, -48) on left wall, robot to the right facing left (180°)
        # BR (3): loader at (72, -48) on right wall, robot to the left facing right (0°)
        offset = ROBOT_LENGTH / 2 + 8.0
        
        if loader_idx == 0:  # Top Left - on left wall
            orientation = np.pi  # 180° - facing left toward loader
            robot_pos = np.array([loader_pos[0] + offset, loader_pos[1]])
        elif loader_idx == 1:  # Top Right - on right wall
            orientation = 0.0  # 0° - facing right toward loader
            robot_pos = np.array([loader_pos[0] - offset, loader_pos[1]])
        elif loader_idx == 2:  # Bottom Left - on left wall
            orientation = np.pi  # 180° - facing left toward loader
            robot_pos = np.array([loader_pos[0] + offset, loader_pos[1]])
        else:  # Bottom Right (3) - on right wall
            orientation = 0.0  # 0° - facing right toward loader
            robot_pos = np.array([loader_pos[0] - offset, loader_pos[1]])
        
        movement_vector = robot_pos - agent_state["position"]
        dist = np.linalg.norm(movement_vector)
        agent_state["position"] = robot_pos.astype(np.float32)
        # Face movement direction by default, then override with loader-specific orientation
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        duration += dist / ROBOT_SPEED
        
        # Take a block from the loader if available
        if self.environment_state["loaders"][loader_idx] > 0:
            self.environment_state["loaders"][loader_idx] -= 1
            agent_state["held_blocks"] += 1
            duration += 0.5  # Time to grab the block
            
            # Update block status
            for block in self.environment_state["blocks"]:
                if block["status"] == BlockStatus.IN_LOADER_TL + loader_idx:
                    block["status"] = BlockStatus.HELD
                    block["position"] = agent_state["position"].copy()
                    break
        
        return duration, penalty

    def _action_clear_loader(self, agent_state, duration, penalty):
        """Clear blocks from the nearest loader."""
        closest_loader = -1
        
        for loader in LOADERS:
            if np.linalg.norm(agent_state["position"] - loader.position) < 18.0:
                closest_loader = loader.index
                break
        
        if closest_loader != -1 and self.environment_state["loaders"][closest_loader] > 0:
            # Dispense a block
            loader_status = 5 + closest_loader
            for block in self.environment_state["blocks"]:
                if block["status"] == loader_status:
                    block["status"] = BlockStatus.ON_FIELD
                    block["position"] = agent_state["position"] + np.random.uniform(-6.0, 6.0, 2).astype(np.float32)
                    self.environment_state["loaders"][closest_loader] -= 1
                    break
            duration += 1.0
        else:
            penalty = DEFAULT_PENALTY
            
        return duration, penalty

    def _action_park(self, agent_state, duration, penalty):
        """Park in the robot's team zone."""
        team = agent_state.get("team", Team.RED)
        team_str = team.value
        
        park_zone = PARK_ZONES[team_str]
        park_center = park_zone.center
        
        movement_vector = park_center - agent_state["position"]
        dist = np.linalg.norm(movement_vector)
        agent_state["position"] = park_center.copy()
        duration += dist / ROBOT_SPEED
        agent_state["parked"] = True
        
        # Parked robots face either up or down
        agent_state["orientation"] = np.array([np.random.choice([np.pi/2, -np.pi/2])], dtype=np.float32)
        
        return duration, penalty

    def _action_turn_toward_center(self, agent_state, duration, penalty):
        """Turn robot to face the center of the field (0, 0) for maximum visibility."""
        robot_pos = agent_state["position"]
        
        # Calculate angle from robot to center (0, 0)
        center = np.array([0.0, 0.0])
        direction = center - robot_pos
        
        # Calculate angle using atan2 (handles all quadrants correctly)
        target_angle = np.arctan2(direction[1], direction[0])
        
        # Set orientation to face center
        agent_state["orientation"] = np.array([target_angle], dtype=np.float32)
        
        # Small time cost for turning
        duration += 0.3
        
        return duration, penalty

    def _update_held_blocks(self):
        """Update positions of held blocks to follow their robot."""
        for agent in self.agents:
            agent_state = self.environment_state["agents"][agent]
            if agent_state["held_blocks"] > 0:
                for block in self.environment_state["blocks"]:
                    if block["status"] == BlockStatus.HELD:
                        block["position"] = agent_state["position"].copy()

    def _compute_score(self):
        """Compute the current score using the score calculator."""
        return self.score_calculator.calculate_total_score(
            self.goal_manager,
            self.environment_state["loaders"],
            self.environment_state["agents"],
            self.environment_state["blocks"],
        )
    
    def _compute_team_scores(self):
        """Compute scores for each team based on their blocks in goals."""
        scores = {"red": 0, "blue": 0}
        
        for block in self.environment_state["blocks"]:
            # Check if block is in a goal (status 2-5)
            if block["status"] > BlockStatus.HELD and block["status"] <= 5:
                block_team = block.get("team", "red")
                scores[block_team] += self.mode_config.block_points
        
        return scores

    def is_valid_action(self, action, observation, last_action=None):
        """Check if an action is valid in the current state."""
        # Scoring requires held blocks
        if is_scoring_action(Actions(action)):
            held_blocks = observation[3]  # Index 3 is held blocks count
            if held_blocks <= 0:
                return False
        return True

    def render(self, actions=None, rewards=None):
        """Render the current environment state."""
        if self.render_mode is None:
            return

        # Create wider figure with info panel on the right
        fig = plt.figure(figsize=(12, 8))
        
        # Main field on the left (larger portion)
        ax = fig.add_axes([0.05, 0.1, 0.65, 0.85])  # [left, bottom, width, height]
        ax.set_xlim(-72, 72)
        ax.set_ylim(-72, 72)
        ax.set_facecolor('#cccccc')
        ax.set_aspect('equal')
        
        # Info panel on the right
        ax_info = fig.add_axes([0.72, 0.1, 0.25, 0.85])
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        
        # Draw auto line
        ax.plot([-72, 72], [0, 0], color='white', linewidth=2)
        
        # Draw Park Zones
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
        
        # Draw Goals
        # Long Goal 1 (top, y=48)
        rect_lg_1 = patches.Rectangle((-24, 46), 48, 4, facecolor='orange', alpha=0.3, edgecolor='orange')
        ax.add_patch(rect_lg_1)
        ax.text(0, 52, 'Long 1', fontsize=8, ha='center', va='bottom', color='orange')
        # Control zone lines (inner 12 inches = x from -6 to 6)
        ax.plot([-6, -6], [46, 50], color='white', linewidth=2, linestyle='--')
        ax.plot([6, 6], [46, 50], color='white', linewidth=2, linestyle='--')
        
        # Long Goal 2 (bottom, y=-48)
        rect_lg_2 = patches.Rectangle((-24, -50), 48, 4, facecolor='orange', alpha=0.3, edgecolor='orange')
        ax.add_patch(rect_lg_2)
        ax.text(0, -54, 'Long 2', fontsize=8, ha='center', va='top', color='orange')
        # Control zone lines (inner 12 inches = x from -6 to 6)
        ax.plot([-6, -6], [-50, -46], color='white', linewidth=2, linestyle='--')
        ax.plot([6, 6], [-50, -46], color='white', linewidth=2, linestyle='--')
        
        # Center Structure (X shape) - two separate diagonals
        w, h = 24, 4
        # Upper center: -45° rotation - green
        rect_center_upper = patches.Rectangle(
            (-w/2, -h/2), w, h, facecolor='green', alpha=0.4, edgecolor='green',
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, -45) + ax.transData
        )
        ax.add_patch(rect_center_upper)
        ax.text(-10, 10, 'Upper', fontsize=6, ha='center', va='center', color='green')
        
        # Lower center: 45° rotation - purple
        rect_center_lower = patches.Rectangle(
            (-w/2, -h/2), w, h, facecolor='purple', alpha=0.4, edgecolor='purple',
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, 45) + ax.transData
        )
        ax.add_patch(rect_center_lower)
        ax.text(-10, -10, 'Lower', fontsize=6, ha='center', va='center', color='purple')
        
        # Draw Loaders and blocks in them
        for idx, loader in enumerate(LOADERS):
            circle = patches.Circle(
                (loader.position[0], loader.position[1]), 
                6.0, fill=False, edgecolor='orange', linewidth=4
            )
            ax.add_patch(circle)
            
            # Draw blocks in this loader (stacked vertically)
            num_blocks_in_loader = self.environment_state["loaders"][idx]
            if num_blocks_in_loader > 0:
                # Stack blocks vertically inside the loader
                # Start from bottom and work up so top block is drawn last
                block_spacing = 1.0  # Vertical spacing between blocks
                start_y = loader.position[1] - (num_blocks_in_loader - 1) * block_spacing / 2
                
                for block_idx in range(num_blocks_in_loader):
                    block_y = start_y + block_idx * block_spacing
                    # Alternate colors for visual distinction
                    block_color = 'red' if block_idx % 2 == 0 else 'blue'
                    hexagon = patches.RegularPolygon(
                        (loader.position[0], block_y), 
                        numVertices=6, 
                        radius=1.8,  # Slightly smaller to fit in loader
                        orientation=0,
                        facecolor=block_color, 
                        edgecolor='black',
                        linewidth=1
                    )
                    ax.add_patch(hexagon)
        
        # Draw Permanent Obstacles
        for obs in PERMANENT_OBSTACLES:
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius, 
                fill=False, edgecolor='black', linestyle=':', linewidth=2
            )
            ax.add_patch(circle)

        # Draw Blocks (don't draw blocks in loaders - status >= 6)
        for block in self.environment_state["blocks"]:
            if block["status"] < 6:  # Not in loaders
                # Fill color based on team
                fill_color = block.get("team", "red")
                
                # Edge color based on status
                if block["status"] == BlockStatus.ON_FIELD:
                    edge_color = 'black'
                    edge_width = 1
                elif block["status"] == BlockStatus.HELD:
                    edge_color = 'yellow'
                    edge_width = 3
                else:  # In a goal (status 2-5)
                    edge_color = 'white'
                    edge_width = 2
                
                hexagon = patches.RegularPolygon(
                    (block["position"][0], block["position"][1]), 
                    numVertices=6, 
                    radius=2.4, 
                    orientation=0,
                    facecolor=fill_color, 
                    edgecolor=edge_color,
                    linewidth=edge_width
                )
                ax.add_patch(hexagon)

        # Draw agent paths (dotted lines showing trajectory)
        if self.path_planner is not None:
            for agent in self.agents:
                # Only draw if the agent has moved
                movement = self.agent_movements.get(agent)
                if movement is None:
                    continue
                
                start_pos, end_pos = movement
                agent_state = self.environment_state["agents"][agent]
                team = agent_state.get("team", Team.RED)
                color = 'red' if team == Team.RED else 'blue'
                
                # Compute and draw last path segment
                try:                    
                    # Solve for path (pass everything in inches, get back positions in inches)
                    positions_inches, _, _ = self.path_planner.Solve(
                        start_point=start_pos,
                        end_point=end_pos,
                        obstacles=PERMANENT_OBSTACLES
                    )
                    
                    # Plot path directly (already in inches)
                    ax.plot(positions_inches[:, 0], positions_inches[:, 1], 
                            linestyle=':', linewidth=1.5, color=color, alpha=0.4)
                except Exception as e:
                    # If path planning fails, draw direct line
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            linestyle=':', linewidth=1.5, color=color, alpha=0.4)
                    print(f"Path planning failed for agent {agent} from {start_pos} to {end_pos}")
                    print(e)
        
        # Draw Robots and build info text
        info_y = 0.95
        ax_info.text(0.5, info_y, "Agent Actions", fontsize=12, fontweight='bold', 
                    ha='center', va='top')
        info_y -= 0.08
        
        for i, agent in enumerate(self.agents):
            st = self.environment_state["agents"][agent]
            team = st.get("team", Team.RED)
            robot_color = 'red' if team == Team.RED else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            theta = st["orientation"][0]

            # Draw buffer circle around robot
            circle = patches.Circle(
                (x, y), BUFFER_RADIUS_INCHES + np.sqrt(ROBOT_LENGTH**2 + ROBOT_WIDTH**2)/2, 
                fill=False, edgecolor='black', linestyle=':', linewidth=2
            )
            ax.add_patch(circle)
            
            # Draw FOV wedge (light yellow, transparent)
            fov_radius = 72  # How far the FOV extends
            fov_start_angle = np.degrees(theta - FOV/2)
            fov_end_angle = np.degrees(theta + FOV/2)
            fov_wedge = patches.Wedge(
                (x, y), fov_radius, fov_start_angle, fov_end_angle,
                facecolor='yellow', alpha=0.15, edgecolor='yellow', linewidth=0.5
            )
            ax.add_patch(fov_wedge)
            
            robot_rect = patches.Rectangle(
                (-ROBOT_LENGTH/2, -ROBOT_WIDTH/2), 
                ROBOT_LENGTH, 
                ROBOT_WIDTH,
                edgecolor='black',
                facecolor=robot_color,
                alpha=0.7,
                linewidth=2,
                label=agent
            )
            
            t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
            robot_rect.set_transform(t)
            ax.add_patch(robot_rect)
            
            # Draw orientation arrow - starts from robot center, points in direction of theta
            arrow_length = ROBOT_LENGTH * 0.4
            # Arrow starts from center and points forward
            ax.arrow(x, y, 
                    np.cos(theta) * arrow_length, 
                    np.sin(theta) * arrow_length, 
                    width=1.5, facecolor='yellow', 
                    head_width=4, head_length=2, zorder=10,
                    edgecolor='black', linewidth=0.5)
            
            # Add agent number label on robot
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center', 
                   color='white', fontweight='bold', zorder=11,
                   bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.8))
            
            # Add to info panel
            action_text = "---"
            if actions and agent in actions:
                # Check if action was skipped
                if st.get("action_skipped", False):
                    action_text = "--"
                else:
                    action_val = actions[agent]
                    # Convert to int (handles numpy types) and get action name
                    try:
                        action_int = int(action_val)
                        action_text = Actions(action_int).name
                    except (ValueError, TypeError):
                        if hasattr(action_val, 'name'):
                            action_text = action_val.name
                        else:
                            action_text = str(action_val)
            
            reward_text = ""
            if rewards and agent in rewards:
                reward_text = f" (R: {rewards[agent]:.2f})"
            
            color = 'red' if team == Team.RED else 'blue'
            ax_info.text(0.05, info_y, f"Robot {i} ({team.value}):", fontsize=9, 
                        color=color, fontweight='bold', va='top')
            info_y -= 0.05
            ax_info.text(0.1, info_y, f"{action_text}{reward_text}", fontsize=8, va='top')
            info_y -= 0.03
            ax_info.text(0.1, info_y, f"Time: {st['gameTime']:.1f}s / {self.mode_config.total_time:.0f}s", 
                        fontsize=7, va='top', color='gray')
            info_y -= 0.03
            ax_info.text(0.1, info_y, f"Pos: ({x:.0f}, {y:.0f}) | Held: {st['held_blocks']}", 
                        fontsize=7, va='top', color='gray')
            info_y -= 0.06

        # Add score and game info to info panel
        info_y -= 0.02
        ax_info.axhline(y=info_y, xmin=0.05, xmax=0.95, color='gray', linewidth=0.5)
        info_y -= 0.05
        
        # Show team scores
        team_scores = self._compute_team_scores()
        ax_info.text(0.05, info_y, "Scores:", fontsize=10, fontweight='bold', va='top')
        info_y -= 0.04
        ax_info.text(0.1, info_y, f"Red: {team_scores['red']}", fontsize=9, va='top', color='red', fontweight='bold')
        info_y -= 0.04
        ax_info.text(0.1, info_y, f"Blue: {team_scores['blue']}", fontsize=9, va='top', color='blue', fontweight='bold')
        info_y -= 0.05
        ax_info.text(0.05, info_y, f"Mode: {self.competition_mode.value}", fontsize=8, va='top')
        info_y -= 0.04
        ax_info.text(0.05, info_y, f"Step: {self.num_moves}", fontsize=8, va='top')

        # Title
        ax.set_title(f"V5RC Push Back", fontsize=14, fontweight='bold')
        
        # Save or display
        if self.render_mode == "human":
            plt.show()
        else:
            step_num = self.num_moves
            os.makedirs(os.path.join(self.output_directory, "steps"), exist_ok=True)
            plt.savefig(os.path.join(self.output_directory, "steps", f"step_{step_num}.png"), dpi=100)
            plt.close()

    def close(self):
        """Clean up resources."""
        pass
    
    def clearStepsDirectory(self):
        """Clear the steps directory for new renders."""
        steps_dir = os.path.join(self.output_directory, "steps")
        if os.path.exists(steps_dir):
            for filename in os.listdir(steps_dir):
                os.remove(os.path.join(steps_dir, filename))

    def createGIF(self):
        """Create a GIF from rendered steps."""
        steps_dir = os.path.join(self.output_directory, "steps")
        if not os.path.exists(steps_dir):
            return
            
        images = []
        files = sorted(
            os.listdir(steps_dir), 
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        for filename in files:
            images.append(imageio.imread(os.path.join(steps_dir, filename)))
            
        if images:
            imageio.mimsave(
                os.path.join(self.output_directory, "simulation.gif"), 
                images, 
                fps=10
            )


# Legacy compatibility - GameMode enum maps to CompetitionMode
class GameMode:
    """Legacy compatibility wrapper for GameMode."""
    SKILLS = CompetitionMode.VEX_U_SKILLS
    COMPETITION = CompetitionMode.VEX_U_COMPETITION


TEST_ACTIONS = [
    Actions.PICK_UP_NEAREST_BLOCK,
    Actions.SCORE_IN_CENTER_UPPER,
    Actions.PICK_UP_NEAREST_BLOCK,
    Actions.SCORE_IN_CENTER_LOWER,
    Actions.PICK_UP_NEAREST_BLOCK,
    Actions.SCORE_IN_LONG_GOAL_1,
    Actions.PICK_UP_NEAREST_BLOCK,
    Actions.SCORE_IN_LONG_GOAL_2
]

if __name__ == "__main__":
    # Test the environment
    print("Testing VEX Push Back environment...")
    
    env = Push_Back_Multi_Agent_Env(
        render_mode="all", 
        output_directory="vexEnv",
        competition_mode=CompetitionMode.VEX_U_SKILLS,
        randomize=True
    )
    
    observations, infos = env.reset()
    env.clearStepsDirectory()
    
    print(f"Agents: {env.agents}")
    print(f"Mode: {env.competition_mode.value}")
    print(f"Time limit: {env.mode_config.total_time}s")
    print()
    
    # Render initial state (step 0)
    print("Step 0: Initial positions")
    env.render()
    
    done = False
    step_count = 0

    i = 0
    while not done and step_count < 100:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        # actions = {agent: TEST_ACTIONS[i % len(TEST_ACTIONS)] for agent in env.agents}
        i += 1

        # Print actions for each agent
        step_count += 1
        print(f"\nStep {step_count}:")
        for agent, action in actions.items():
            action_name = Actions(action).name
            print(f"  {agent}: {action_name}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations.get("__all__", False) or truncations.get("__all__", False)
        env.render(actions=actions, rewards=rewards)
    
    # Render final state
    print("\nRendering final state...")
    env.render()
        
    print(f"\nSimulation complete after {step_count} steps.")
    print(f"Final score: {env.score}")
    
    env.createGIF()
    print("GIF saved.")
