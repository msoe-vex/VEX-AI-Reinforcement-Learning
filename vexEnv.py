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
                    max_accel=100.0 / FIELD_SIZE_INCHES
                )
            except Exception:
                self.path_planner = None
        else:
            self.path_planner = None

    def inches_to_planner_scale(self, pos_inches):
        """Convert position from inches (-72 to 72) to path planner scale (0-1)."""
        if isinstance(pos_inches, np.ndarray):
            return (pos_inches + 72.0) / FIELD_SIZE_INCHES
        return (np.array(pos_inches) + 72.0) / FIELD_SIZE_INCHES
    
    def planner_scale_to_inches(self, pos_normalized):
        """Convert position from path planner scale (0-1) to inches (-72 to 72)."""
        if isinstance(pos_normalized, np.ndarray):
            return pos_normalized * FIELD_SIZE_INCHES - 72.0
        return np.array(pos_normalized) * FIELD_SIZE_INCHES - 72.0

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
        
        # Reset goal manager
        self.goal_manager.reset()
        
        if self.randomize:
            self.environment_state = self._get_random_environment_state(seed)
        else:
            self.environment_state = self._get_initial_environment_state()
            
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
            }

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
            float(goal_counts[GoalType.LONG_TOP]),
            float(goal_counts[GoalType.LONG_BOTTOM]),
            float(goal_counts[GoalType.CENTER]),
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
        
        # Determine current game time
        min_game_time = min([
            self.environment_state["agents"][agent]["gameTime"] 
            for agent in self.agents
        ])
        
        # Check terminations
        terminations = {
            agent: min_game_time >= self.mode_config.total_time 
            for agent in self.agents
        }
        terminations["__all__"] = all(terminations.values())
        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = False

        for agent, action in actions.items():
            agent_state = self.environment_state["agents"][agent]
            
            if agent_state["gameTime"] > min_game_time or not agent_state["active"]:
                continue

            initial_score = self._compute_score()
            penalty = 0.0
            duration = 0.5  # Default action duration

            # Execute action
            duration, penalty = self._execute_action(agent, action, agent_state, duration, penalty)

            agent_state["gameTime"] += duration
            
            # Calculate reward
            new_score = self._compute_score()
            rewards[agent] = compute_instant_reward(initial_score, new_score, penalty)
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
        
        # PICK UP NEAREST BLOCK
        if action == Actions.PICK_UP_NEAREST_BLOCK.value:
            duration, penalty = self._action_pickup_block(agent_state, duration, penalty)
        
        # SCORE IN GOALS
        elif action == Actions.SCORE_IN_LONG_GOAL_TOP.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_TOP, duration, penalty
            )
        elif action == Actions.SCORE_IN_LONG_GOAL_BOTTOM.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.LONG_BOTTOM, duration, penalty
            )
        elif action == Actions.SCORE_IN_CENTER_GOAL.value:
            duration, penalty = self._action_score_in_goal(
                agent_state, GoalType.CENTER, duration, penalty
            )
        
        # DRIVE TO LOADERS
        elif action in [
            Actions.DRIVE_TO_LOADER_TL.value, 
            Actions.DRIVE_TO_LOADER_TR.value,
            Actions.DRIVE_TO_LOADER_BL.value, 
            Actions.DRIVE_TO_LOADER_BR.value
        ]:
            idx = action - Actions.DRIVE_TO_LOADER_TL.value
            duration, penalty = self._action_drive_to_loader(agent_state, idx, duration, penalty)
        
        # CLEAR LOADER
        elif action == Actions.CLEAR_LOADER.value:
            duration, penalty = self._action_clear_loader(agent_state, duration, penalty)
        
        # PARK (dynamic based on team)
        elif action == Actions.PARK.value:
            duration, penalty = self._action_park(agent_state, duration, penalty)
        
        # IDLE
        elif action == Actions.IDLE.value:
            duration = 0.1
        
        return duration, penalty

    def _action_pickup_block(self, agent_state, duration, penalty):
        """Pick up the nearest block on the field."""
        target_block_idx = -1
        min_dist = float('inf')
        
        for i, block in enumerate(self.environment_state["blocks"]):
            if block["status"] == BlockStatus.ON_FIELD:
                dist = np.linalg.norm(agent_state["position"] - block["position"])
                if dist < min_dist:
                    min_dist = dist
                    target_block_idx = i
                    
        if target_block_idx != -1:
            # Move to block
            target_pos = self.environment_state["blocks"][target_block_idx]["position"]
            dist_travelled = np.linalg.norm(agent_state["position"] - target_pos)
            agent_state["position"] = target_pos.copy()
            duration += dist_travelled / ROBOT_SPEED
            
            # Pickup
            self.environment_state["blocks"][target_block_idx]["status"] = BlockStatus.HELD
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
        # Long goals: robots face up (90°) or down (-90°)
        # Center goal: robots face at 45° angles
        if goal_type == GoalType.LONG_TOP:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - ROBOT_LENGTH/2 - 2.0, 48.0])
                orientation = 0.0  # Face right toward goal
            else:
                robot_pos = np.array([24.0 + ROBOT_LENGTH/2 + 2.0, 48.0])
                orientation = np.pi  # Face left toward goal
        elif goal_type == GoalType.LONG_BOTTOM:
            if scoring_side == "left":
                robot_pos = np.array([-24.0 - ROBOT_LENGTH/2 - 2.0, -48.0])
                orientation = 0.0  # Face right
            else:
                robot_pos = np.array([24.0 + ROBOT_LENGTH/2 + 2.0, -48.0])
                orientation = np.pi  # Face left
        else:  # CENTER goal
            if scoring_side == "left":
                robot_pos = np.array([-12.0 - ROBOT_LENGTH/2 - 2.0, 0.0])
                orientation = 0.0  # Face right toward center
            else:
                robot_pos = np.array([12.0 + ROBOT_LENGTH/2 + 2.0, 0.0])
                orientation = np.pi  # Face left toward center
        
        dist = np.linalg.norm(agent_state["position"] - robot_pos)
        agent_state["position"] = robot_pos.astype(np.float32)
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        duration += dist / ROBOT_SPEED
        
        # Score all held blocks
        scored_count = 0
        target_status = BlockStatus.get_status_for_goal(goal_type)
        
        for block in self.environment_state["blocks"]:
            if block["status"] == BlockStatus.HELD:
                # Add block to goal queue
                ejected_id, _ = goal.add_block_from_nearest(
                    id(block), 
                    agent_state["position"]
                )
                
                block["status"] = target_status
                block["position"] = nearest_entry.copy()
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
        
        agent_state["held_blocks"] = 0
        duration += 0.5 * scored_count
        
        return duration, penalty

    def _action_drive_to_loader(self, agent_state, loader_idx, duration, penalty):
        """Drive to a specific loader, positioning flush and facing it at exact 45° angle."""
        loader_pos = LOADERS[loader_idx].position
        
        # Set exact 45-degree angles based on loader position (loaders are in corners)
        # TL (0): loader at (-72, 48), robot faces 45° (toward corner)
        # TR (1): loader at (72, 48), robot faces 135°
        # BL (2): loader at (-72, -48), robot faces -45° (315°)
        # BR (3): loader at (72, -48), robot faces -135° (225°)
        offset = ROBOT_LENGTH / 2 + 8.0
        
        if loader_idx == 0:  # Top Left
            orientation = np.pi / 4  # 45°
            robot_pos = loader_pos + np.array([offset * 0.707, -offset * 0.707])
        elif loader_idx == 1:  # Top Right
            orientation = 3 * np.pi / 4  # 135°
            robot_pos = loader_pos + np.array([-offset * 0.707, -offset * 0.707])
        elif loader_idx == 2:  # Bottom Left
            orientation = -np.pi / 4  # -45°
            robot_pos = loader_pos + np.array([offset * 0.707, offset * 0.707])
        else:  # Bottom Right (3)
            orientation = -3 * np.pi / 4  # -135°
            robot_pos = loader_pos + np.array([-offset * 0.707, offset * 0.707])
        
        dist = np.linalg.norm(agent_state["position"] - robot_pos)
        agent_state["position"] = robot_pos.astype(np.float32)
        agent_state["orientation"] = np.array([orientation], dtype=np.float32)
        duration += dist / ROBOT_SPEED
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
        
        dist = np.linalg.norm(agent_state["position"] - park_center)
        agent_state["position"] = park_center.copy()
        duration += dist / ROBOT_SPEED
        agent_state["parked"] = True
        
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
        rect_lg_t = patches.Rectangle((-24, 46), 48, 4, color='orange', alpha=0.3)
        ax.add_patch(rect_lg_t)
        
        rect_lg_b = patches.Rectangle((-24, -50), 48, 4, color='orange', alpha=0.3)
        ax.add_patch(rect_lg_b)
        
        # Center Structure (X shape)
        w, h = 24, 4
        rect_center_1 = patches.Rectangle(
            (-w/2, -h/2), w, h, color='orange', alpha=0.3,
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, 45) + ax.transData
        )
        ax.add_patch(rect_center_1)
        
        rect_center_2 = patches.Rectangle(
            (-w/2, -h/2), w, h, color='orange', alpha=0.3,
            transform=mtransforms.Affine2D().rotate_deg_around(0, 0, -45) + ax.transData
        )
        ax.add_patch(rect_center_2)
        
        # Draw Loaders
        for loader in LOADERS:
            circle = patches.Circle(
                (loader.position[0], loader.position[1]), 
                6.0, color='orange'
            )
            ax.add_patch(circle)
        
        # Draw Permanent Obstacles
        for obs in PERMANENT_OBSTACLES:
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius, 
                fill=False, edgecolor='black', linestyle=':', linewidth=2
            )
            ax.add_patch(circle)

        # Draw Blocks
        for block in self.environment_state["blocks"]:
            if block["status"] < 5:
                c = 'red'
                if block["status"] == BlockStatus.HELD:
                    c = 'blue'
                elif block["status"] > BlockStatus.HELD:
                    c = 'purple'
                
                hexagon = patches.RegularPolygon(
                    (block["position"][0], block["position"][1]), 
                    numVertices=6, 
                    radius=2.4, 
                    orientation=0,
                    facecolor=c, 
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(hexagon)

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
                    width=1.5, color='yellow', 
                    head_width=4, head_length=2, zorder=10,
                    edgecolor='black', linewidth=0.5)
            
            # Add agent number label on robot
            ax.text(x, y, str(i), fontsize=8, ha='center', va='center', 
                   color='white', fontweight='bold', zorder=11)
            
            # Add to info panel
            action_text = "---"
            if actions and agent in actions:
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
        ax_info.text(0.05, info_y, f"Score: {self.score}", fontsize=11, fontweight='bold', va='top')
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
    
    while not done and step_count < 100:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Print actions for each agent
        step_count += 1
        print(f"\nStep {step_count}:")
        for agent, action in actions.items():
            action_name = Actions(action).name
            print(f"  {agent}: {action_name}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations.get("__all__", False) or truncations.get("__all__", False)
        env.render(actions=actions, rewards=rewards)
        
    print(f"\nSimulation complete after {step_count} steps.")
    print(f"Final score: {env.score}")
    
    env.createGIF()
    print("GIF saved.")
