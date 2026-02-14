"""
VEX Core - Generic Multi-Agent Environment

Provides the base infrastructure for VEX robotics reinforcement learning.
Delegates game-specific logic to a VexGame implementation.
"""

import functools
import os
import numpy as np

from gymnasium import spaces
from pettingzoo import ParallelEnv
from ray.rllib.env import MultiAgentEnv
from typing import Dict, List, Tuple, Optional, Any

from .base_game import VexGame, Robot, RobotSize, Team

from typing import Optional

DELTA_T = 0.1  # Discrete time step in seconds


class VexMultiAgentEnv(MultiAgentEnv, ParallelEnv):
    """
    Generic VEX multi-agent reinforcement learning environment.
    
    This environment handles:
    - Multi-agent coordination (PettingZoo ParallelEnv)
    - RLlib integration (MultiAgentEnv)
    - Robot state management (position, orientation, time)
    - Rendering infrastructure
    - GIF creation
    
    Game-specific logic is delegated to the VexGame instance.
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "all"], "name": "vex_env"}
    
    def __init__(
        self,
        game: VexGame,
        render_mode: Optional[str] = None,
        output_directory: str = "",
        randomize: bool = True,
    ):
        """
        Initialize the VEX environment.
        
        Args:
            game: VexGame instance defining game-specific mechanics (with robots)
            render_mode: 'human' for display, 'rgb_array' or 'all' to save frames
            output_directory: Directory for saving renders
            randomize: Whether to randomize initial positions
        """
        super().__init__()
        
        self.game = game
            
        self.render_mode = render_mode
        self.output_directory = output_directory
        self.randomize = randomize
        
        # Agent configuration from game
        self.possible_agents = game.possible_agents
        self._agent_ids = self.possible_agents
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        
        # Spaces from game
        self.observation_spaces = {
            agent: game.observation_space(agent) 
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: game.action_space(agent) 
            for agent in self.possible_agents
        }
        
        # Environment state
        self.agents: List[str] = []
        self.environment_state: Dict = {}
        self.num_moves = 0
        self.score = 0
        self.agent_movements: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        
        # Busy state for discrete time steps
        # Maps agent -> {
        #   "start_pos": np.array, "target_pos": np.array,
        #   "target_orient": np.array,
        #   "total_ticks": int, "remaining_ticks": int
        # }
        self.busy_state: Dict[str, Dict] = {}
        
        # Environment-managed agent states (Time, etc.)
        self.env_agent_states: Dict[str, Dict] = {}
        
        # Path planner (optional, set up by subclass or game)
        self.path_planner = None
        self._setup_path_planner()
    
    def _setup_path_planner(self):
        """Set up path planner if available."""
        try:
            from path_planner import PathPlanner
            
            field_size = self.game.field_size_inches
            
            self.path_planner = PathPlanner(
                field_size_inches=field_size,
                field_center=(0, 0)
            )
        except (ImportError, Exception):
            self.path_planner = None
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self.game.observation_space(agent)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return self.game.action_space(agent)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the environment to initial state."""
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.agent_movements = {agent: None for agent in self.possible_agents}
        
        # Reset game-specific state
        self.game.reset()
        
        # Get initial state from game
        self.environment_state = self.game.get_initial_state(
            randomize=self.randomize, 
            seed=seed
        )
        
        # Compute initial score
        self.score = self.game.compute_score()
        
        # Initialize env state
        self.env_agent_states = {
            agent: {"time": 0.0} for agent in self.agents
        }
        
        observations = {
            agent: self.game.get_observation(agent, game_time=0.0)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step (discrete tick).
        Advances time by DELTA_T. Handles busy states and interpolation.
        Handles Tuple Actions (Control, Message).
        """
        if not actions and not self.agents:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
        
        self.num_moves += 1
        
        # DEBUG
        if self.environment_state is None:
            print("CRITICAL: self.environment_state is None in step()!")
        
        # Keep all agents in self.agents until all are terminated
        active_agents = list(self.agents)
        
        
        rewards = {agent: 0.0 for agent in active_agents}
        terminations = {agent: False for agent in active_agents}
        truncations = {agent: False for agent in active_agents}
        infos = {agent: {} for agent in active_agents}
        
        # 1. Handle Busy Agents (Interpolation)
        for agent in active_agents:
            if agent in self.busy_state:
                busy_info = self.busy_state[agent]
                busy_info["remaining_ticks"] -= 1
                
                agent_state = self.environment_state["agents"][agent]
                
                if busy_info["remaining_ticks"] <= 0:
                    # Action complete - snap to target
                    agent_state["position"] = busy_info["target_pos"].copy()
                    agent_state["orientation"] = busy_info["target_orient"].copy()
                    del self.busy_state[agent]
                    infos[agent]["action_completed"] = True
                    # Apply any pending events scheduled for this agent's action
                    if hasattr(self.game, "apply_pending_events"):
                        try:
                            self.game.apply_pending_events(agent)
                        except Exception:
                            pass
                else:
                    # Interpolate
                    total = busy_info["total_ticks"]
                    rem = busy_info["remaining_ticks"]
                    # Calculate progress (0.0 to 1.0)
                    alpha = (total - rem) / total
                    
                    start_pos = busy_info["start_pos"]
                    target_pos = busy_info["target_pos"]
                    
                    # Linear Int: P = P0 + (P1 - P0) * alpha
                    current_pos = start_pos + (target_pos - start_pos) * alpha
                    self.game._update_held_blocks(agent, current_pos + 5)
                    agent_state["position"] = current_pos
                    
                    # Orientation: Point in direction of movement
                    diff = target_pos - start_pos
                    dist = np.linalg.norm(diff)
                    if dist > 0.1: # Only if moving significantly
                        heading = np.arctan2(diff[1], diff[0])
                        agent_state["orientation"] = np.array([heading])
            
                if agent in self.busy_state:
                     infos[agent]["action_skipped"] = True

        # 2. Process Actions for Non-Busy Agents
        for agent, action in actions.items():
            if agent not in active_agents:
                continue
            
            # Skip if terminated or busy
            if terminations[agent]:
                infos[agent]["action_skipped"] = True
                continue
            
            if agent in self.busy_state:
                continue # Already handled above
                
            agent_state = self.environment_state["agents"][agent]
            infos[agent]["action_skipped"] = False
            
            # Capture state BEFORE action execution
            start_pos = agent_state["position"].copy()
            start_orient = agent_state["orientation"].copy()
            
            # Get scores before action
            initial_scores = self.game.compute_score()
            
            # Parsing Action Tuple (Control, Message)
            action_val = action
            emitted_msg = None
            
            # Check if action is sequence (list/tuple/ndarray with len > 1)
            # RLlib Tuple action typically comes as tuple or list
            if isinstance(action, (tuple, list)) and len(action) >= 2:
                # [Control (int), Message (array)]
                action_val = action[0]
                emitted_msg = action[1]
            elif isinstance(action, np.ndarray) and action.ndim > 0 and action.size > 1:
                # Potentially flat array? But standard PPO Output for Tuple is complex.
                # Assuming simple environment interaction script passes tuple.
                pass
                
            # Execute physical action
            action_int = int(action_val.value if hasattr(action_val, 'value') else action_val)
            # Record executed action in the game state so the UI "Current:" row
            # only changes when an actual action was performed.
            agent_state["current_action"] = action_int
            duration, penalty = self.game.execute_action(
                agent, action_int
            )
            
            # Store Emitted Message (if any) - validate length and dtype
            if emitted_msg is not None:
                arr = np.array(emitted_msg, dtype=np.float32).ravel()
                # Pad or truncate to length 8 to avoid downstream broadcasting errors
                if arr.size != 8:
                    if arr.size < 8:
                        padded = np.zeros(8, dtype=np.float32)
                        if arr.size > 0:
                            padded[: arr.size] = arr
                        arr = padded
                    else:
                        arr = arr[:8].astype(np.float32)
                agent_state["emitted_message"] = arr
            else:
                agent_state["emitted_message"] = np.zeros(8, dtype=np.float32)

            # Capture state AFTER action execution (Target)
            target_pos = agent_state["position"].copy()
            target_orient = agent_state["orientation"].copy()
            
            # Prepare Busy State
            ticks = int(max(1, duration / DELTA_T)) 
            
            if ticks > 0:
                # Revert visible state to start
                agent_state["position"] = start_pos
                agent_state["orientation"] = start_orient
                
                self.busy_state[agent] = {
                    "start_pos": start_pos,
                    "target_pos": target_pos,
                    "target_orient": target_orient,
                    "total_ticks": ticks,
                    "remaining_ticks": ticks
                }

                # Update movement trail
                if not np.array_equal(start_pos, target_pos):
                    self.agent_movements[agent] = (start_pos.copy(), target_pos.copy())
            else:
                self.agent_movements[agent] = None
            
            # Rewards
            new_scores = self.game.compute_score()
            reward = self.game.compute_reward(agent, initial_scores, new_scores, penalty)
            rewards[agent] = reward
            
            # Update accumulated duration in ENV STATE
            if agent in self.env_agent_states:
                self.env_agent_states[agent]["time"] += duration

        # 3. Message Aggregation (Update Received Messages for NEXT step)
        self._update_messages(active_agents)

        # Update total score
        self.score = self.game.compute_score()
        
        # Check terminations
        # Check terminations
        terminations = {}
        agents_to_remove = set()
        for agent in active_agents:
             current_time = self.env_agent_states[agent]["time"] if agent in self.env_agent_states else 0.0
             term = self.game.is_agent_terminated(agent, game_time=current_time)
             terminations[agent] = term
             if term:
                 agents_to_remove.add(agent)
                 
        # Remove from self.agents
        for agent in agents_to_remove:
            if agent in self.agents:
                self.agents.remove(agent)
        
        terminations["__all__"] = len(self.agents) == 0
        truncations = {agent: False for agent in active_agents}
        truncations["__all__"] = False
        
        # Observations
        observations = {
            agent: self.game.get_observation(
                agent,
                game_time=self.env_agent_states[agent]["time"] if agent in self.env_agent_states else 0.0
            )
            for agent in active_agents
        }
        
        # Only return rewards/infos for agents receiving observations?
        # PettingZoo standard is return all.
        
        return observations, rewards, terminations, truncations, infos

    def _update_messages(self, active_agents: List[str]) -> None:
        """
        Aggregate messages from neighbors for each agent.
        Updates agent_state["received_messages"].
        """
        COMM_RADIUS = 72.0  # Approx half-field
        
        # Pre-fetch positions and messages
        positions = {}
        messages = {}
        
        # Safety check for None state (fixing test issues just in case)
        if self.environment_state is None:
             return
             
        for agent in active_agents:
            st = self.environment_state["agents"][agent]
            positions[agent] = st["position"]
            # Normalize emitted_message to an 8-dim float32 vector (robustness against malformed state)
            m = st.get("emitted_message", None)
            if m is None:
                messages[agent] = np.zeros(8, dtype=np.float32)
            else:
                m_arr = np.array(m, dtype=np.float32).ravel()
                if m_arr.size != 8:
                    if m_arr.size < 8:
                        padded = np.zeros(8, dtype=np.float32)
                        if m_arr.size > 0:
                            padded[: m_arr.size] = m_arr
                        m_arr = padded
                    else:
                        m_arr = m_arr[:8]
                messages[agent] = m_arr

        for agent in active_agents:
            my_pos = positions[agent]
            received_sum = np.zeros(8, dtype=np.float32)
            count = 0
            
            for other in active_agents:
                if agent == other:
                    continue
                
                other_pos = positions[other]
                dist = np.linalg.norm(my_pos - other_pos)
                
                if dist <= COMM_RADIUS:
                    received_sum += messages[other]
                    count += 1
            
            # Average
            if count > 0:
                received_sum /= count
                
            self.environment_state["agents"][agent]["received_messages"] = received_sum

    

    def is_valid_action(
        self, 
        action: int, 
        observation: np.ndarray, 
        last_action: Optional[int] = None
    ) -> bool:
        """Check if an action is valid."""
        return self.game.is_valid_action(action, observation)
    
    def render(
        self, 
        actions: Optional[Dict] = None, 
        rewards: Optional[Dict] = None
    ) -> None:
        """Render the current environment state."""
        if self.render_mode is None:
            return
        
        # Import matplotlib only when rendering
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.transforms as mtransforms
        
        # Create figure with info panel
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.05, 0.1, 0.65, 0.85])
        ax_info = fig.add_axes([0.72, 0.1, 0.25, 0.85])
        
        field_half = self.game.field_size_inches / 2
        ax.set_xlim(-field_half, field_half)
        ax.set_ylim(-field_half, field_half)
        ax.set_facecolor('#cccccc')
        ax.set_aspect('equal')
        
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        
        # Draw auto line
        ax.plot([-field_half, field_half], [0, 0], color='white', linewidth=2)
        
        # Render game-specific elements
        self.game.render_game_elements(ax)
        
        # Draw robot paths
        self._render_paths(ax)
        
        # Draw robots and info panel
        self._render_robots_and_info(ax, ax_info, actions, rewards)
        
        # Title
        ax.set_title("VEX Environment", fontsize=14, fontweight='bold')
        
        # Save or display
        if self.render_mode == "human":
            plt.show()
        else:
            os.makedirs(os.path.join(self.output_directory, "steps"), exist_ok=True)
            plt.savefig(
                os.path.join(self.output_directory, "steps", f"step_{self.num_moves}.png"),
                dpi=100
            )
            plt.close()
    
    def _render_paths(self, ax) -> None:
        """Render robot paths."""
        if self.path_planner is None:
            return
        
        for agent in self.agents:
            movement = self.agent_movements.get(agent)
            if movement is None:
                continue
            
            start_pos, end_pos = movement
            agent_state = self.environment_state["agents"][agent]
            team = self.game.get_team_for_agent(agent)
            color = 'red' if team == 'red' else 'blue'
            
            try:
                # Get robot for this agent via direct lookup
                robot_config = self.game.get_robot_for_agent(agent)
                
                # Fallback if no specific robot found
                if robot_config is None:
                    robot_config = Robot(
                        name=agent, team=Team.RED, size=RobotSize.INCH_24,
                        start_position=np.array([0.0, 0.0])
                    )

                obstacles = self.game.get_permanent_obstacles()
                positions_inches, _, _ = self.path_planner.Solve(
                    start_point=start_pos,
                    end_point=end_pos,
                    obstacles=obstacles,
                    robot=robot_config
                )
                ax.plot(
                    positions_inches[:, 0], positions_inches[:, 1],
                    linestyle=':', linewidth=1.5, color=color, alpha=0.4
                )
            except Exception:
                ax.plot(
                    [start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                    linestyle=':', linewidth=1.5, color=color, alpha=0.4
                )
    
    def _render_robots_and_info(
        self, 
        ax, 
        ax_info, 
        actions: Optional[Dict], 
        rewards: Optional[Dict]
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.transforms as mtransforms

        """Render robots and delegate info panel to game."""
        # Draw robots on field
        for i, agent in enumerate(self.agents):
            st = self.environment_state["agents"][agent]
            team = self.game.get_team_for_agent(agent)
            robot_color = 'red' if team == 'red' else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            theta = st["orientation"][0]
            
            robot_len, robot_wid = self.game.get_robot_dimensions(
                agent
            )
            
            # Draw robot rectangle
            robot_rect = patches.Rectangle(
                (-robot_len/2, -robot_wid/2),
                robot_len, robot_wid,
                edgecolor='black', facecolor=robot_color,
                alpha=0.7, linewidth=2
            )
            t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
            robot_rect.set_transform(t)
            ax.add_patch(robot_rect)
            
            # Draw orientation arrow
            arrow_length = robot_len * 0.4
            ax.arrow(x, y,
                    np.cos(theta) * arrow_length,
                    np.sin(theta) * arrow_length,
                    width=1.5, facecolor='yellow',
                    head_width=4, head_length=2, zorder=10,
                    edgecolor='black', linewidth=0.5)
            
            # Agent number label
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center',
                   color='white', fontweight='bold', zorder=11,
                   bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.8))
        
        # Prepare agent times map
        agent_times = {
            agent: self.env_agent_states[agent]["time"] if agent in self.env_agent_states else 0.0
            for agent in self.agents
        }

        # Delegate info panel rendering to game
        self.game.render_info_panel(
            ax_info=ax_info,
            agents=self.agents,
            actions=actions,
            rewards=rewards,
            num_moves=self.num_moves,
            agent_times=agent_times
        )
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def clearStepsDirectory(self) -> None:
        """Clear the steps directory for new renders."""
        steps_dir = os.path.join(self.output_directory, "steps")
        if os.path.exists(steps_dir):
            for filename in os.listdir(steps_dir):
                os.remove(os.path.join(steps_dir, filename))
    
    def createGIF(self) -> None:
        """Create a GIF from rendered steps."""
        steps_dir = os.path.join(self.output_directory, "steps")
        if not os.path.exists(steps_dir):
            return
        
        images = []
        files = sorted(
            os.listdir(steps_dir),
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        import imageio.v2 as imageio

        for filename in files:
            images.append(imageio.imread(os.path.join(steps_dir, filename)))
        
        if images:
            imageio.mimsave(
                os.path.join(self.output_directory, "simulation.gif"),
                images,
                fps=10
            )
