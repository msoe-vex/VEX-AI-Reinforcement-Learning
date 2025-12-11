"""
VEX Core - Generic Multi-Agent Environment

Provides the base infrastructure for VEX robotics reinforcement learning.
Delegates game-specific logic to a VexGame implementation.
"""

import functools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import imageio.v2 as imageio

from gymnasium import spaces
from pettingzoo import ParallelEnv
from ray.rllib.env import MultiAgentEnv
from typing import Dict, List, Tuple, Optional, Any

from .base_game import VexGame, Robot, RobotSize, Team

from typing import Optional


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
        self.score = self.game.compute_score(self.environment_state)
        
        observations = {
            agent: self.game.get_observation(agent, self.environment_state)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        if not actions:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
        
        self.num_moves += 1
        
        # Keep all agents in self.agents until all are terminated
        active_agents = list(self.agents)
        
        # Check which agents are terminated
        terminated_agents = {
            agent for agent in active_agents
            if self.game.is_agent_terminated(agent, self.environment_state)
        }
        
        # If all agents are terminated, end the episode
        if len(terminated_agents) == len(active_agents):
            terminations = {agent: True for agent in active_agents}
            terminations["__all__"] = True
            truncations = {agent: False for agent in active_agents}
            truncations["__all__"] = False
            observations = {
                agent: self.game.get_observation(agent, self.environment_state)
                for agent in active_agents
            }
            rewards = {agent: 0.0 for agent in active_agents}
            infos = {agent: {} for agent in active_agents}
            return observations, rewards, terminations, truncations, infos
        
        rewards = {agent: 0.0 for agent in active_agents}
        
        # Get current minimum game time for action synchronization (only from non-terminated agents)
        non_terminated_agents = [a for a in active_agents if a not in terminated_agents]
        if non_terminated_agents:
            min_game_time = min(
                self.environment_state["agents"][agent]["gameTime"]
                for agent in non_terminated_agents
            )
        else:
            min_game_time = 0
        
        total_time = self.game.total_time
        
        for agent, action in actions.items():
            if agent not in active_agents:
                continue
                
            agent_state = self.environment_state["agents"][agent]
            
            # Skip terminated agents
            if agent in terminated_agents:
                agent_state["action_skipped"] = True
                continue
            
            # Skip agent if their time is ahead
            if agent_state["gameTime"] > min_game_time:
                agent_state["action_skipped"] = True
                continue
            
            agent_state["action_skipped"] = False
            
            # Store initial position for path tracking
            initial_pos = agent_state["position"].copy()
            
            # Get scores before action
            initial_scores = self.game.compute_score(self.environment_state)
            
            # Execute action through game
            action_int = int(action.value if hasattr(action, 'value') else action)
            duration, penalty = self.game.execute_action(
                agent, action_int, self.environment_state
            )
            
            agent_state["gameTime"] += duration
            
            # Calculate reward via game (allows variant overrides)
            new_scores = self.game.compute_score(self.environment_state)
            reward = self.game.compute_reward(agent, initial_scores, new_scores, penalty)
            
            rewards[agent] = reward
            
            # Track movement for rendering
            final_pos = agent_state["position"]
            if not np.array_equal(initial_pos, final_pos):
                self.agent_movements[agent] = (initial_pos.copy(), final_pos.copy())
            else:
                self.agent_movements[agent] = None
        
        # Update total score (stored as property)
        self.score = self.game.compute_score(self.environment_state)
        
        # Check terminations AFTER actions to see which agents just terminated
        new_terminations = {
            agent: self.game.is_agent_terminated(agent, self.environment_state)
            for agent in active_agents
        }
        
        # Determine which agents were newly terminated this step
        newly_terminated = set()
        for agent in active_agents:
            if agent not in terminated_agents and new_terminations[agent]:
                newly_terminated.add(agent)
        
        # Only return observations/rewards/infos for non-terminated or newly terminated
        agents_to_return = [a for a in active_agents if a not in terminated_agents or a in newly_terminated]
        
        terminations = {agent: new_terminations[agent] for agent in agents_to_return}
        terminations["__all__"] = all(new_terminations.values())
        
        truncations = {agent: False for agent in agents_to_return}
        truncations["__all__"] = False
        
        # Return observations only for agents that should receive them
        observations = {
            agent: self.game.get_observation(agent, self.environment_state)
            for agent in agents_to_return
        }
        
        # Only return rewards/infos for agents receiving observations
        filtered_rewards = {agent: rewards.get(agent, 0.0) for agent in agents_to_return}
        infos = {agent: {} for agent in agents_to_return}
        
        return observations, filtered_rewards, terminations, truncations, infos
    
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
        self.game.render_game_elements(ax, self.environment_state)
        
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
        """Render robots and info panel."""
        info_y = 0.95
        ax_info.text(0.5, info_y, "Agent Actions", fontsize=12, fontweight='bold',
                    ha='center', va='top')
        info_y -= 0.08
        
        for i, agent in enumerate(self.agents):
            st = self.environment_state["agents"][agent]
            team = self.game.get_team_for_agent(agent)
            robot_color = 'red' if team == 'red' else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            theta = st["orientation"][0]
            
            robot_len, robot_wid = self.game.get_robot_dimensions(
                agent, self.environment_state
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
            
            # Info panel text
            action_text = "---"
            if actions and agent in actions:
                if st.get("action_skipped", False):
                    action_text = "--"
                else:
                    try:
                        action_text = self.game.action_to_name(actions[agent])
                    except Exception:
                        action_text = str(actions[agent])
            
            reward_text = ""
            if rewards and agent in rewards:
                reward_text = f" (R: {rewards[agent]:.2f})"
            
            ax_info.text(0.05, info_y, f"Robot {i} ({team}):",
                        fontsize=9, color=robot_color, fontweight='bold', va='top')
            info_y -= 0.05
            ax_info.text(0.1, info_y, f"{action_text}{reward_text}", fontsize=8, va='top')
            info_y -= 0.03
            ax_info.text(0.1, info_y, 
                        f"Time: {st['gameTime']:.1f}s / {self.game.total_time:.0f}s",
                        fontsize=7, va='top', color='gray')
            info_y -= 0.03
            ax_info.text(0.1, info_y,
                        f"Pos: ({x:.0f}, {y:.0f}) | Held: {st.get('held_blocks', 0)}",
                        fontsize=7, va='top', color='gray')
            info_y -= 0.06
        
        # Score section
        info_y -= 0.02
        ax_info.axhline(y=info_y, xmin=0.05, xmax=0.95, color='gray', linewidth=0.5)
        info_y -= 0.05
        
        team_scores = self.game.compute_score(self.environment_state)
        ax_info.text(0.05, info_y, "Scores:", fontsize=10, fontweight='bold', va='top')
        info_y -= 0.04
        ax_info.text(0.1, info_y, f"Red: {team_scores.get('red', 0)}",
                    fontsize=9, va='top', color='red', fontweight='bold')
        info_y -= 0.04
        ax_info.text(0.1, info_y, f"Blue: {team_scores.get('blue', 0)}",
                    fontsize=9, va='top', color='blue', fontweight='bold')
        info_y -= 0.05
        ax_info.text(0.05, info_y, f"Step: {self.num_moves}", fontsize=8, va='top')
    
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
        for filename in files:
            images.append(imageio.imread(os.path.join(steps_dir, filename)))
        
        if images:
            imageio.mimsave(
                os.path.join(self.output_directory, "simulation.gif"),
                images,
                fps=10
            )
