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
        enable_communication: bool = False,
    ):
        """
        Initialize the VEX environment.
        
        Args:
            game: VexGame instance defining game-specific mechanics (with robots)
            render_mode: 'human' for display, 'rgb_array' or 'all' to save frames
            output_directory: Directory for saving renders
            randomize: Whether to randomize initial positions
            enable_communication: Whether to enable agent-to-agent communication
        """
        super().__init__()
        
        self.game = game
        self.enable_communication = enable_communication
            
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
                    # Keep any already-held blocks centered on the robot.
                    if hasattr(self.game, "update_robot_position"):
                        try:
                            self.game.update_robot_position(agent, agent_state["position"])
                        except Exception:
                            pass
                    del self.busy_state[agent]
                    infos[agent]["action_completed"] = True
                    # Apply any pending events scheduled for this agent's action
                    if hasattr(self.game, "apply_pending_events"):
                        try:
                            self.game.apply_pending_events(agent)
                        except Exception:
                            pass
                else:
                    # Interpolate using a potentially multi-segment plan
                    total = busy_info["total_ticks"]
                    rem = busy_info["remaining_ticks"]
                    elapsed = total - rem
                    plan = busy_info.get("plan", [])

                    current_pos = busy_info["start_pos"].copy()
                    current_orient = busy_info["start_orient"].copy()

                    # Find active segment by elapsed ticks
                    for seg in plan:
                        seg_start = seg["tick_start"]
                        seg_end = seg["tick_end"]
                        if elapsed <= seg_end:
                            seg_ticks = max(1, seg_end - seg_start)
                            seg_alpha = (elapsed - seg_start) / seg_ticks
                            seg_alpha = float(np.clip(seg_alpha, 0.0, 1.0))

                            s_pos = seg["start_pos"]
                            e_pos = seg["end_pos"]
                            current_pos = s_pos + (e_pos - s_pos) * seg_alpha

                            s_or = float(seg["start_orient"][0])
                            e_or = float(seg["end_orient"][0])
                            d_or = np.arctan2(np.sin(e_or - s_or), np.cos(e_or - s_or))
                            interp_or = s_or + d_or * seg_alpha

                            # If actually moving in this segment, face movement direction.
                            diff = e_pos - s_pos
                            dist = np.linalg.norm(diff)
                            if dist > 0.1:
                                interp_or = np.arctan2(diff[1], diff[0])

                            current_orient = np.array([interp_or], dtype=np.float32)
                            break
                        else:
                            # Segment fully completed
                            current_pos = seg["end_pos"].copy()
                            current_orient = seg["end_orient"].copy()

                    self.game.update_robot_position(agent, current_pos)
                    agent_state["position"] = current_pos
                    agent_state["orientation"] = current_orient
            
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
            # Provide current synchronized match time to game action logic.
            agent_state["game_time"] = self.env_agent_states.get(agent, {}).get("time", 0.0)
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

            # Build interpolation plan:
            # 1) consume action-authored plan, if provided by game.execute_action()
            # 2) otherwise use generic game hook fallback
            raw_plan = None
            if hasattr(self.game, "consume_last_interpolation_plan"):
                try:
                    raw_plan = self.game.consume_last_interpolation_plan(agent)
                except Exception:
                    raw_plan = None

            if raw_plan is None:
                raw_plan = self.game.get_interpolation_plan(
                    agent=agent,
                    action=action_int,
                    start_pos=start_pos,
                    start_orient=start_orient,
                    target_pos=target_pos,
                    target_orient=target_orient,
                    duration=duration,
                )

            # Normalize plan into tick-based segments
            plan = []
            cursor_pos = start_pos.copy()
            cursor_orient = start_orient.copy()
            tick_cursor = 0
            for seg in raw_plan or []:
                seg_duration = float(max(0.0, seg.get("duration", 0.0)))
                if seg_duration <= 0.0:
                    continue
                seg_ticks = max(1, int(np.ceil(seg_duration / DELTA_T)))
                end_pos = np.array(seg.get("target_pos", cursor_pos), dtype=np.float32).copy()
                end_orient = np.array(seg.get("target_orient", cursor_orient), dtype=np.float32).copy()

                plan.append({
                    "start_pos": cursor_pos.copy(),
                    "end_pos": end_pos,
                    "start_orient": cursor_orient.copy(),
                    "end_orient": end_orient,
                    "tick_start": tick_cursor,
                    "tick_end": tick_cursor + seg_ticks,
                })

                tick_cursor += seg_ticks
                cursor_pos = end_pos.copy()
                cursor_orient = end_orient.copy()

            # Fallback to single segment if plan is empty
            if not plan:
                seg_ticks = max(1, int(np.ceil(max(0.0, duration) / DELTA_T)))
                plan = [{
                    "start_pos": start_pos.copy(),
                    "end_pos": target_pos.copy(),
                    "start_orient": start_orient.copy(),
                    "end_orient": target_orient.copy(),
                    "tick_start": 0,
                    "tick_end": seg_ticks,
                }]
                tick_cursor = seg_ticks

            # Prepare Busy State
            ticks = max(1, int(tick_cursor))
            final_target_pos = plan[-1]["end_pos"].copy()
            final_target_orient = plan[-1]["end_orient"].copy()
            
            if ticks > 0:
                # Revert visible state to start
                agent_state["position"] = start_pos
                agent_state["orientation"] = start_orient
                
                self.busy_state[agent] = {
                    "start_pos": start_pos,
                    "start_orient": start_orient,
                    "target_pos": final_target_pos,
                    "target_orient": final_target_orient,
                    "plan": plan,
                    "total_ticks": ticks,
                    "remaining_ticks": ticks
                }

                # Update movement trail
                if not np.array_equal(start_pos, final_target_pos):
                    self.agent_movements[agent] = (start_pos.copy(), final_target_pos.copy())
            else:
                self.agent_movements[agent] = None
            
            # Rewards
            new_scores = self.game.compute_score()
            reward = self.game.compute_reward(agent, initial_scores, new_scores, penalty)
            rewards[agent] = reward

        # 3a. Projection-based collision check (prevent collisions using next-step projections)
        # Compute projected next-step positions for all agents and immediately
        # terminate any agents whose projections would collide.
        try:
            self._resolve_projected_collisions(active_agents, infos)
        except Exception:
            # Fail-safe: do not break the step if projection logic errors
            pass

        # Apply projected-collision penalties (environment-level)
        for agent in active_agents:
            if infos.get(agent, {}).get("action_terminated_early", False):
                # Subtract a collision penalty from the agent's reward so it matches
                # game-level collision semantics. compute_reward already subtracts
                # penalties returned by execute_action; here we apply the env-level
                # projected collision penalty using the same value as the game.
                try:
                    penalty_value = float(self.game.get_collision_penalty())
                except Exception:
                    penalty_value = 0.0
                rewards[agent] = rewards.get(agent, 0.0) - penalty_value

        # Advance synchronized match clock for all active agents.
        # Busy/idle/action differences are modeled via busy_state ticks,
        # but game time remains globally aligned across agents.
        for agent in active_agents:
            if agent in self.env_agent_states:
                self.env_agent_states[agent]["time"] += DELTA_T

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

    

    def _project_agent_next_position(self, agent: str) -> np.ndarray:
        """
        Project the position this agent is trying to reach on the next step.

        - If the agent is busy (has a queued target), return its `target_pos`.
        - Otherwise return its current `position` (projection within robot).

        This projection is used for collision checks that operate on planned
        next-step positions instead of the robot's current footprint.
        """
        if self.environment_state is None:
            raise RuntimeError("Environment state is not initialized")

        agent_state = self.environment_state["agents"].get(agent)
        if agent_state is None:
            raise KeyError(f"Unknown agent: {agent}")

        busy = self.busy_state.get(agent)
        if busy is not None:
            proj = busy["target_pos"].copy()
        else:
            proj = agent_state["position"].copy()

        agent_state["projected_position"] = proj
        return proj

    def _resolve_projected_collisions(self, active_agents: list, infos: Dict[str, Dict]) -> None:
        """
        Project all agents' intended next-step positions and force immediate
        termination of any agents whose projections collide.

        - Agents with `busy_state` use `busy_state["target_pos"]` as projection.
        - Stationary agents project to their current center (within robot).
        - If two projections overlap (distance < sum of radii) both agents are
          forced to terminate their current action immediately (busy_state cleared)
          and `infos[agent]["action_terminated_early"] = True` is set.
        """
        # Build projected positions
        projections: Dict[str, np.ndarray] = {}
        for agent in active_agents:
            try:
                projections[agent] = self._project_agent_next_position(agent)
            except Exception:
                projections[agent] = self.environment_state["agents"][agent]["position"].copy()

        # Pairwise collision check on projected positions
        colliding_pairs: set[tuple[str, str]] = set()
        agents_list = list(active_agents)
        for i in range(len(agents_list)):
            a = agents_list[i]
            pa = projections[a]
            ra = (self.game.get_robot_for_agent(a).size.value / 2.0) if self.game.get_robot_for_agent(a) else 9.0
            for j in range(i + 1, len(agents_list)):
                b = agents_list[j]
                pb = projections[b]
                rb = (self.game.get_robot_for_agent(b).size.value / 2.0) if self.game.get_robot_for_agent(b) else 9.0
                if float(np.linalg.norm(pa - pb)) < (ra + rb):
                    colliding_pairs.add((a, b))

        # Enforce immediate termination for any agent involved in a projected collision
        impacted: set[str] = set()
        for a, b in colliding_pairs:
            impacted.add(a)
            impacted.add(b)

        for agent in impacted:
            # Cancel busy action (if any) so robot will not continue moving
            if agent in self.busy_state:
                try:
                    del self.busy_state[agent]
                except KeyError:
                    pass
                self.agent_movements[agent] = None
                infos.setdefault(agent, {})["action_terminated_early"] = True
            else:
                # Not busy but projection conflicts with someone else
                infos.setdefault(agent, {})["projection_conflict"] = True

            # Mark flag on state for tests/logging
            try:
                self.environment_state["agents"][agent]["projected_collision"] = True
            except Exception:
                pass

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
            
            # Draw projected position (if available) as a dashed outline rectangle
            proj = st.get("projected_position", None)
            if proj is not None:
                px, py = float(proj[0]), float(proj[1])
                proj_theta = float(theta)
                if agent in self.busy_state:
                    try:
                        proj_theta = float(self.busy_state[agent]["target_orient"][0])
                    except Exception:
                        proj_theta = float(theta)
                proj_rect = patches.Rectangle(
                    (-robot_len/2, -robot_wid/2),
                    robot_len, robot_wid,
                    edgecolor='yellow', facecolor='none',
                    linestyle='--', linewidth=1.2, alpha=0.8, zorder=2
                )
                tproj = mtransforms.Affine2D().rotate(proj_theta).translate(px, py) + ax.transData
                proj_rect.set_transform(tproj)
                ax.add_patch(proj_rect)

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

        # Prepare per-agent remaining action time map (seconds)
        action_time_remaining = {
            agent: (
                max(0, int(self.busy_state[agent].get("remaining_ticks", 0))) * DELTA_T
                if agent in self.busy_state else 0.0
            )
            for agent in self.agents
        }

        # Delegate info panel rendering to game
        self.game.render_info_panel(
            ax_info=ax_info,
            agents=self.agents,
            actions=actions,
            rewards=rewards,
            num_moves=self.num_moves,
            agent_times=agent_times,
            action_time_remaining=action_time_remaining,
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
