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

from .base_game import VexGame, Robot, RobotSize, Team, ActionEvent, ActionStep

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
        deterministic: bool = True,
    ):
        """
        Initialize the VEX environment.
        
        Args:
            game: VexGame instance defining game-specific mechanics (with robots)
            render_mode: 'image' (render + print), 'terminal' (print only), or 'none' (silent)
            output_directory: Directory for saving renders
            randomize: Whether to randomize initial positions
            enable_communication: Whether to enable agent-to-agent communication
            deterministic: Whether environment/game stochastic mechanics are disabled
        """
        super().__init__()
        
        self.game = game
        self.enable_communication = enable_communication
        self.deterministic = deterministic
        if hasattr(self.game, "deterministic"):
            self.game.deterministic = deterministic
            
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
        self.num_ticks = 0  # Internal tick counter
        self.score = 0
        self.agent_movements: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        
        # Busy state for discrete time steps
        # Maps agent -> {
        #   "start_pos": np.array, "target_pos": np.array,
        #   "target_orient": np.array,
        #   "total_ticks": int, "remaining_ticks": int
        # }
        self.busy_state: Dict[str, Dict] = {}
        self._deferred_rewards: Dict[str, Dict] = {}  # {agent: {penalty, action_name, accumulated_reward}}
        self.projected_positions: Dict[str, np.ndarray] = {}
        
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
        self.num_ticks = 0
        self._terminated_agents = set()
        self.num_steps = 0  # Counter for external environment steps
        self.agent_movements = {agent: None for agent in self.possible_agents}
        self.busy_state = {}
        self._deferred_rewards = {}
        self.projected_positions = {}
        
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
            agent: {"tick": 0} for agent in self.agents
        }

        # Populate initial inter-agent messages before first observation
        self._update_messages(list(self.agents))
        
        observations = {
            agent: self.game.get_observation(agent, game_time=0.0)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def _tick_busy_agents(self) -> Dict[str, bool]:
        """Tick all busy agents by one step. Returns dict of agent->completed."""
        completed = {}
        for agent in list(self.busy_state.keys()):
            busy_info = self.busy_state[agent]
            busy_info["remaining_ticks"] -= 1

            agent_state = self.environment_state["agents"][agent]

            total = busy_info["total_ticks"]
            rem = busy_info["remaining_ticks"]
            elapsed = total - rem
            plan = busy_info.get("plan", [])

            # Apply events for any segments that just completed this tick
            completed_segments = busy_info.get("completed_segments", set())
            for seg_idx, seg in enumerate(plan):
                if seg_idx not in completed_segments and elapsed >= seg["tick_end"]:
                    completed_segments.add(seg_idx)
                    seg_events = seg.get("events", [])
                    if seg_events:
                        score_before = self.game.compute_score()
                        try:
                            self.game.apply_events(agent, seg_events)
                        except Exception:
                            pass
                        score_after = self.game.compute_score()
                        if agent in self._deferred_rewards:
                            delta = self.game.compute_reward(agent, score_before, score_after, 0.0)
                            self._deferred_rewards[agent]["accumulated_reward"] += delta
            busy_info["completed_segments"] = completed_segments

            if busy_info["remaining_ticks"] <= 0:
                # Action complete - snap to target
                agent_state["position"] = busy_info["target_pos"].copy()
                agent_state["orientation"] = busy_info["target_orient"].copy()
                if hasattr(self.game, "update_robot_position"):
                    try:
                        self.game.update_robot_position(agent, agent_state["position"])
                    except Exception:
                        pass
                del self.busy_state[agent]
                completed[agent] = True
            else:
                # Interpolate position within multi-segment plan
                current_pos = busy_info["start_pos"].copy()
                current_orient = busy_info["start_orient"].copy()
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
                        diff = e_pos - s_pos
                        if np.linalg.norm(diff) > 0.1:
                            interp_or = np.arctan2(diff[1], diff[0])
                        current_orient = np.array([interp_or], dtype=np.float32)
                        break
                    else:
                        current_pos = seg["end_pos"].copy()
                        current_orient = seg["end_orient"].copy()
                self.game.update_robot_position(agent, current_pos)
                agent_state["position"] = current_pos
                agent_state["orientation"] = current_orient
                completed[agent] = False
        return completed

    def _advance_time_and_messages(self):
        """Advance game clock by one tick and update inter-agent messages."""
        for agent in self.possible_agents:
            if agent in self.env_agent_states and agent not in self._terminated_agents:
                self.env_agent_states[agent]["tick"] += 1
        self._update_messages(list(self.possible_agents))

    def _get_agent_time(self, agent: str) -> float:
        """Get agent-local game time in seconds from integer ticks."""
        ticks = int(self.env_agent_states.get(agent, {}).get("tick", 0))
        return ticks * DELTA_T

    def _check_terminations(self) -> Tuple[Dict[str, bool], set]:
        """Check if any agents should be terminated."""
        terminations = {}
        agents_to_remove = set()
        for agent in self.possible_agents:
            current_time = self._get_agent_time(agent)
            term = self.game.is_agent_terminated(agent, game_time=current_time)
            terminations[agent] = term
            if term:
                agents_to_remove.add(agent)
        return terminations, agents_to_remove

    def step(
        self, 
        actions: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step.

        Only agents that are NOT busy appear in self.agents.  RLlib/the caller
        sends actions only for those agents.  After processing the new actions
        (which make agents busy), we fast-forward through ticks until at
        least one agent becomes free again.  Each internal tick renders a
        frame and updates messages so communication is never lost.

        Rewards are computed AFTER the busy state completes (events have been
        applied and scores have changed), giving correct score-based rewards.
        """
        # Remove any previously terminated agents so we don't expect actions from them
        for agent in self._terminated_agents:
            if agent in self.agents:
                self.agents.remove(agent)
                
        if not actions and not self.agents:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
        
        self.num_ticks += 1
        
        if self.environment_state is None:
            print("CRITICAL: self.environment_state is None in step()!")
        
        all_agents = [
            agent
            for agent in self.possible_agents
            if agent not in self._terminated_agents and agent in self.environment_state.get("agents", {})
        ]
        
        rewards = {agent: 0.0 for agent in all_agents}
        infos = {agent: {} for agent in all_agents}
        
        # ──────────────────────────────────────────────────────────────
        # 1. Process new actions for non-busy agents
        # ──────────────────────────────────────────────────────────────
        newly_actioned = set()
        for agent, action in actions.items():
            if agent not in all_agents:
                continue
            if agent in self.busy_state:
                continue

            agent_state = self.environment_state["agents"][agent]
            infos[agent]["action_skipped"] = False
            
            start_pos = agent_state["position"].copy()
            start_orient = agent_state["orientation"].copy()
            score_before_action = self.game.compute_score()
            # Parse action tuple (Control, Message)
            action_val = action
            emitted_msg = None
            if isinstance(action, (tuple, list)) and len(action) >= 2:
                action_val = action[0]
                emitted_msg = action[1]
            elif isinstance(action, np.ndarray) and action.ndim > 0 and action.size > 1:
                pass
                
            action_int = int(action_val.value if hasattr(action_val, 'value') else action_val)
            agent_state["current_action"] = action_int
            agent_state["game_time"] = self._get_agent_time(agent)
            action_steps, penalty = self.game.execute_action(agent, action_int)
            score_after_start = self.game.compute_score()
            immediate_action_reward = self.game.compute_reward(
                agent,
                score_before_action,
                score_after_start,
                0.0,
            )
            
            # Store emitted message
            if emitted_msg is not None:
                arr = np.array(emitted_msg, dtype=np.float32).ravel()
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

            # Build tick-based interpolation plan
            plan = []
            cursor_pos = start_pos.copy()
            cursor_orient = start_orient.copy()
            tick_cursor = 0
            for step in action_steps:
                seg_duration = float(max(0.0, step.duration))
                if seg_duration <= 0.0:
                    continue
                seg_ticks = max(1, int(np.ceil(seg_duration / DELTA_T)))
                end_pos = np.array(step.target_pos, dtype=np.float32).copy()
                end_orient = np.array(step.target_orient, dtype=np.float32).copy()
                plan.append({
                    "start_pos": cursor_pos.copy(),
                    "end_pos": end_pos,
                    "start_orient": cursor_orient.copy(),
                    "end_orient": end_orient,
                    "tick_start": tick_cursor,
                    "tick_end": tick_cursor + seg_ticks,
                    "events": step.events,
                })
                tick_cursor += seg_ticks
                cursor_pos = end_pos.copy()
                cursor_orient = end_orient.copy()

            if not plan:
                seg_ticks = 1
                plan = [{
                    "start_pos": start_pos.copy(),
                    "end_pos": start_pos.copy(),
                    "start_orient": start_orient.copy(),
                    "end_orient": start_orient.copy(),
                    "tick_start": 0,
                    "tick_end": seg_ticks,
                    "events": [],
                }]
                tick_cursor = seg_ticks

            ticks = max(1, int(tick_cursor))
            final_target_pos = plan[-1]["end_pos"].copy()
            final_target_orient = plan[-1]["end_orient"].copy()
            
            if ticks > 0:
                self.busy_state[agent] = {
                    "start_pos": start_pos,
                    "start_orient": start_orient,
                    "target_pos": final_target_pos,
                    "target_orient": final_target_orient,
                    "plan": plan,
                    "total_ticks": ticks,
                    "remaining_ticks": ticks,
                    "completed_segments": set(),
                }
                newly_actioned.add(agent)
                if not np.array_equal(start_pos, final_target_pos):
                    self.agent_movements[agent] = (start_pos.copy(), final_target_pos.copy())
            else:
                self.agent_movements[agent] = None

            # Store for deferred reward computation (include action name for info panel)
            try:
                action_name = self.game.action_to_name(action_int)
            except Exception:
                action_name = str(action_int)
            self._deferred_rewards[agent] = {
                "penalty": penalty,
                "action_name": action_name,
                "accumulated_reward": immediate_action_reward,
            }

            # Agent is now busy — remove from self.agents
            if agent in self.agents:
                self.agents.remove(agent)

        # ──────────────────────────────────────────────────────────────
        # 2. Projected collision check (only newly started actions)
        # ──────────────────────────────────────────────────────────────
        failed_agents: set[str] = set()
        try:
            failed_agents = self._resolve_projected_collisions(all_agents, newly_actioned)
        except Exception:
            failed_agents = set()

        for agent in failed_agents:
            busy = self.busy_state.get(agent)
            if busy is None:
                continue
            start_pos = busy["start_pos"].copy()
            start_orient = busy["start_orient"].copy()
            self.busy_state[agent] = {
                "start_pos": start_pos,
                "start_orient": start_orient,
                "target_pos": start_pos.copy(),
                "target_orient": start_orient.copy(),
                "plan": [{
                    "start_pos": start_pos.copy(),
                    "end_pos": start_pos.copy(),
                    "start_orient": start_orient.copy(),
                    "end_orient": start_orient.copy(),
                    "tick_start": 0,
                    "tick_end": 1,
                    "events": [],
                }],
                "total_ticks": 1,
                "remaining_ticks": 1,
                "completed_segments": set(),
            }
            self.agent_movements[agent] = None
            try:
                penalty_value = float(self.game.get_collision_penalty())
            except Exception:
                penalty_value = 0.0
            stored = self._deferred_rewards.get(agent)
            if stored is not None:
                stored["penalty"] = float(stored.get("penalty", 0.0)) + penalty_value

        # ──────────────────────────────────────────────────────────────
        # 3. Advance time + messages for this tick
        # ──────────────────────────────────────────────────────────────
        tick_results = self._tick_busy_agents()
        for agent, did_complete in tick_results.items():
            if did_complete:
                stored = self._deferred_rewards.pop(agent, None)
                if stored is not None:
                    reward = stored.get("accumulated_reward", 0.0) - stored.get("penalty", 0.0)
                    rewards[agent] = reward
                    self.environment_state["agents"][agent]["last_action_name"] = stored.get("action_name", "--")
                    self.environment_state["agents"][agent]["last_action_reward"] = reward
                infos[agent]["action_completed"] = True
                if agent not in self.agents:
                    self.agents.append(agent)

        self._advance_time_and_messages()
        self.score = self.game.compute_score()

        self.render(action_dict=actions, rewards=rewards, infos=infos)

        # ──────────────────────────────────────────────────────────────
        # 4. Fast-forward while ALL agents are busy
        # ──────────────────────────────────────────────────────────────
        while (not self.agents) and any(a in self.busy_state for a in all_agents):
            self.num_ticks += 1

            tick_results = self._tick_busy_agents()
            for agent, did_complete in tick_results.items():
                if did_complete:
                    stored = self._deferred_rewards.pop(agent, None)
                    if stored is not None:
                        reward = stored.get("accumulated_reward", 0.0) - stored.get("penalty", 0.0)
                        rewards[agent] = reward
                        self.environment_state["agents"][agent]["last_action_name"] = stored.get("action_name", "--")
                        self.environment_state["agents"][agent]["last_action_reward"] = reward
                    infos[agent]["action_completed"] = True
                    if agent not in self.agents:
                        self.agents.append(agent)

            self._advance_time_and_messages()
            self.score = self.game.compute_score()

            self.render(rewards=rewards, infos=infos)

            # Check terminations during fast-forward
            _, ff_remove = self._check_terminations()
            for agent in ff_remove:
                if agent not in self._terminated_agents:
                    self._terminated_agents.add(agent)
                    # Add them to self.agents so RLlib sees their final terminating state
                    if agent not in self.agents:
                        self.agents.append(agent)
                    if agent in self.busy_state:
                        del self.busy_state[agent]
            if all(
                self.game.is_agent_terminated(
                    a, game_time=self._get_agent_time(a)
                )
                for a in all_agents
            ):
                break

        # ──────────────────────────────────────────────────────────────
        # 5. Terminations and observations
        # ──────────────────────────────────────────────────────────────
        terminations, agents_to_remove = self._check_terminations()
        for agent in agents_to_remove:
            if agent not in self._terminated_agents:
                self._terminated_agents.add(agent)
                # Keep them in self.agents for one final return to RLlib
                if agent not in self.agents:
                    self.agents.append(agent)
                if agent in self.busy_state:
                    del self.busy_state[agent]
        
        # Build strictly filtered observation dicts only for returning agents
        observations = {
            agent: self.game.get_observation(
                agent,
                game_time=self._get_agent_time(agent)
            )
            for agent in self.agents
        }
        
        rewards_out = {agent: rewards.get(agent, 0.0) for agent in self.agents}
        terminations_out = {agent: terminations.get(agent, False) for agent in self.agents}
        truncations_out = {agent: False for agent in self.agents}
        infos_out = {agent: infos.get(agent, {}) for agent in self.agents}
        
        # Game is over when all agents have terminated OR no free agents and no busy agents
        all_done = (len(self._terminated_agents) == len(self.possible_agents)) or \
                   (len(self.agents) == 0 and not any(a in self.busy_state for a in all_agents))

        terminations_out["__all__"] = all_done
        truncations_out["__all__"] = False
        
        self.num_steps += 1
        
        return observations, rewards_out, terminations_out, truncations_out, infos_out


    def _update_messages(self, active_agents: List[str]) -> None:
        """
        Aggregate messages from neighbors for each agent.
        Updates agent_state["received_messages"].
        """
        
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
            received_sum = np.zeros(8, dtype=np.float32)
            count = 0
            
            for other in active_agents:
                if agent == other:
                    continue
                
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

        return proj

    def _resolve_projected_collisions(self, active_agents: List[str], newly_actioned: Optional[set] = None) -> set:
        """
        Project all agents' intended next-step positions and convert any
        NEWLY STARTED conflicting actions into one-tick failed actions.

        Only agents in `newly_actioned` (those that started a new action this
        step) are marked as impacted. Agents already mid-action from
        a previous step continue uninterrupted.

        - Agents with `busy_state` use `busy_state["target_pos"]` as projection.
        - Stationary agents project to their current center (within robot).
                - If two projections overlap (distance < sum of radii), impacted
                    agents are returned to be converted into one-tick failures by step().
        """
        if newly_actioned is None:
            newly_actioned = set()

        # Build projected positions
        projections: Dict[str, np.ndarray] = {}
        for agent in active_agents:
            try:
                projections[agent] = self._project_agent_next_position(agent)
            except Exception:
                projections[agent] = self.environment_state["agents"][agent]["position"].copy()
        self.projected_positions = projections

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

        # Only cancel agents that JUST started a new action this step
        impacted: set[str] = set()
        for a, b in colliding_pairs:
            if a in newly_actioned:
                impacted.add(a)
            if b in newly_actioned:
                impacted.add(b)

        return impacted

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
        action_dict: Optional[Dict] = None, 
        rewards: Optional[Dict] = None,
        infos: Optional[Dict] = None
    ) -> None:
        """Render the current environment state to an image and/or terminal."""
        # Centralized console output logic
        if self.render_mode in ["terminal", "image"]:
            has_started = action_dict is not None and len(action_dict) > 0
            has_completed = infos is not None and any(i.get("action_completed", False) for i in infos.values())
            
            if has_started or has_completed:
                current_step = self.num_steps + 1
                print(
                    f"\nStep {current_step} | Tick {self.num_ticks}: "
                    f"Time {self.num_ticks * DELTA_T:.1f}s | Scores: {self.score}"
                )
                
                if has_started:
                    for agent, action in action_dict.items():
                        if isinstance(action, (tuple, list, np.ndarray)):
                            action_val = action[0]
                        else:
                            action_val = action
                        action_name = self.game.get_action_name(int(action_val))
                        print(f"  {agent}: STARTED {action_name}")
                
                if has_completed:
                    for agent, info in infos.items():
                        if info.get("action_completed", False):
                            reward = rewards.get(agent, 0.0)
                            action_name = self.environment_state["agents"][agent].get("last_action_name", "--")
                            reward_str = f"  (r={reward:.2f})"
                            
                            print(f"  {agent}: COMPLETED {action_name}{reward_str}")

        if self.render_mode != "image":
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
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        
        # Draw auto line
        ax.plot([-field_half, field_half], [0, 0], color='#d0d0d0', linewidth=2)
        
        # Render game-specific elements
        self.game.render_game_elements(ax)
        
        # Draw robot paths
        self._render_paths(ax)
        
        # Draw robots and info panel
        self._render_robots_and_info(ax, ax_info, action_dict, rewards)
        
        # Title
        ax.set_title("VEX Environment", fontsize=14, fontweight='bold')
        
        # Save or display
        if self.render_mode == "human":
            plt.show()
        else:
            os.makedirs(os.path.join(self.output_directory, "ticks"), exist_ok=True)
            plt.savefig(
                os.path.join(self.output_directory, "ticks", f"tick_{self.num_ticks}.png"),
                dpi=100
            )
            plt.close()
    
    def _render_paths(self, ax) -> None:
        """Render robot paths."""
        if self.path_planner is None:
            return
        
        for agent in self.possible_agents:
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
        for i, agent in enumerate(self.possible_agents):
            st = self.environment_state["agents"][agent]
            team = self.game.get_team_for_agent(agent)
            robot_color = 'red' if team == 'red' else 'blue'
            
            x, y = st["position"][0], st["position"][1]
            theta = st["orientation"][0]
            
            robot_len, robot_wid = self.game.get_robot_dimensions(
                agent
            )
            
            # Draw projected position (if available) as a dashed outline rectangle
            proj = self.projected_positions.get(agent)
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
            agent: self._get_agent_time(agent)
            for agent in self.possible_agents
        }

        # Prepare per-agent remaining action time map (seconds)
        action_time_remaining = {
            agent: (
                max(0, int(self.busy_state[agent].get("remaining_ticks", 0))) * DELTA_T
                if agent in self.busy_state else 0.0
            )
            for agent in self.possible_agents
        }

        # Delegate info panel rendering to game
        self.game.render_info_panel(
            ax_info=ax_info,
            agents=self.possible_agents,
            actions=actions,
            rewards=rewards,
            num_steps=self.num_steps + 1,
            agent_times=agent_times,
            action_time_remaining=action_time_remaining,
        )
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def clearTicksDirectory(self) -> None:
        """Clear the ticks directory for new renders."""
        ticks_dir = os.path.join(self.output_directory, "ticks")
        if os.path.exists(ticks_dir):
            for filename in os.listdir(ticks_dir):
                os.remove(os.path.join(ticks_dir, filename))
    
    def createGIF(self) -> None:
        """Create a GIF from rendered ticks."""
        ticks_dir = os.path.join(self.output_directory, "ticks")
        if not os.path.exists(ticks_dir):
            return
        
        images = []
        files = sorted(
            os.listdir(ticks_dir),
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        import imageio.v2 as imageio

        for filename in files:
            img = imageio.imread(os.path.join(ticks_dir, filename))
            # Flatten alpha channel onto white background for GIF compatibility
            # (GIF only supports 1-bit transparency; semi-transparent pixels get clipped)
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                composited = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
                img = composited
            images.append(img)
        
        if images:
            imageio.mimsave(
                os.path.join(self.output_directory, "simulation.gif"),
                images,
                fps=10
            )

    def clearStepsDirectory(self) -> None:
        """Backward-compatible alias for clearTicksDirectory()."""
        self.clearTicksDirectory()
