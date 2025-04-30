import functools

import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.test import parallel_api_test
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from path_planner import PathPlanner, Obstacle

import numpy as np
from enum import Enum
import os

NUM_ITERS = 100

class Actions(Enum):
    PICK_UP_NEAREST_GOAL = 0
    PICK_UP_NEAREST_RING = 1
    CLIMB = 2
    DROP_GOAL = 3
    DRIVE_TO_CORNER_BL = 4
    DRIVE_TO_CORNER_BR = 5
    DRIVE_TO_CORNER_TL = 6
    DRIVE_TO_CORNER_TR = 7
    ADD_RING_TO_GOAL = 8
    DRIVE_TO_WALL_STAKE_L = 9
    DRIVE_TO_WALL_STAKE_R = 10
    DRIVE_TO_WALL_STAKE_B = 11
    DRIVE_TO_WALL_STAKE_T = 12
    ADD_RING_TO_WALL_STAKE = 13
    TURN_TOWARDS_CENTER = 14
    PICK_UP_NEXT_NEAREST_GOAL = 15

# Constants
INCHES_PER_FIELD = 144
ENV_FIELD_SIZE = 12
BUFFER_RADIUS = 2 # buffer around robot for collision detection
NUM_WALL_STAKES = 4
NUM_GOALS = 5
NUM_RINGS = 24
TIME_LIMIT = 60
DEFAULT_PENALTY = -0.1

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if (render_mode == "ansi"):
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = High_Stakes_Multi_Agent_Env(render_mode=render_mode)
    # env = parallel_to_aec(env)
    return env


class High_Stakes_Multi_Agent_Env(MultiAgentEnv, ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "vex_high_stakes"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = ["robot_0", "robot_1"]  # Define all possible agents
        self._agent_ids = self.possible_agents
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode
        self.agents = []  # Initialize as empty; will be set in reset()
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.output_directory = "run_petting_zoo"

        self.wall_stakes_positions = [
            np.array([0, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE/2, 0]),
            np.array([ENV_FIELD_SIZE/2, ENV_FIELD_SIZE])
        ]

        self.permanent_obstacles = [
                    Obstacle(3/6, 2/6, 3.5/INCHES_PER_FIELD, False), # Bottom
                    Obstacle(3/6, 4/6, 3.5/INCHES_PER_FIELD, False), # Top
                    Obstacle(2/6, 3/6, 3.5/INCHES_PER_FIELD, False), # Left
                    Obstacle(4/6, 3/6, 3.5/INCHES_PER_FIELD, False) # Right
                    ]
        self.climb_positions = np.array([
            [(self.permanent_obstacles[0].x + self.permanent_obstacles[2].x) / 2 * ENV_FIELD_SIZE,
             (self.permanent_obstacles[0].y + self.permanent_obstacles[2].y) / 2 * ENV_FIELD_SIZE],
            [(self.permanent_obstacles[0].x + self.permanent_obstacles[3].x) / 2 * ENV_FIELD_SIZE,
             (self.permanent_obstacles[0].y + self.permanent_obstacles[3].y) / 2 * ENV_FIELD_SIZE],
            [(self.permanent_obstacles[1].x + self.permanent_obstacles[2].x) / 2 * ENV_FIELD_SIZE,
             (self.permanent_obstacles[1].y + self.permanent_obstacles[2].y) / 2 * ENV_FIELD_SIZE],
            [(self.permanent_obstacles[1].x + self.permanent_obstacles[3].x) / 2 * ENV_FIELD_SIZE,
             (self.permanent_obstacles[1].y + self.permanent_obstacles[3].y) / 2 * ENV_FIELD_SIZE]
        ])
        self.initial_climb_positions = self.climb_positions + np.array([
            [np.cos(np.arctan2(self.permanent_obstacles[2].y - self.permanent_obstacles[1].y, self.permanent_obstacles[2].x - self.permanent_obstacles[1].x)) * 1,
             np.sin(np.arctan2(self.permanent_obstacles[2].y - self.permanent_obstacles[1].y, self.permanent_obstacles[2].x - self.permanent_obstacles[1].x)) * 1],
            [np.cos(np.arctan2(self.permanent_obstacles[3].y - self.permanent_obstacles[1].y, self.permanent_obstacles[3].x - self.permanent_obstacles[1].x)) * 1,
             np.sin(np.arctan2(self.permanent_obstacles[3].y - self.permanent_obstacles[1].y, self.permanent_obstacles[3].x - self.permanent_obstacles[1].x)) * 1],
            [np.cos(np.arctan2(self.permanent_obstacles[2].y - self.permanent_obstacles[0].y, self.permanent_obstacles[2].x - self.permanent_obstacles[0].x)) * 1,
             np.sin(np.arctan2(self.permanent_obstacles[2].y - self.permanent_obstacles[0].y, self.permanent_obstacles[2].x - self.permanent_obstacles[0].x)) * 1],
            [np.cos(np.arctan2(self.permanent_obstacles[3].y - self.permanent_obstacles[0].y, self.permanent_obstacles[3].x - self.permanent_obstacles[0].x)) * 1,
             np.sin(np.arctan2(self.permanent_obstacles[3].y - self.permanent_obstacles[0].y, self.permanent_obstacles[3].x - self.permanent_obstacles[0].x)) * 1]
        ])
        self.realistic_vision = True
        self.robot_num = 0
        self.robot_length = 15 # inches
        self.robot_width = 15 # inches
        self.robot_radius = np.sqrt( \
                                (self.robot_length*(ENV_FIELD_SIZE/INCHES_PER_FIELD))**2 + \
                                (self.robot_width*(ENV_FIELD_SIZE/INCHES_PER_FIELD))**2
                            ) / 2 + BUFFER_RADIUS*(ENV_FIELD_SIZE/INCHES_PER_FIELD)
        self.initial_robot_position = np.array([1, 6.0], dtype=np.float32)
        self.initial_robot_orientation = 0.0
        self.mobile_goal_positions = np.array([[4.0, 2.0], [4.0, 10.0], [8.0, 4.0], [8.0, 8.0], [10.0, 6.0]], dtype=np.float32)
        self.ring_positions = np.array([
            [6.0, 1.0], [6.0, 11.0],
            [2.0, 2.0], [2.0, 6.0], [2.0, 10.0], [4.0, 4.0], [4.0, 8.0],
            [5.7, 5.7], [5.7, 6.3], [6.3, 5.7], [6.3, 6.3],
            [6.0, 2.0], [6.0, 10.0], [8.0, 2.0], [8.0, 10.0], [10.0, 2.0], [10.0, 10.0],
            [1, 1], [11, 11], [11, 1], [1, 11],
            [10.0, 4.0], [10.0, 8.0], [11.0, 6.0]
        ], dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            "position": spaces.Box(low=np.float32(0), high=np.float32(ENV_FIELD_SIZE), shape=(2,), dtype=np.float32),
            "robot_orientation": spaces.Box(low=np.float32(-np.pi), high=np.float32(np.pi), shape=(1,), dtype=np.float32),
            "holding_goal": spaces.Discrete(2),
            "holding_rings": spaces.Discrete(3),
            "rings": spaces.Box(low=np.float32(-1), high=np.float32(ENV_FIELD_SIZE), shape=(NUM_RINGS * 2,), dtype=np.float32),
            "goals": spaces.Box(low=np.float32(-1), high=np.float32(ENV_FIELD_SIZE), shape=(NUM_GOALS * 2,), dtype=np.float32),
            "wall_stake_ring_count": spaces.Box(low=np.int32(0), high=np.int32(6), shape=(NUM_WALL_STAKES,), dtype=np.int32),
            "holding_goal_full": spaces.Discrete(2),
            "time_remaining": spaces.Box(low=np.float32(0), high=np.float32(TIME_LIMIT), shape=(1,), dtype=np.float32),
            "visible_rings_count": spaces.Discrete(NUM_RINGS + 1),
            "visible_goals_count": spaces.Discrete(NUM_GOALS + 1)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Actions))

    def observation_space_sample(self, agent_ids=None):
        agent_ids = agent_ids or self.get_agent_ids()
        return {agent_id: self.observation_space(agent_id).sample() for agent_id in agent_ids}

    def action_space_sample(self, agent_ids=None):
        agent_ids = agent_ids or self.get_agent_ids()
        return {agent_id: self.action_space(agent_id).sample() for agent_id in agent_ids}

    def observation_space_contains(self, observations):
        return all(
            agent_id in self.get_agent_ids() and
            self.observation_space(agent_id).contains(obs)
            for agent_id, obs in observations.items()
        )

    def action_space_contains(self, actions):
        return all(
            agent_id in self.get_agent_ids() and
            self.action_space(agent_id).contains(action)
            for agent_id, action in actions.items()
        )

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]  # Reset agents to all possible agents
        self.num_moves = 0
        self.environment_state = self._get_initial_environment_state()
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _get_initial_environment_state(self):
        ring_positions = [
            {"position": np.array([6.0, 1.0], dtype=np.float32), "status": 0}, {"position": np.array([6.0, 11.0], dtype=np.float32), "status": 0},
            {"position": np.array([2.0, 2.0], dtype=np.float32), "status": 0}, {"position": np.array([2.0, 6.0], dtype=np.float32), "status": 0},
            {"position": np.array([2.0, 10.0], dtype=np.float32), "status": 0}, {"position": np.array([4.0, 4.0], dtype=np.float32), "status": 0},
            {"position": np.array([4.0, 8.0], dtype=np.float32), "status": 0}, {"position": np.array([5.7, 5.7], dtype=np.float32), "status": 0},
            {"position": np.array([5.7, 6.3], dtype=np.float32), "status": 0}, {"position": np.array([6.3, 5.7], dtype=np.float32), "status": 0},
            {"position": np.array([6.3, 6.3], dtype=np.float32), "status": 0}, {"position": np.array([6.0, 2.0], dtype=np.float32), "status": 0},
            {"position": np.array([6.0, 10.0], dtype=np.float32), "status": 0}, {"position": np.array([8.0, 2.0], dtype=np.float32), "status": 0},
            {"position": np.array([8.0, 10.0], dtype=np.float32), "status": 0}, {"position": np.array([10.0, 2.0], dtype=np.float32), "status": 0},
            {"position": np.array([10.0, 10.0], dtype=np.float32), "status": 0}, {"position": np.array([1.0, 1.0], dtype=np.float32), "status": 0},
            {"position": np.array([11.0, 11.0], dtype=np.float32), "status": 0}, {"position": np.array([11.0, 1.0], dtype=np.float32), "status": 0},
            {"position": np.array([1.0, 11.0], dtype=np.float32), "status": 0}, {"position": np.array([10.0, 4.0], dtype=np.float32), "status": 0},
            {"position": np.array([10.0, 8.0], dtype=np.float32), "status": 0}, {"position": np.array([11.0, 6.0], dtype=np.float32), "status": 0},
            ]
        goal_positions = [
            {"position": np.array([4.0, 2.0], dtype=np.float32), "status": 0},
            {"position": np.array([4.0, 10.0], dtype=np.float32), "status": 0},
            {"position": np.array([8.0, 4.0], dtype=np.float32), "status": 0},
            {"position": np.array([8.0, 8.0], dtype=np.float32), "status": 0},
            {"position": np.array([10.0, 6.0], dtype=np.float32), "status": 0},
            ]
        return {
            "agents": {
            agent: {
                "position": np.array([1.0, 6.0], dtype=np.float32),
                "orientation": np.array([0.0], dtype=np.float32),
                "holding_goal_index": -1,
                "held_rings": 0,
                "size": 15,
                "climbed": False,
                "gameTime": 0,
            }
            for agent in self.agents
            },
            "goals": goal_positions,
            "rings": ring_positions,
        }

    def get_observation(self, agent):
        agent_state = self.environment_state["agents"][agent]
        rings = self.get_available_rings(agent_state)
        goals = self.get_available_goals(agent_state)
        return {
            "position": np.clip(agent_state["position"].astype(np.float32), 0, np.float32(ENV_FIELD_SIZE)),
            "robot_orientation": np.clip(agent_state["orientation"].astype(np.float32), np.float32(-np.pi), np.float32(np.pi)),
            "holding_goal": np.int32(0) if agent_state["holding_goal_index"] == -1 else np.int32(1),
            "holding_rings": np.int32(agent_state["held_rings"]),
            "rings": np.clip(np.array(rings, dtype=np.float32).flatten(), np.float32(-1), np.float32(ENV_FIELD_SIZE)),
            "goals": np.clip(np.array(goals, dtype=np.float32).flatten(), np.float32(-1), np.float32(ENV_FIELD_SIZE)),
            "wall_stake_ring_count": np.array([
            np.int32(sum(1 for ring in self.environment_state["rings"] if ring["status"] == stake_index + 7))
            for stake_index in range(NUM_WALL_STAKES)
            ], dtype=np.int32),
            "holding_goal_full": np.int32(0) if np.int32(sum(
            1 for ring in self.environment_state["rings"] if ring["status"] == agent_state["holding_goal_index"] + 2
            )) < 6 else np.int32(1),
            "time_remaining": np.clip(np.array([TIME_LIMIT - agent_state["gameTime"]], dtype=np.float32), np.float32(0), np.float32(TIME_LIMIT)),
            "visible_rings_count": np.int32(np.count_nonzero(rings != -1) // 2),
            "visible_goals_count": np.int32(np.count_nonzero(goals != -1) // 2)
        }

    def step(self, actions):
        # Ensure actions are provided for all agents
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Increment move counter
        self.num_moves += 1

        rewards = {agent: 0 for agent in self.agents} # Initialize rewards for each agent

        # TODO: Implement logic for terminating the episode
        # Initialize terminations and truncations
        terminations = {agent: self.num_moves > NUM_ITERS for agent in self.agents}
        terminations["__all__"] = all(terminations.values())

        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = all(truncations.values())
        
        minGameTime = min([self.environment_state["agents"][agent]["gameTime"] for agent in self.agents])

        # Perform actions for each agent
        # TODO: penalize robot for colliding with other robot
        for agent, action in actions.items():
            agent_state = self.environment_state["agents"][agent]

            if agent_state["climbed"]:
                # If the robot has climbed, it cannot perform any actions
                rewards[agent] = DEFAULT_PENALTY
                continue
            if agent_state["gameTime"] > minGameTime:
                continue # skip this agent if it is ahead of the others

            initial_score = self.compute_field_score()
            
            penalty = 0
            duration = 0.1
            if action == Actions.PICK_UP_NEAREST_GOAL.value:
                if agent_state["holding_goal_index"] == -1:
                    # Find the nearest available goal
                    nearest_goal = None
                    min_distance = float('inf')
                    for i, goal in enumerate(self.environment_state["goals"]):
                        if self.is_goal_available(i, agent_state):  # Check if the goal is available
                            distance = np.linalg.norm(agent_state["position"] - goal["position"])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_goal = i

                    # Pick up the nearest available goal if found
                    if nearest_goal is not None:
                        agent_state["holding_goal_index"] = nearest_goal
                        self.environment_state["goals"][nearest_goal]["status"] = self.agent_name_mapping[agent] + 1
                        
                        # Update robot orientation to face the goal
                        goal_position = self.environment_state["goals"][nearest_goal]["position"]
                        direction_vector = goal_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)

                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance

                        # Update robot position to the goal's position
                        agent_state["position"] = self.environment_state["goals"][nearest_goal]["position"].copy()
                    else:
                        penalty = DEFAULT_PENALTY  # No available goal to pick up
                else:
                    penalty = DEFAULT_PENALTY  # Already holding a goal

            elif action == Actions.PICK_UP_NEAREST_RING.value:
                # Check if the robot can hold more rings
                if agent_state["held_rings"] < 2:
                    # Find the nearest available ring
                    nearest_ring = None
                    min_distance = float('inf')
                    for i, ring in enumerate(self.environment_state["rings"]):
                        if self.is_ring_available(i, agent_state):  # Check if the ring is available
                            distance = np.linalg.norm(agent_state["position"] - ring["position"])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_ring = i

                    # Pick up the nearest available ring if found
                    if nearest_ring is not None:
                        self.environment_state["rings"][nearest_ring]["status"] = self.agent_name_mapping[agent] + 1
                        agent_state["held_rings"] += 1

                        # Update robot orientation to face the ring
                        ring_position = self.environment_state["rings"][nearest_ring]["position"]
                        direction_vector = ring_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)
                        
                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance

                        # Update robot position to the ring's position
                        agent_state["position"] = self.environment_state["rings"][nearest_ring]["position"].copy()
                    else:
                        penalty = DEFAULT_PENALTY  # No available ring to pick up
                else:
                    penalty = DEFAULT_PENALTY # Already holding maximum rings

            # TODO: Implement other climbing logic
            elif action == Actions.CLIMB.value:
                if agent_state["holding_goal_index"] != -1:
                    penalty = DEFAULT_PENALTY # Cannot climb while holding a goal
                else:
                    # Find the nearest climb position
                    nearest_climb_position = self.climb_positions[np.argmin(np.linalg.norm(self.climb_positions - agent_state["position"], axis=1))]
                    
                    # Calculate distance traveled and update duration
                    distance = np.linalg.norm(nearest_climb_position - agent_state["position"])
                    duration += distance
                    
                    agent_state["position"] = nearest_climb_position
                    direction_to_center = np.array([ENV_FIELD_SIZE / 2, ENV_FIELD_SIZE / 2]) - nearest_climb_position
                    agent_state["orientation"] = np.array([np.arctan2(direction_to_center[1], direction_to_center[0])], dtype=np.float32)
                    agent_state["climbed"] = True
            
            elif action == Actions.DROP_GOAL.value:
                if agent_state["holding_goal_index"] != -1:
                    goal_index = agent_state["holding_goal_index"]
                    self.environment_state["goals"][goal_index]["status"] = 0  # Goal is now available
                    agent_state["holding_goal_index"] = -1
                    duration += 0.5  # Add duration for dropping the goal
                else:
                    penalty = DEFAULT_PENALTY  # No goal to drop

            elif Actions.DRIVE_TO_CORNER_BL.value <= action <= Actions.DRIVE_TO_CORNER_TR.value:
                corner_index = action - Actions.DRIVE_TO_CORNER_BL.value
                if corner_index == 0:  # BL
                    target_position = np.array([1.0, 1.0], dtype=np.float32)
                elif corner_index == 1:  # BR
                    target_position = np.array([ENV_FIELD_SIZE - 1.0, 1.0], dtype=np.float32)
                elif corner_index == 2:  # TL
                    target_position = np.array([1.0, ENV_FIELD_SIZE - 1.0], dtype=np.float32)
                elif corner_index == 3:  # TR
                    target_position = np.array([ENV_FIELD_SIZE - 1.0, ENV_FIELD_SIZE - 1.0], dtype=np.float32)

                # Calculate distance traveled and update duration
                distance = np.linalg.norm(target_position - agent_state["position"])
                duration += distance

                # Update agent position and orientation
                agent_state["position"] = target_position
                agent_state["orientation"] = np.array([np.arctan2(target_position[1] - agent_state["position"][1], target_position[0] - agent_state["position"][0])], dtype=np.float32)

            elif action == Actions.ADD_RING_TO_GOAL.value:
                if agent_state["holding_goal_index"] != -1 and agent_state["held_rings"] > 0:
                    goal_index = agent_state["holding_goal_index"]
                    # Find a ring held by the agent
                    for i, ring in enumerate(self.environment_state["rings"]):
                        if ring["status"] - 1 == self.agent_name_mapping[agent]:
                            ring["status"] = goal_index + 2
                            agent_state["held_rings"] -= 1
                            break
                    duration += 0.5  # Add duration for adding ring to goal
                else:
                    penalty = DEFAULT_PENALTY  # No goal or rings to add

            elif Actions.DRIVE_TO_WALL_STAKE_L.value <= action <= Actions.DRIVE_TO_WALL_STAKE_T.value:
                stake_index = action - Actions.DRIVE_TO_WALL_STAKE_L.value
                if stake_index == 0:  # L
                    target_position = np.array([1.0, ENV_FIELD_SIZE / 2], dtype=np.float32)
                    agent_state["orientation"] = np.array([np.pi], dtype=np.float32)
                elif stake_index == 1:  # R
                    target_position = np.array([ENV_FIELD_SIZE - 1.0, ENV_FIELD_SIZE / 2], dtype=np.float32)
                    agent_state["orientation"] = np.array([0.0], dtype=np.float32)
                elif stake_index == 2:  # B
                    target_position = np.array([ENV_FIELD_SIZE / 2, 1.0], dtype=np.float32)
                    agent_state["orientation"] = np.array([-np.pi / 2], dtype=np.float32)
                elif stake_index == 3:  # T
                    target_position = np.array([ENV_FIELD_SIZE / 2, ENV_FIELD_SIZE - 1.0], dtype=np.float32)
                    agent_state["orientation"] = np.array([np.pi / 2], dtype=np.float32)

                # Calculate distance traveled and update duration
                distance = np.linalg.norm(target_position - agent_state["position"])
                duration += distance

                # Update agent position
                agent_state["position"] = target_position

            elif action == Actions.ADD_RING_TO_WALL_STAKE.value:
                 # Check if the robot is near a wall stake and has rings to add
                if agent_state["held_rings"] > 0:
                    # Determine which stake is closest
                    distances = [np.linalg.norm(agent_state["position"] - self.wall_stakes_positions[i]) for i in range(len(self.wall_stakes_positions))]
                    closest_stake_index = np.argmin(distances)

                    # Check if the robot is close enough to the stake
                    if distances[closest_stake_index] < 1.5:
                        # Find a ring held by the agent
                        for i, ring in enumerate(self.environment_state["rings"]):
                            if ring["status"] - 1 == self.agent_name_mapping[agent]:
                                # Add ring to the wall stake
                                ring["status"] = closest_stake_index + 7  # Assign ring to the closest wall stake
                                agent_state["held_rings"] -= 1
                                break
                        duration += 1  # Add duration for adding ring to wall stake
                    else:
                        penalty = DEFAULT_PENALTY  # Not close enough to a wall stake
                else:
                    penalty = DEFAULT_PENALTY  # No rings to add

            elif action == Actions.TURN_TOWARDS_CENTER.value:
                center = np.array([ENV_FIELD_SIZE / 2, ENV_FIELD_SIZE / 2], dtype=np.float32)
                direction_vector = center - agent_state["position"]
                agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)
                duration += 1  # Add duration for turning towards center

            elif action == Actions.PICK_UP_NEXT_NEAREST_GOAL.value:
                if agent_state["holding_goal_index"] != -1:
                    # Find the next nearest available goal
                    available_goals = []
                    for i, goal in enumerate(self.environment_state["goals"]):
                        if self.is_goal_available(i, agent_state):  # Check if the goal is available
                            available_goals.append((i, np.linalg.norm(agent_state["position"] - goal["position"])))

                    if len(available_goals) > 1:
                        # Sort goals by distance
                        available_goals = sorted(available_goals, key=lambda x: x[1])
                        next_nearest_goal_index = available_goals[1][0]  # Get the second nearest goal

                        # Update agent state to pick up the next nearest goal
                        agent_state["holding_goal_index"] = next_nearest_goal_index
                        self.environment_state["goals"][next_nearest_goal_index]["status"] = self.agent_name_mapping[agent] + 1

                        # Update robot orientation to face the goal
                        goal_position = self.environment_state["goals"][next_nearest_goal_index]["position"]
                        direction_vector = goal_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)

                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance

                        # Update robot position to the goal's position
                        agent_state["position"] = self.environment_state["goals"][next_nearest_goal_index]["position"].copy()
                    else:
                        penalty = DEFAULT_PENALTY  # Not enough goals available
                else:
                    penalty = DEFAULT_PENALTY  # No goal to pick up
            
            agent_state["gameTime"] += duration
            rewards[agent] = self.reward_function(initial_score, penalty)
            
            # Update agent state
            self.environment_state["agents"][agent] = agent_state

        # Update positions of goals and rings if held by an agent
        for agent, agent_state in self.environment_state["agents"].items():
            # Update goal position if held by the agent
            if agent_state["holding_goal_index"] != -1:
                goal_index = agent_state["holding_goal_index"]
                self.environment_state["goals"][goal_index]["position"] = agent_state["position"].copy()

            # Update ring positions if held by the agent
            for i, ring in enumerate(self.environment_state["rings"]):
                if ring["status"] - 1 == self.agent_name_mapping[agent]:
                    ring["position"] = agent_state["position"].copy()

        # TODO: Implement time synchronization issues

        infos = {agent: {} for agent in self.agents}

        # Update observations based on the updated environment state
        observations = {agent: self.get_observation(agent) for agent in self.agents}

        # Remove agents if the environment is truncated or terminated
        if terminations["__all__"] or truncations["__all__"]:
            self.agents = []
        else:
            # Filter out terminated agents # TODO: Figure out how to handle terminations with agents
            self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

        return observations, rewards, terminations, truncations, infos

    def render(self, actions=None, rewards=None):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        mode = self.render_mode
        if mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if mode == "terminal" or mode == "all":
            if len(self.agents) > 0:
                print("Step:", self.num_moves)
            for agent in self.agents:
                print(f"\t{agent}:")
                if actions is not None:
                    print(f"\t\tAction: {Actions(actions[agent]).name}")
                if rewards is not None:
                    print(f"\t\tReward: {rewards[agent]}")
                for attribute, value in self.environment_state['agents'][agent].items():
                    print(f"\t\t{attribute}: {value}")
            print()
        if mode == "image" or mode == "all":
            self.save_image(self.num_moves, actions)
    
    def save_image(self, step_num, actions=None):
        # Create visualization
            fig, ax = plt.subplots(figsize=(10,8))
            ax.set_xlim(0, ENV_FIELD_SIZE)
            ax.set_ylim(0, ENV_FIELD_SIZE)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
            
            # Draw the rings
            for i, ring in enumerate(self.environment_state["rings"]):
                visible_by_any_agent = any(self.is_ring_available(i, self.environment_state["agents"][agent]) for agent in self.agents)
                color = 'red' if visible_by_any_agent else 'gray'
                circle = patches.Circle((ring["position"][0], ring["position"][1]), 0.3, color=color, alpha=0.7)
                ax.add_patch(circle)

            # Draw the goals
            for goal_idx, goal in enumerate(self.environment_state["goals"]):
                visible_by_any_agent = any(self.is_goal_available(goal_idx, self.environment_state["agents"][agent]) for agent in self.agents)
                color = 'green' if visible_by_any_agent else 'gray'
                hexagon = patches.RegularPolygon((goal["position"][0], goal["position"][1]), numVertices=6, radius=0.5, orientation=np.pi/6, color=color, alpha=0.25)
                ax.add_patch(hexagon)
                rings_on_goal = 0
                ax.text(goal["position"][0], goal["position"][1] + 0.6, str(rings_on_goal), color='black', ha='center')

            # Draw the wall stakes
            for idx, pos in enumerate(self.wall_stakes_positions):
                if idx == 0:
                    text_pos = pos + np.array([-0.5, 0.0])
                elif idx == 1:
                    text_pos = pos + np.array([0.5, 0.0])
                elif idx == 2:
                    text_pos = pos + np.array([0.0, -0.5])
                elif idx == 3:
                    text_pos = pos + np.array([0.0, 0.5])
                ax.text(text_pos[0], text_pos[1], f"{0}", color='black', ha='center')

            # Draw the robots
            for agent in self.agents:
                x = self.environment_state["agents"][agent]["position"][0]
                y = self.environment_state["agents"][agent]["position"][1]
                orientation = self.environment_state["agents"][agent]["orientation"][0]
                width = self.environment_state["agents"][agent]["size"] / 12  # Convert inches to feet
                
                # Draw the rectangle representing the robot
                rect = patches.Rectangle(
                    (x - width / 2, y - width / 2), width, width, 
                    color='blue', alpha=0.5, ec='black'
                )
                t = patches.transforms.Affine2D().rotate_deg_around(x, y, np.degrees(orientation)) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)
                
                # Draw an arrow to indicate the orientation
                arrow_dx = np.cos(orientation) * width / 2
                arrow_dy = np.sin(orientation) * width / 2
                orientation_arrow = patches.FancyArrow(
                    x, y, arrow_dx, arrow_dy, width=0.1, color='yellow', length_includes_head=True, alpha=0.8
                )
                ax.add_patch(orientation_arrow)
                
                # Add agent label
                ax.text(x, y, agent, fontsize=12, ha='center', va='center')

            # Draw the obstacles
            for obstacle in self.permanent_obstacles:
                circle = patches.Circle(
                    (obstacle.x * ENV_FIELD_SIZE, obstacle.y * ENV_FIELD_SIZE), 
                    obstacle.radius * ENV_FIELD_SIZE, 
                    edgecolor='black', facecolor='none', 
                    linestyle='dotted', alpha=0.5)
                ax.add_patch(circle)
            
            for agent in self.agents:
                robot_position = self.environment_state["agents"][agent]["position"]
                robot_orientation = self.environment_state["agents"][agent]["orientation"][0]
                fov_length = 50
                left_fov_angle = robot_orientation + np.pi / 4
                right_fov_angle = robot_orientation - np.pi / 4
                left_fov_end = robot_position + fov_length * np.array([np.cos(left_fov_angle), np.sin(left_fov_angle)])
                right_fov_end = robot_position + fov_length * np.array([np.cos(right_fov_angle), np.sin(right_fov_angle)])
                ax.plot([robot_position[0], left_fov_end[0]], [robot_position[1], left_fov_end[1]], 'k--', alpha=0.5)
                ax.plot([robot_position[0], right_fov_end[0]], [robot_position[1], right_fov_end[1]], 'k--', alpha=0.5)
                overlay_polygon = np.array([
                    robot_position,
                    left_fov_end,
                    right_fov_end,
                    robot_position
                ])
                overlay = patches.Polygon(overlay_polygon, closed=True, color='yellow', alpha=0.1)
                ax.add_patch(overlay)
            
            total_score = 0
            ax.text(-2.5, 6, f"Total Score: {total_score}", color='black', ha='center')
            ax.text(6, 13.25, f'Step {step_num}', color='black', ha='center')

            if actions:
                action_str = ", ".join([f"{agent}: {Actions(actions[agent]).name}" for agent in self.agents])
                ax.text(6, -1, f"Actions: {action_str}", color='black', ha='center')

            os.makedirs(self.output_directory+"/steps", exist_ok=True)
            plt.savefig(f"{self.output_directory}/steps/step_{step_num}.png")
            plt.close()
    
    def clearStepsDirectory(self):
        """
        Clear the steps directory to remove old images.
        """
        output_dir = "run_petting_zoo/steps"
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reward_function(self, initial_score, penalty):
        new_score = self.compute_field_score()
        delta_score = new_score - initial_score
        reward = delta_score + penalty
        return reward

    def compute_field_score(self, skills=True):
        score = 0
        for goal_idx, goal in enumerate(self.environment_state["goals"]):
            rings_on_goal = sum(1 for ring in self.environment_state["rings"] if ring["status"] == goal_idx + 2)
            if rings_on_goal > 0:
                in_corner = any(np.linalg.norm(goal["position"] - corner) < 0.5 for corner in self.climb_positions)
                if(skills):
                    score += (3 + (rings_on_goal - 1)) + (5 if in_corner else 0)
                else:
                    score += (3 + (rings_on_goal - 1)) * (2 if in_corner else 1)

        for stake_idx, stake_pos in enumerate(self.wall_stakes_positions):
            rings_on_stake = sum(1 for ring in self.environment_state["rings"] if ring["status"] == stake_idx + 7)
            if rings_on_stake > 0:
                score += 3 + (rings_on_stake - 1)

        if self.environment_state["agents"][self.agents[0]]["climbed"]:
            score += 3

        self.total_score = score
        return score
    
    def is_goal_available(self, goal_index, agent_state):
        goal_position = self.environment_state["goals"][goal_index]["position"]
        if self.robot_num == 1 and goal_position[1] < ENV_FIELD_SIZE / 2:
            return False
        if self.robot_num == 2 and goal_position[1] >= ENV_FIELD_SIZE / 2:
            return False
        return self.is_visible(goal_position, agent_state) and self.environment_state["goals"][goal_index]["status"] == 0

    def is_ring_available(self, ring_index, agent_state):
        ring_position = self.environment_state["rings"][ring_index]["position"]
        if self.robot_num == 1 and ring_position[1] < ENV_FIELD_SIZE / 2:
            return False
        if self.robot_num == 2 and ring_position[1] >= ENV_FIELD_SIZE / 2:
            return False
        return self.is_visible(ring_position, agent_state) and self.environment_state["rings"][ring_index]["status"] == 0

    def is_visible(self, position, agent_state):
        if not self.realistic_vision:
            return True
        if position[0] < 0 or position[0] > ENV_FIELD_SIZE or position[1] < 0 or position[1] > ENV_FIELD_SIZE:
            return False
        direction = position - agent_state["position"]
        angle = np.arctan2(direction[1], direction[0])
        relative_angle = (angle - agent_state["orientation"] + np.pi) % (2 * np.pi) - np.pi
        return -np.pi / 4 <= relative_angle <= np.pi / 4

    def get_available_goals(self, agent_state):
        padded_goals = np.full((NUM_GOALS * 2,), -1, dtype=np.float32)
        visible_goals = []

        for i, goal in enumerate(self.environment_state["goals"]):
            if self.is_goal_available(i, agent_state):
                visible_goals.append(goal["position"])

        visible_goals = np.array(visible_goals).flatten()
        padded_goals[:len(visible_goals)] = visible_goals

        return padded_goals

    def get_available_rings(self, agent_state):
        padded_rings = np.full((NUM_RINGS * 2,), -1, dtype=np.float32)
        visible_rings = []

        for i, ring in enumerate(self.environment_state["rings"]):
            if self.is_ring_available(i, agent_state):
                visible_rings.append(ring["position"])

        visible_rings = np.array(visible_rings).flatten()
        padded_rings[:len(visible_rings)] = visible_rings

        return padded_rings

if __name__ == "__main__":
    env = High_Stakes_Multi_Agent_Env(render_mode="terminal")
    parallel_api_test(env)

    observations, infos = env.reset()

    env.clearStepsDirectory()

    done = False
    env.render()
    while not done:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        env.render(actions, rewards)
    env.close()