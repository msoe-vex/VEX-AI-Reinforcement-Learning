import functools

import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from ray.rllib.env import MultiAgentEnv

import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    from .path_planner import Obstacle, PathPlanner
except:
    from path_planner import Obstacle, PathPlanner

import numpy as np
from enum import Enum
import os
import imageio

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
FOV = np.pi / 2

POSSIBLE_AGENTS = ["robot_0", "robot_1"]

# Offsets for status IDs
AGENT_ID_OFFSET = 1
GOAL_ID_OFFSET = AGENT_ID_OFFSET + len(POSSIBLE_AGENTS)
STAKE_ID_OFFSET = GOAL_ID_OFFSET + NUM_GOALS

def env_creator(config=None):
    """
    Creates an instance of the High_Stakes_Multi_Agent_Env.
    Args:
        config (dict): Configuration dictionary for the environment.
    """
    config = config or {}  # Ensure config is a dictionary
    return High_Stakes_Multi_Agent_Env(render_mode=None, randomize=config.get("randomize", True))

class High_Stakes_Multi_Agent_Env(MultiAgentEnv, ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "vex_high_stakes"}

    def __init__(self, render_mode=None, output_directory="", randomize=True):
        super().__init__()
        self.possible_agents = POSSIBLE_AGENTS # Needed for PettingZoo and RlLib API
        self._agent_ids = POSSIBLE_AGENTS
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode
        self.agents = []  # Initialize as empty; will be set in reset()
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.output_directory = output_directory
        self.randomize = randomize

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
        self.score = 0
        self.path_planner = PathPlanner(
            robot_length=24/INCHES_PER_FIELD,
            robot_width=24/INCHES_PER_FIELD,
            buffer_radius=BUFFER_RADIUS/INCHES_PER_FIELD,
            max_velocity=80/INCHES_PER_FIELD,
            max_accel=100/INCHES_PER_FIELD)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation structure:
        # 1. Position (x, y): 2 elements
        # 2. Orientation: 1 element
        # 3. Holding Goal (boolean): 1 element
        # 4. Held Rings (count): 1 element
        # 5. Ring positions (x, y for each): NUM_RINGS * 2 elements
        # 6. Goal positions (x, y for each): NUM_GOALS * 2 elements
        # 7. Rings on wall stakes (count for each): NUM_WALL_STAKES elements
        # 8. Holding Goal Full (boolean): 1 element
        # 9. Time Remaining: 1 element
        # 10. Visible Rings Count: 1 element
        # 11. Visible Goals Count: 1 element
        # Total elements = 2+1+1+1 + (NUM_RINGS*2) + (NUM_GOALS*2) + NUM_WALL_STAKES + 1+1+1+1 = 71

        low = np.array(
            [0.0, 0.0, -np.pi, 0.0, 0.0] +  # Pos(2), Orient(1), HoldGoal(1), HeldRings(1)
            [-1.0] * (NUM_RINGS * 2) +
            [-1.0] * (NUM_GOALS * 2) +
            [0.0] * NUM_WALL_STAKES +
            [0.0, 0.0, 0.0, 0.0],  # HoldGoalFull(1), TimeRem(1), VisRings(1), VisGoals(1)
            dtype=np.float32
        )
        high = np.array(
            [float(ENV_FIELD_SIZE), float(ENV_FIELD_SIZE), np.pi, 1.0, 2.0] +  # Pos(2), Orient(1), HoldGoal(1), HeldRings(1)
            [float(ENV_FIELD_SIZE)] * (NUM_RINGS * 2) +
            [float(ENV_FIELD_SIZE)] * (NUM_GOALS * 2) +
            [6.0] * NUM_WALL_STAKES +
            [1.0, float(TIME_LIMIT), float(NUM_RINGS), float(NUM_GOALS)],  # HoldGoalFull(1), TimeRem(1), VisRings(1), VisGoals(1)
            dtype=np.float32
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

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
        if(self.randomize):
            self.environment_state = self._get_random_environment_state(seed)
        else:
            self.environment_state = self._get_initial_environment_state()
        observations = {agent: self._get_observation(agent) for agent in self.agents}
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
                    "active": True,
                }
                for agent in self.agents
            },
            "goals": goal_positions,
            "rings": ring_positions,
        }

    def _get_random_environment_state(self, seed=None):
        # Use the provided seed for reproducibility
        rng = np.random.default_rng(seed)
        ring_positions = [
            {"position": rng.random(2) * ENV_FIELD_SIZE, "status": 0}
            for _ in range(NUM_RINGS)
        ]
        goal_positions = [
            {"position": rng.random(2) * ENV_FIELD_SIZE, "status": 0}
            for _ in range(NUM_GOALS)
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
                "active": True,
            }
            for agent in self.agents
            },
            "goals": goal_positions,
            "rings": ring_positions,
        }

    def _get_observation(self, agent):
        agent_state = self.environment_state["agents"][agent]
        available_rings = self._get_available_rings(agent_state)
        available_goals = self._get_available_goals(agent_state)

        # Pad available objects with -1
        padded_rings = np.full((NUM_RINGS * 2,), -1, dtype=np.float32)
        if len(available_rings) > 0:
            flat_rings = available_rings.flatten()
            padded_rings[:len(flat_rings)] = flat_rings

        padded_goals = np.full((NUM_GOALS * 2,), -1, dtype=np.float32)
        if len(available_goals) > 0:
            flat_goals = available_goals.flatten()
            padded_goals[:len(flat_goals)] = flat_goals

        observation = np.concatenate([
            np.clip(agent_state["position"], 0, ENV_FIELD_SIZE),
            np.clip(agent_state["orientation"], -np.pi, np.pi),
            [0 if agent_state["holding_goal_index"] == -1 else 1],
            [np.clip(agent_state["held_rings"], 0, 2)],
            padded_rings,
            padded_goals,
            np.clip(
                np.array([
                    sum(1 for ring in self.environment_state["rings"] if ring["status"] == stake_index + STAKE_ID_OFFSET)
                    for stake_index in range(NUM_WALL_STAKES)
                ], dtype=np.float32),
                0, 6
            ),
            [0 if sum(1 for ring in self.environment_state["rings"] if ring["status"] == agent_state["holding_goal_index"] + GOAL_ID_OFFSET) < 6 else 1],
            [np.clip(TIME_LIMIT - agent_state["gameTime"], 0, TIME_LIMIT)],
            [np.clip(len(available_rings) // 2, 0, NUM_RINGS)],
            [np.clip(len(available_goals) // 2, 0, NUM_GOALS)]
        ])
        return observation.astype(np.float32)

    def step(self, actions):
        # Ensure actions are provided for all agents
        if not actions:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}
        
        # Increment move counter
        self.num_moves += 1

        rewards = {agent: 0 for agent in self.agents} # Initialize rewards for each agent
        
        minGameTime = min([self.environment_state["agents"][agent]["gameTime"] for agent in self.agents])

        # Initialize terminations and truncations
        terminations = {agent: minGameTime >= TIME_LIMIT for agent in self.agents}
        terminations["__all__"] = all(terminations.values())

        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = all(truncations.values())

        # Perform actions for each agent
        # TODO: penalize robot for colliding with other robot
        for agent, action in actions.items():
            agent_state = self.environment_state["agents"][agent]

            if agent_state["climbed"]:
                rewards[agent] = -1
                agent_state["active"] = False
                continue # If the robot has climbed, skip the rest of the actions
            if agent_state["gameTime"] > minGameTime:
                agent_state["active"] = False
                continue # skip this agent if it is ahead of the others

            initial_score = self._compute_field_score()

            agent_state["active"] = True
            
            penalty = 0
            duration = 0.1
            if action == Actions.PICK_UP_NEAREST_GOAL.value:
                if agent_state["holding_goal_index"] == -1:
                    # Find the nearest available goal
                    nearest_goal = None
                    min_distance = float('inf')
                    for i, goal in enumerate(self.environment_state["goals"]):
                        if self._is_goal_available(i, agent_state):  # Check if the goal is available
                            distance = np.linalg.norm(agent_state["position"] - goal["position"])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_goal = i

                    # Pick up the nearest available goal if found
                    if nearest_goal is not None:
                        agent_state["holding_goal_index"] = nearest_goal
                        self.environment_state["goals"][nearest_goal]["status"] = self.agent_name_mapping[agent] + AGENT_ID_OFFSET
                        
                        # Update robot orientation to face the goal
                        goal_position = self.environment_state["goals"][nearest_goal]["position"]
                        direction_vector = goal_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)

                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance / 2

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
                        if self._is_ring_available(i, agent_state):  # Check if the ring is available
                            distance = np.linalg.norm(agent_state["position"] - ring["position"])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_ring = i

                    # Pick up the nearest available ring if found
                    if nearest_ring is not None:
                        self.environment_state["rings"][nearest_ring]["status"] = self.agent_name_mapping[agent] + AGENT_ID_OFFSET
                        agent_state["held_rings"] += 1

                        # Update robot orientation to face the ring
                        ring_position = self.environment_state["rings"][nearest_ring]["position"]
                        direction_vector = ring_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)
                        
                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance / 2

                        # Update robot position to the ring's position
                        agent_state["position"] = self.environment_state["rings"][nearest_ring]["position"].copy()
                    else:
                        penalty = DEFAULT_PENALTY  # No available ring to pick up
                else:
                    penalty = DEFAULT_PENALTY # Already holding maximum rings

            elif action == Actions.CLIMB.value:
                if agent_state["holding_goal_index"] != -1:
                    penalty = DEFAULT_PENALTY # Cannot climb while holding a goal
                elif TIME_LIMIT - agent_state["gameTime"] < 10:
                    penalty = DEFAULT_PENALTY # Not enough time to climb
                else:
                    # Find the nearest climb position
                    nearest_climb_position = self.climb_positions[np.argmin(np.linalg.norm(self.climb_positions - agent_state["position"], axis=1))]
                    
                    # Calculate distance traveled and update duration
                    distance = np.linalg.norm(nearest_climb_position - agent_state["position"])
                    
                    agent_state["position"] = nearest_climb_position
                    direction_to_center = np.array([ENV_FIELD_SIZE / 2, ENV_FIELD_SIZE / 2]) - nearest_climb_position
                    agent_state["orientation"] = np.array([np.arctan2(direction_to_center[1], direction_to_center[0])], dtype=np.float32)
                    agent_state["climbed"] = True
                    duration = TIME_LIMIT - agent_state["gameTime"]  # Set duration to time remaining

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
                duration += distance / 2

                # Update agent position and orientation
                agent_state["orientation"] = np.array([np.arctan2(target_position[1] - agent_state["position"][1], target_position[0] - agent_state["position"][0])], dtype=np.float32)
                agent_state["position"] = target_position

            elif action == Actions.ADD_RING_TO_GOAL.value:
                if agent_state["holding_goal_index"] != -1 and agent_state["held_rings"] > 0:
                    goal_index = agent_state["holding_goal_index"]
                    # Find a ring held by the agent
                    for i, ring in enumerate(self.environment_state["rings"]):
                        if ring["status"] == self.agent_name_mapping[agent] + AGENT_ID_OFFSET:
                            ring["status"] = goal_index + GOAL_ID_OFFSET # Assign ring to the goal
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
                duration += distance / 2

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
                            if ring["status"] == self.agent_name_mapping[agent] + AGENT_ID_OFFSET:
                                # Add ring to the wall stake
                                ring["status"] = closest_stake_index + STAKE_ID_OFFSET  # Assign ring to the closest wall stake
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
                        if self._is_goal_available(i, agent_state):  # Check if the goal is available
                            available_goals.append((i, np.linalg.norm(agent_state["position"] - goal["position"])))

                    if len(available_goals) > 1:
                        # Sort goals by distance
                        available_goals = sorted(available_goals, key=lambda x: x[1])
                        next_nearest_goal_index = available_goals[1][0]  # Get the second nearest goal

                        # Update agent state to pick up the next nearest goal
                        agent_state["holding_goal_index"] = next_nearest_goal_index
                        self.environment_state["goals"][next_nearest_goal_index]["status"] = self.agent_name_mapping[agent] + AGENT_ID_OFFSET

                        # Update robot orientation to face the goal
                        goal_position = self.environment_state["goals"][next_nearest_goal_index]["position"]
                        direction_vector = goal_position - agent_state["position"]
                        agent_state["orientation"] = np.array([np.arctan2(direction_vector[1], direction_vector[0])], dtype=np.float32)

                        # Calculate distance traveled and update duration
                        distance = np.linalg.norm(direction_vector)
                        duration += distance / 2

                        # Update robot position to the goal's position
                        agent_state["position"] = self.environment_state["goals"][next_nearest_goal_index]["position"].copy()
                    else:
                        penalty = DEFAULT_PENALTY  # Not enough goals available
                else:
                    penalty = DEFAULT_PENALTY  # No goal to pick up
            
            # Removed collision detection, since no information is provided about the other robot's position
            # 
            # # Check for collision with other agents
            # for other_agent, other_agent_state in self.environment_state["agents"].items():
            #     if agent != other_agent:
            #         distance = np.linalg.norm(agent_state["position"] - other_agent_state["position"])
            #         combined_radius = (agent_state["size"] * np.sqrt(2) + other_agent_state["size"] * np.sqrt(2)) / 24  # Convert inches to feet
            #         if distance < combined_radius:
            #             penalty += -10  # Apply penalty for collision
            
            agent_state["gameTime"] += duration
            rewards[agent] = self._clearStepsDirectory(initial_score, penalty)
            
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
                if ring["status"] == self.agent_name_mapping[agent] + AGENT_ID_OFFSET:
                    ring["position"] = agent_state["position"].copy()

        infos = {agent: {} for agent in self.agents}

        # Update observations based on the updated environment state
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        self.score = self._compute_field_score() # Compute score before agents are removed

        # Remove agents if the environment is truncated or terminated
        if terminations["__all__"] or truncations["__all__"]:
            self.agents = []
        else:
            self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

        return observations, rewards, terminations, truncations, infos

    def is_valid_action(self, action, observation, last_action=None):
        """
        Check if the action is valid for the given observation.
        """

        # Check if the action is within the action space
        if not self.action_space(None).contains(action):
            return False

        # Check if the action is valid based on the observation
        if (action == Actions.PICK_UP_NEAREST_GOAL.value or action == Actions.PICK_UP_NEXT_NEAREST_GOAL) and observation[3] == 1:
            return False  # Already holding a goal
        if (action == Actions.PICK_UP_NEAREST_GOAL.value or action == Actions.PICK_UP_NEXT_NEAREST_GOAL) and observation[42] == 0:
            return False  # No visible goals to pick up
        if action == Actions.PICK_UP_NEAREST_RING.value and observation[4] >= 2:
            return False  # Already holding 2 rings
        if action == Actions.PICK_UP_NEAREST_RING.value and observation[38] == 0:
            return False  # No visible rings to pick up
        if action == Actions.DROP_GOAL.value and observation[3] == 0:
            return False # No goal to drop
        if action == Actions.ADD_RING_TO_GOAL.value and observation[4] == 0:
            return False  # No rings to add to goal
        if action == Actions.ADD_RING_TO_WALL_STAKE.value and observation[4] == 0:
            return False  # No rings to add to wall stake
        if action == Actions.CLIMB.value and observation[3] == 1:
            return False  # Can't climb while holding a goal
        
        # Prevent repeating certain actions consecutively
        if last_action is not None:
            repeat_actions = [
                Actions.DRIVE_TO_CORNER_BL.value,
                Actions.DRIVE_TO_CORNER_BR.value,
                Actions.DRIVE_TO_CORNER_TL.value,
                Actions.DRIVE_TO_CORNER_TR.value,
                Actions.DRIVE_TO_WALL_STAKE_L.value,
                Actions.DRIVE_TO_WALL_STAKE_R.value,
                Actions.DRIVE_TO_WALL_STAKE_B.value,
                Actions.DRIVE_TO_WALL_STAKE_T.value,
                Actions.TURN_TOWARDS_CENTER.value,
            ]
            if action == last_action and action in repeat_actions:
                return False
        
        return True

    def calculate_path(self, start_point, end_point):
        sp = np.array(start_point, dtype=np.float64)
        ep = np.array(end_point, dtype=np.float64)
        sol = self.path_planner.Solve(start_point=sp, end_point=ep, obstacles=self.permanent_obstacles)
        return self.path_planner.getPath(sol)

    def generate_path(self, action, observation):
        # Use observation to get robot position and orientation
        robot_position = np.array([observation[0], observation[1]])

        if action == Actions.CLIMB.value:
            # Path from last robot position to nearest initial climb position
            nearest_initial_climb_position = self.initial_climb_positions[np.argmin(np.linalg.norm(self.initial_climb_positions - robot_position, axis=1))]
            planned_x1, planned_y1, time1 = self.calculate_path(robot_position / ENV_FIELD_SIZE, nearest_initial_climb_position / ENV_FIELD_SIZE)
            planned_x1 *= ENV_FIELD_SIZE
            planned_y1 *= ENV_FIELD_SIZE

            # Path from initial climb position to corresponding climb position
            corresponding_climb_position = self.climb_positions[np.argmin(np.linalg.norm(self.initial_climb_positions - nearest_initial_climb_position, axis=1))]
            planned_x2, planned_y2, time2 = self.calculate_path(nearest_initial_climb_position / ENV_FIELD_SIZE, corresponding_climb_position / ENV_FIELD_SIZE)
            planned_x2 *= ENV_FIELD_SIZE
            planned_y2 *= ENV_FIELD_SIZE

        else:
            # For other actions, use robot_position and assume last_robot_position is also from observation
            # If you have access to previous observation, you can use that for last_robot_position
            planned_x, planned_y, time = self.calculate_path(robot_position / ENV_FIELD_SIZE, robot_position / ENV_FIELD_SIZE)
            planned_x *= ENV_FIELD_SIZE
            planned_y *= ENV_FIELD_SIZE

        forward_actions = [
            Actions.PICK_UP_NEAREST_RING.value,
            Actions.CLIMB.value,
        ]
        reverse_actions = [
            Actions.PICK_UP_NEAREST_GOAL.value,
            Actions.PICK_UP_NEXT_NEAREST_GOAL.value,
            Actions.DRIVE_TO_CORNER_BL.value,
            Actions.DRIVE_TO_CORNER_BR.value,
            Actions.DRIVE_TO_CORNER_TL.value,
            Actions.DRIVE_TO_CORNER_TR.value,
            Actions.DRIVE_TO_WALL_STAKE_L.value,
            Actions.DRIVE_TO_WALL_STAKE_R.value,
            Actions.DRIVE_TO_WALL_STAKE_B.value,
            Actions.DRIVE_TO_WALL_STAKE_T.value,
        ]
        has_path = action in forward_actions or action in reverse_actions
        reverse = action in reverse_actions

        if action == Actions.CLIMB.value:
            return has_path, reverse, [planned_x1, planned_x2], [planned_y1, planned_y2]
        elif has_path:
            return has_path, reverse, planned_x, planned_y
        else:
            return has_path, reverse, None, None

    def break_down_action(self, action, observation, generate_path_output=None):
        if generate_path_output is None:
            has_path, reverse, planned_x, planned_y = self.generate_path(action, observation)
        else:
            has_path, reverse, planned_x, planned_y = generate_path_output

        broken_down_actions = []
        path_action = 'BACKWARD' if reverse else 'FORWARD'

        # Use observation to get robot orientation
        robot_orientation = observation[2]

        # Treat CLIMB specially, it has two paths
        if action == Actions.CLIMB.value:
            path_1 = []
            for x, y in zip(planned_x[0], planned_y[0]):
                path_1.append(x)
                path_1.append(y)
            path_2 = []
            for x, y in zip(planned_x[1], planned_y[1]):
                path_2.append(x)
                path_2.append(y)

            broken_down_actions.append((path_action, path_1))
            broken_down_actions.append(('START_CLIMB', None))
            broken_down_actions.append((path_action, path_2))
            broken_down_actions.append(('END_CLIMB', None))
            
            return broken_down_actions

        if has_path:
            path = []
            for x, y in zip(planned_x, planned_y):
                path.append(x)
                path.append(y)
            broken_down_actions.append((path_action, path))

        if Actions.PICK_UP_NEAREST_GOAL.value == action or Actions.PICK_UP_NEXT_NEAREST_GOAL.value == action:
            broken_down_actions.append(('PICKUP_GOAL', None))
        elif Actions.PICK_UP_NEAREST_RING.value == action:
            broken_down_actions.append(('PICKUP_RING', None))
        elif Actions.DROP_GOAL.value == action:
            broken_down_actions.append(('DROP_GOAL', None))
        elif Actions.DRIVE_TO_CORNER_BL.value <= action <= Actions.DRIVE_TO_CORNER_TR.value:
            broken_down_actions.append(('TURN_TO', [robot_orientation]))
        elif Actions.ADD_RING_TO_GOAL.value == action:
            broken_down_actions.append(('ADD_RING_TO_GOAL', None))
        elif Actions.DRIVE_TO_WALL_STAKE_L.value <= action <= Actions.DRIVE_TO_WALL_STAKE_T.value:
            broken_down_actions.append(('TURN_TO', [robot_orientation]))
        elif Actions.ADD_RING_TO_WALL_STAKE.value == action:
            broken_down_actions.append(('ADD_RING_TO_WALL_STAKE', None))
        elif Actions.TURN_TOWARDS_CENTER.value == action:
            broken_down_actions.append(('TURN_TO', [robot_orientation]))

        return broken_down_actions

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
                agent_state = self.environment_state["agents"][agent]
                print(f"\t{agent}:")
                if actions is not None:
                    print(f"\t\tAction: {Actions(actions[agent]).name}")
                if rewards is not None:
                    print(f"\t\tReward: {rewards[agent]}")
                for attribute, value in agent_state.items():
                    print(f"\t\t{attribute}: {value}")
            print("Score:", self.score)
            gameTimes = [self.environment_state["agents"][agent]["gameTime"] for agent in self.agents]
            timeRemaining = 0
            if len(gameTimes) > 0:
                timeRemaining = TIME_LIMIT - min(gameTimes)
            print("Time Remaining:", timeRemaining)            
            print()
        if mode == "image" or mode == "all":
            self._save_image(self.num_moves, actions)
    
    def _save_image(self, step_num, actions=None):
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
                visible_by_any_agent = any(self._is_ring_available(i, self.environment_state["agents"][agent]) for agent in self.agents)
                color = 'red' if visible_by_any_agent else 'gray'
                circle = patches.Circle((ring["position"][0], ring["position"][1]), 0.3, color=color, alpha=0.7)
                ax.add_patch(circle)

            # Draw the goals
            for goal_idx, goal in enumerate(self.environment_state["goals"]):
                visible_by_any_agent = any(self._is_goal_available(goal_idx, self.environment_state["agents"][agent]) for agent in self.agents)
                color = 'green' if visible_by_any_agent else 'gray'
                hexagon = patches.RegularPolygon((goal["position"][0], goal["position"][1]), numVertices=6, radius=0.5, orientation=np.pi/6, color=color, alpha=0.25)
                ax.add_patch(hexagon)
                rings_on_goal = sum(1 for ring in self.environment_state["rings"] if ring["status"] == goal_idx + GOAL_ID_OFFSET)
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
                rings_on_stake = sum(1 for ring in self.environment_state["rings"] if ring["status"] == idx + STAKE_ID_OFFSET)
                ax.text(text_pos[0], text_pos[1], str(rings_on_stake), color='black', ha='center')

            # Draw the robots
            for agent in self.agents:
                x = self.environment_state["agents"][agent]["position"][0]
                y = self.environment_state["agents"][agent]["position"][1]
                orientation = self.environment_state["agents"][agent]["orientation"][0]
                width = self.environment_state["agents"][agent]["size"] / 12  # Convert inches to feet
                
                # Draw the rectangle representing the robot
                color = 'blue' if self.environment_state["agents"][agent]["active"] else 'gray'
                rect = patches.Rectangle(
                    (x - width / 2, y - width / 2), width, width, 
                    color=color, alpha=0.5, ec='black'
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
                rings_on_bot = sum(1 for ring in self.environment_state["rings"] if ring["status"] == self.agent_name_mapping[agent] + AGENT_ID_OFFSET)
                ax.text(x, y-width*.75, agent, fontsize=12, ha='center', va='center')
                ax.text(x, y, str(rings_on_bot), color='black', ha='center')

            # Draw the obstacles
            for obstacle in self.permanent_obstacles:
                circle = patches.Circle(
                    (obstacle.x * ENV_FIELD_SIZE, obstacle.y * ENV_FIELD_SIZE), 
                    obstacle.radius * ENV_FIELD_SIZE, 
                    edgecolor='black', facecolor='none', 
                    linestyle='dotted', alpha=0.5)
                ax.add_patch(circle)
            
            # Draw overlays for field of view
            for agent in self.agents:
                robot_position = self.environment_state["agents"][agent]["position"]
                robot_orientation = self.environment_state["agents"][agent]["orientation"][0]
                fov_length = 50
                left_fov_angle = robot_orientation + FOV / 2
                right_fov_angle = robot_orientation - FOV / 2
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
            
            ax.text(-2.5, 6, f"Total Score: {self.score}", color='black', ha='center')
            ax.text(6, 13.25, f'Step {step_num}', color='black', ha='center')

            if actions:
                action_str = ", ".join([f"{agent}: {Actions(actions[agent]).name}" for agent in self.agents])
                ax.text(6, -1, f"Actions: {action_str}", color='black', ha='center')

            os.makedirs(os.path.join(self.output_directory, "steps"), exist_ok=True)
            plt.savefig(os.path.join(self.output_directory, "steps", f"step_{step_num}.png"))
            plt.close()
    
    def clearStepsDirectory(self):
        """
        Clear the steps directory to remove old images.
        """
        steps_dir = os.path.join(self.output_directory, "steps")
        if (os.path.exists(steps_dir)):
            for filename in os.listdir(steps_dir):
                file_path = os.path.join(steps_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
    
    def createGIF(self):
        """
        Create a GIF from the saved images in the steps directory.
        """
        steps_dir = os.path.join(self.output_directory, "steps")
        images = []
        for filename in sorted(os.listdir(steps_dir), key=lambda x: int(x.split('_')[1].split('.')[0])):
            file_path = os.path.join(steps_dir, filename)
            images.append(imageio.imread(file_path))
        filename = os.path.join(self.output_directory, "simulation.gif")
        imageio.mimsave(
            filename,
            images,
            fps=10
        )
        print(f"GIF saved successfully to {filename}")


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def _clearStepsDirectory(self, initial_score, penalty):
        new_score = self._compute_field_score()
        delta_score = new_score - initial_score
        reward = delta_score + penalty
        return reward

    def _compute_field_score(self, skills=True):
        score = 0
        for goal_idx, goal in enumerate(self.environment_state["goals"]):
            rings_on_goal = sum(1 for ring in self.environment_state["rings"] if ring["status"] == goal_idx + GOAL_ID_OFFSET)
            if rings_on_goal > 0:
                in_corner = any(np.linalg.norm(goal["position"] - corner) < 0.5 for corner in self.climb_positions)
                if(skills):
                    score += (3 + (rings_on_goal - 1)) + (5 if in_corner else 0)
                else:
                    score += (3 + (rings_on_goal - 1)) * (2 if in_corner else 1)

        for stake_idx, stake_pos in enumerate(self.wall_stakes_positions):
            rings_on_stake = sum(1 for ring in self.environment_state["rings"] if ring["status"] == stake_idx + STAKE_ID_OFFSET)
            if rings_on_stake > 0:
                score += 3 + (rings_on_stake - 1)

        for agent in self.agents:
            if self.environment_state["agents"][agent]["climbed"]:
                score += 3

        self.total_score = score
        return score
    
    def _is_goal_available(self, goal_index, agent_state):
        goal_position = self.environment_state["goals"][goal_index]["position"]
        return (
            self._is_visible(goal_position, agent_state) and
            self.environment_state["goals"][goal_index]["status"] == 0  # Status 0 means available
        )

    def _is_ring_available(self, ring_index, agent_state):
        ring_position = self.environment_state["rings"][ring_index]["position"]
        return (
            self._is_visible(ring_position, agent_state) and
            self.environment_state["rings"][ring_index]["status"] == 0  # Status 0 means available
        )

    def _is_visible(self, position, agent_state):
        if not self.realistic_vision:
            return True
        if position[0] < 0 or position[0] > ENV_FIELD_SIZE or position[1] < 0 or position[1] > ENV_FIELD_SIZE:
            return False
        direction = position - agent_state["position"]
        angle = np.arctan2(direction[1], direction[0])
        relative_angle = (angle - agent_state["orientation"] + np.pi) % (2 * np.pi) - np.pi
        return -FOV / 2 <= relative_angle <= FOV / 2

    def _get_available_goals(self, agent_state):
        return np.array([
            goal["position"]
            for i, goal in enumerate(self.environment_state["goals"])
            if self._is_goal_available(i, agent_state)
        ])

    def _get_available_rings(self, agent_state):
        return np.array([
            ring["position"]
            for i, ring in enumerate(self.environment_state["rings"])
            if self._is_ring_available(i, agent_state)
        ])

if __name__ == "__main__":
    env = High_Stakes_Multi_Agent_Env(render_mode="all", output_directory="pettingZooEnv")
    print("Testing the environment...")
    parallel_api_test(env)

    observations, infos = env.reset()

    env.clearStepsDirectory()

    done = False
    print("Running the environment...")
    env.render()
    while not done:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations["__all__"] or truncations["__all__"]
        env.render(actions, rewards)
    env.close()