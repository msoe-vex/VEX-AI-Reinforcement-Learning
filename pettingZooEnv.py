import functools

import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.test import parallel_api_test
from ray.rllib.env import MultiAgentEnv

import numpy as np
from enum import Enum

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
DEFAULT_PENALTY = -1

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
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(MultiAgentEnv, ParallelEnv):
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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            "robot_x": spaces.Box(low=0, high=ENV_FIELD_SIZE, shape=(1,), dtype=np.float32),
            "robot_y": spaces.Box(low=0, high=ENV_FIELD_SIZE, shape=(1,), dtype=np.float32),
            "robot_orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "holding_goal": spaces.Discrete(2),
            "holding_rings": spaces.Discrete(3),
            "ring_positions": spaces.Box(low=-1, high=ENV_FIELD_SIZE, shape=(NUM_RINGS * 2,), dtype=np.float32),
            "goal_positions": spaces.Box(low=-1, high=ENV_FIELD_SIZE, shape=(NUM_GOALS * 2,), dtype=np.float32),
            "wall_stake_ring_count": spaces.Box(low=0, high=6, shape=(NUM_WALL_STAKES,), dtype=np.int32),
            "holding_goal_full": spaces.Discrete(2),
            "time_remaining": spaces.Box(low=0, high=TIME_LIMIT, shape=(1,), dtype=np.float32),
            "visible_rings_count": spaces.Discrete(NUM_RINGS + 1),
            "visible_goals_count": spaces.Discrete(NUM_GOALS + 1)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Actions))

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]  # Reset agents to all possible agents
        self.num_moves = 0
        self.state = {agent: self._get_initial_observation(agent) for agent in self.agents}  # Initialize state
        observations = {agent: self.state[agent] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_initial_observation(self, agent):
        return {
            "robot_x": np.array([0.0], dtype=np.float32),
            "robot_y": np.array([0.0], dtype=np.float32),
            "robot_orientation": np.array([0.0], dtype=np.float32),
            "holding_goal": 0,
            "holding_rings": 0,
            "ring_positions": np.full((NUM_RINGS * 2,), -1, dtype=np.float32),
            "goal_positions": np.full((NUM_GOALS * 2,), -1, dtype=np.float32),
            "wall_stake_ring_count": np.zeros(NUM_WALL_STAKES, dtype=np.int32),
            "holding_goal_full": 0,
            "time_remaining": np.array([TIME_LIMIT], dtype=np.float32),
            "visible_rings_count": 0,
            "visible_goals_count": 0
        }

    def step(self, actions):
        # Ensure actions are provided for all agents
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Perform actions for each agent
        # TODO: Implement the logic for each action
        # TODO: Implement time synchronization issues

        # TODO: Implement the logic for calculating rewards
        rewards = {agent: 0 for agent in self.agents}

        # TODO: Implement logic for terminating the episode
        terminations = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}
        terminations["__all__"] = all(terminations.values())

        truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = all(truncations.values())

        infos = {agent: {} for agent in self.agents}

        # Increment move counter
        self.num_moves += 1

        # TODO: Update observations based on actions taken
        observations = {agent: self._get_initial_observation(agent) for agent in self.agents}

        # Remove agents if the environment is truncated or terminated
        if all(truncations.values()) or all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Robot positions: Agent1 (x: {}, y: {}), Agent2 (x: {}, y: {})".format(
                self.state[self.agents[0]]["robot_x"][0],
                self.state[self.agents[0]]["robot_y"][0],
                self.state[self.agents[1]]["robot_x"][0],
                self.state[self.agents[1]]["robot_y"][0]
            )
            print(string)
        else:
            print("Game over")

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
    

if __name__ == "__main__":
    env = parallel_env(render_mode="human")
    parallel_api_test(env, num_cycles=1000)

    observations, infos = env.reset()

    done = False
    while not done:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        env.render()
    env.close()