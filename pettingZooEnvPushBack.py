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
BORDER_WALL_INCHES=2
ENV_FIELD_SIZE = 12
BUFFER_RADIUS = 2 # buffer around robot for collision detection
NUM_WALL_STAKES = 4
NUM_GOALS = 5
NUM_RINGS = 24
TIME_LIMIT = 60
DEFAULT_PENALTY = -0.1
FOV = np.pi / 2

POSSIBLE_AGENTS = ["robot_0"]

# Offsets for status IDs
AGENT_ID_OFFSET = 1
GOAL_ID_OFFSET = AGENT_ID_OFFSET + len(POSSIBLE_AGENTS)
STAKE_ID_OFFSET = GOAL_ID_OFFSET + NUM_GOALS

WALL_GOAL_COORDINATES = {
    "BOTTOM_WALL_GOAL_ENTERANCE_1": (50/INCHES_PER_FIELD, 25/INCHES_PER_FIELD),
    "BOTTOM_WALL_GOAL_ENTERANCE_2": (94/INCHES_PER_FIELD, 25/INCHES_PER_FIELD),
    "TOP_WALL_GOAL_ENTERANCE_1": (50/INCHES_PER_FIELD, 119/INCHES_PER_FIELD),
    "TOP_WALL_GOAL_ENTERANCE_2": (94/INCHES_PER_FIELD, 118/INCHES_PER_FIELD),
}
CENTER_GOAL_COORDINATES = {
    "BOTTOM_LEFT_GOAL": (62/INCHES_PER_FIELD, 64/INCHES_PER_FIELD),
    "BOTTOM_RIGHT_GOAL": (78/INCHES_PER_FIELD, 80/INCHES_PER_FIELD),
    "TOP_LEFT_GOAL": (62/INCHES_PER_FIELD, 80/INCHES_PER_FIELD),
    "TOP_RIGHT_GOAL": (78/INCHES_PER_FIELD, 64/INCHES_PER_FIELD),
}

BLOCK_COORDINATES = [
    {"x": 116/INCHES_PER_FIELD, "y": 25/INCHES_PER_FIELD, "color": "blue"},
    {"x": 116/INCHES_PER_FIELD, "y": 116/INCHES_PER_FIELD, "color": "blue"},
    {"x": 22/INCHES_PER_FIELD, "y": 25/INCHES_PER_FIELD, "color": "red"},
    {"x": 22/INCHES_PER_FIELD, "y": 116/INCHES_PER_FIELD, "color": "red"},
    {"x": 48/INCHES_PER_FIELD, "y": 48/INCHES_PER_FIELD, "color": "red"},
    {"x": 48/INCHES_PER_FIELD, "y": 91/INCHES_PER_FIELD, "color": "blue"},
    {"x": 91/INCHES_PER_FIELD, "y": 48/INCHES_PER_FIELD, "color": "blue"},
    {"x": 91/INCHES_PER_FIELD, "y": 91/INCHES_PER_FIELD, "color": "red"},
]

ROBOT_STARTING_POSITIONS = {
    "red_robot_0": {"x": 0/INCHES_PER_FIELD, "y":70/INCHES_PER_FIELD},
    "blue_robot_0": {"x": 140/INCHES_PER_FIELD, "y":70/INCHES_PER_FIELD},
}

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
        self.invalid_actions = []

        self.wall_stakes_positions = [
            np.array([0, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE/2, 0]),
            np.array([ENV_FIELD_SIZE/2, ENV_FIELD_SIZE])
        ]
        self.goals = [Obstacle(WALL_GOAL_COORDINATES[goal][0], WALL_GOAL_COORDINATES[goal][1], 3.5/INCHES_PER_FIELD, False) for goal in WALL_GOAL_COORDINATES]
        self.goals += [Obstacle(CENTER_GOAL_COORDINATES[goal][0], CENTER_GOAL_COORDINATES[goal][1], 3.5/INCHES_PER_FIELD, False) for goal in CENTER_GOAL_COORDINATES]
        
        self.blocks = [Obstacle(block["x"], block["y"], 3.5/INCHES_PER_FIELD, False) for block in BLOCK_COORDINATES]
        self.realistic_vision = True
        self.score = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """"""

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Actions))

    def observation_space_sample(self, agent_ids=None):
        """agent_ids = agent_ids or self.get_agent_ids()
        return {agent_id: self.observation_space(agent_id).sample() for agent_id in agent_ids}"""

    def action_space_sample(self, agent_ids=None):
        """agent_ids = agent_ids or self.get_agent_ids()
        return {agent_id: self.action_space(agent_id).sample() for agent_id in agent_ids}"""

    def observation_space_contains(self, observations):
        """return all(
            agent_id in self.get_agent_ids() and
            self.observation_space(agent_id).contains(obs)
            for agent_id, obs in observations.items()
        )"""

    def action_space_contains(self, actions):
        """return all(
            agent_id in self.get_agent_ids() and
            self.action_space(agent_id).contains(action)
            for agent_id, action in actions.items()
        )"""

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
        }

    def _get_random_environment_state(self, seed=None):
        """"""

    def _get_observation(self, agent):
        """"""

    def step(self, actions):
        """"""

    def is_valid_action(self, action, observation, last_action=None):
        """
        Check if the action is valid for the given observation.
        """
        return True

    def calculate_path(self, start_point, end_point):
        """"""

    def generate_path(self, action, observation):
        """"""

    def break_down_action(self, action, observation, generate_path_output=None):
        """"""

    def render(self, actions=None, rewards=None):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        self._save_image(1, actions)
    
    def _save_image(self, step_num, actions=None):
        # Create visualization
        fig, ax = plt.subplots(figsize=(10,8))

        # Draw the obstacles
        for obstacle in self.goals:
            circle = patches.Circle(
                (obstacle.x * ENV_FIELD_SIZE, obstacle.y * ENV_FIELD_SIZE), 
                obstacle.radius * ENV_FIELD_SIZE, 
                edgecolor='black', facecolor='none', 
                linestyle='dotted', alpha=0.5)
            ax.add_patch(circle)
        for obstacle in self.blocks:
            circle = patches.Circle(
                (obstacle.x * ENV_FIELD_SIZE, obstacle.y * ENV_FIELD_SIZE), 
                obstacle.radius * ENV_FIELD_SIZE, 
                edgecolor='black', facecolor='none', 
                linestyle='dotted', alpha=0.5)
            ax.add_patch(circle)
        
        ax.text(-2.5, 6, f"Total Score: {self.score}", color='black', ha='center')
        ax.text(6, 13.25, f'Step {step_num}', color='black', ha='center')

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
    
    #print("Testing the environment...")
    #parallel_api_test(env)

    observations, infos = env.reset()

    env.clearStepsDirectory()

    done = False
    print("Running the environment...")
    env.render()
    while not done:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # observations, rewards, terminations, truncations, infos = env.step(actions)
        
        env.render(actions, None)
        done = True
    env.createGIF()
    env.close()