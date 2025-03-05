import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from enum import Enum
from path_planner import PathPlanner, Obstacle

# =============================================================================
# Enum to assign integer values to robot actions
# =============================================================================
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
TIME_LIMIT = 120
DEFAULT_PENALTY = -1

# =============================================================================
# Environment class for the VEX High Stakes Challenge
# =============================================================================
class VEXHighStakesEnv(gym.Env):

    # -----------------------------------------------------------------------------
    # Initialize environment settings and spaces.
    # -----------------------------------------------------------------------------
    def __init__(self, save_path, randomize_positions=True, realistic_pathing=False, realistic_vision=True):
        super(VEXHighStakesEnv, self).__init__()
        self.randomize_positions = randomize_positions
        self.realistic_pathing = realistic_pathing
        self.realistic_vision = realistic_vision
        self.num_goals = NUM_GOALS
        self.num_rings = NUM_RINGS

        self.robot_length = 15 # inches
        self.robot_width = 15 # inches
        self.robot_radius = np.sqrt( \
                                (self.robot_length*(ENV_FIELD_SIZE/INCHES_PER_FIELD))**2 + \
                                (self.robot_width*(ENV_FIELD_SIZE/INCHES_PER_FIELD))**2
                            ) / 2 + BUFFER_RADIUS*(ENV_FIELD_SIZE/INCHES_PER_FIELD)

        self.wall_stakes_positions = np.array([
            [self.robot_radius, ENV_FIELD_SIZE/2],
            [ENV_FIELD_SIZE-self.robot_radius, ENV_FIELD_SIZE/2],
            [ENV_FIELD_SIZE/2, self.robot_radius],
            [ENV_FIELD_SIZE/2, ENV_FIELD_SIZE-self.robot_radius]
        ])
        self.corner_positions = np.array([
            [self.robot_radius, self.robot_radius],
            [ENV_FIELD_SIZE-self.robot_radius, self.robot_radius],
            [self.robot_radius, ENV_FIELD_SIZE-self.robot_radius],
            [ENV_FIELD_SIZE-self.robot_radius, ENV_FIELD_SIZE-self.robot_radius]
        ])
        # Define the observation space for the environment
        self.observation_space = spaces.Dict({
            "robot_x": spaces.Box(low=0, high=ENV_FIELD_SIZE, shape=(1,), dtype=np.float32),
            "robot_y": spaces.Box(low=0, high=ENV_FIELD_SIZE, shape=(1,), dtype=np.float32),
            "robot_orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "holding_goal": spaces.Discrete(2),
            "holding_rings": spaces.Discrete(3),
            "rings": spaces.Box(low=-1, high=ENV_FIELD_SIZE, shape=(self.num_rings*2,), dtype=np.float32),
            "goals": spaces.Box(low=-1, high=ENV_FIELD_SIZE, shape=(self.num_goals*2,), dtype=np.float32),
            "wall_stakes": spaces.Box(low=0, high=6, shape=(NUM_WALL_STAKES,), dtype=np.int32),
            "holding_goal_full": spaces.Discrete(2),
            "time_remaining": spaces.Box(low=0, high=TIME_LIMIT, shape=(1,), dtype=np.float32),
            "visible_rings_count": spaces.Discrete(NUM_RINGS + 1),
            "visible_goals_count": spaces.Discrete(NUM_GOALS + 1)
        })
        self.action_space = spaces.Discrete(15)

        self.save_path = save_path
        self.steps_save_path = f"{save_path}/steps"

        self.permanent_obstacles = [Obstacle(3/6, 2/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(3/6, 4/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(2/6, 3/6, 3.5/INCHES_PER_FIELD, False),
                     Obstacle(4/6, 3/6, 3.5/INCHES_PER_FIELD, False)]
        self.path_planner = PathPlanner(
            robot_length=self.robot_length/INCHES_PER_FIELD,
            robot_width=self.robot_width/INCHES_PER_FIELD,
            buffer_radius=BUFFER_RADIUS/INCHES_PER_FIELD,
            max_velocity=80/INCHES_PER_FIELD,
            max_accel=100/INCHES_PER_FIELD)

        self.reset()

    # -----------------------------------------------------------------------------
    # Build and return the current observation.
    # -----------------------------------------------------------------------------
    def _get_observation(self):
        self.padded_goals[:self.mobile_goal_positions.size] = self.mobile_goal_positions.flatten()
        self.padded_rings[:self.ring_positions.size] = self.ring_positions.flatten()
        self.update_visible_objects()
        normalized_orientation = (self.robot_orientation + np.pi) % (2 * np.pi) - np.pi
        return {
            "robot_x": np.array([self.robot_position[0]], dtype=np.float32),
            "robot_y": np.array([self.robot_position[1]], dtype=np.float32),
            "robot_orientation": np.array([normalized_orientation], dtype=np.float32),
            "holding_goal": self.holding_goal,
            "holding_rings": int(np.sum(self.ring_status == 1)),
            "rings": self.padded_rings,
            "goals": self.padded_goals,
            "wall_stakes": self.wall_stakes,
            "holding_goal_full": self.holding_goal_full,
            "time_remaining": np.array([self.time_remaining], dtype=np.float32),
            "visible_rings_count": self.visible_rings_count,
            "visible_goals_count": self.visible_goals_count
        }

    # -----------------------------------------------------------------------------
    # Reset environment to its initial state.
    # -----------------------------------------------------------------------------
    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.randomize_positions:
            self.mobile_goal_positions = np.random.uniform(0.5, 11.5, size=(NUM_GOALS, 2)).astype(np.float32)
            self.ring_positions = np.random.uniform(0.5, 11.5, size=(NUM_RINGS, 2)).astype(np.float32)
            self.robot_position = np.random.uniform(0.5, 11.5, size=(2,)).astype(np.float32)
            self.robot_orientation = np.random.uniform(-np.pi, np.pi)
        else:
            self.robot_position = np.array([0.5, 6.0], dtype=np.float32)
            self.robot_orientation = 0.0
            self.mobile_goal_positions = np.array([[4.0, 2.0], [4.0, 10.0], [8.0, 4.0], [8.0, 8.0], [10.0, 6.0]], dtype=np.float32)
            self.ring_positions = np.array([
                [6.0, 1.0], [6.0, 11.0],
                [2.0, 2.0], [2.0, 6.0], [2.0, 10.0], [4.0, 4.0], [4.0, 8.0],
                [5.7, 5.7], [5.7, 6.3], [6.3, 5.7], [6.3, 6.3],
                [6.0, 2.0], [6.0, 10.0], [8.0, 2.0], [8.0, 10.0], [10.0, 2.0], [10.0, 10.0],
                [1, 1], [11, 11], [11, 1], [1, 11],
                [10.0, 4.0], [10.0, 8.0], [11.0, 6.0]
            ], dtype=np.float32)
        self.last_robot_position = self.robot_position.copy()
        self.goal_available = np.ones(len(self.mobile_goal_positions), dtype=bool)
        self.ring_status = np.zeros(self.ring_positions.shape[0], dtype=np.int32)
        self.holding_goal = 0
        self.holding_goal_index = -1
        self.holding_goal_full = 0
        self.time_remaining = TIME_LIMIT
        self.holding_rings = 0
        self.total_score = 0
        self.wall_stakes = np.zeros(NUM_WALL_STAKES, dtype=np.int32)
        self.climbed = False
        self.padded_goals = np.full((NUM_GOALS * 2,), -1, dtype=np.float32)
        self.padded_rings = np.full((NUM_RINGS * 2,), -1, dtype=np.float32)
        self.last_action_success = False
        obs = self._get_observation()
        return obs, {}

    # -----------------------------------------------------------------------------
    # Take an action and update the environment state.
    # -----------------------------------------------------------------------------
    def step(self, action):
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        done = False
        truncated = False
        time_cost = 0.1
        penalty = 0
        initial_score = self.compute_field_score()
        initial_time_remaining = self.time_remaining
        self.last_robot_position = self.robot_position.copy()
        self.last_action_success = False

        # Local variables for frequently accessed attributes
        robot_position = self.robot_position
        robot_orientation = self.robot_orientation
        holding_goal = self.holding_goal
        holding_rings = self.holding_rings
        goal_available = self.goal_available
        ring_status = self.ring_status
        mobile_goal_positions = self.mobile_goal_positions
        ring_positions = self.ring_positions

        # ----------------------------------------------------------------------------- 
        # PICK_UP_NEAREST_GOAL (Restrictions: not holding a goal; at least one goal is visible)
        # Drives the robot to the nearest visible goal and picks it up.
        # -----------------------------------------------------------------------------
        if action == Actions.PICK_UP_NEAREST_GOAL.value:
            penalty = DEFAULT_PENALTY
            if holding_goal == 0:
                candidate = np.where(goal_available)[0]
                visible_candidates = [i for i in candidate if self.is_visible(mobile_goal_positions[i])]
                if visible_candidates:
                    goals = mobile_goal_positions[visible_candidates]
                    distances = np.linalg.norm(goals - robot_position, axis=1)
                    min_index = np.argmin(distances)
                    target_position = mobile_goal_positions[visible_candidates[min_index]]
                    old_position = robot_position
                    robot_position = target_position
                    robot_orientation = np.arctan2(target_position[1] - old_position[1],
                                                   target_position[0] - old_position[0])
                    time_cost = distances[min_index] / 2 + 0.5

                    penalty = 0
                    self.last_action_success = True

                    chosen_idx = visible_candidates[min_index]
                    holding_goal = 1
                    self.holding_goal_index = chosen_idx
                    goal_available[chosen_idx] = False
                    self.holding_goal_full = 1 if np.sum(ring_status == (chosen_idx + 2)) == 6 else 0

        # ----------------------------------------------------------------------------- 
        # PICK_UP_NEAREST_RING (Restrictions: holding less than 2 rings; at least one ring is visible)
        # Drives the robot to the nearest visible ring and picks it up.
        # -----------------------------------------------------------------------------
        elif action == Actions.PICK_UP_NEAREST_RING.value:
            penalty = DEFAULT_PENALTY
            if holding_rings < 2:
                candidate = np.where(ring_status == 0)[0]
                visible_candidates = [i for i in candidate if self.is_visible(ring_positions[i])]
                if visible_candidates:
                    rings = ring_positions[visible_candidates]
                    distances = np.linalg.norm(rings - robot_position, axis=1)
                    min_index = np.argmin(distances)
                    target_position = ring_positions[visible_candidates[min_index]]
                    old_position = robot_position
                    robot_position = target_position
                    robot_orientation = np.arctan2(target_position[1] - old_position[1],
                                                   target_position[0] - old_position[0])
                    time_cost = distances[min_index] / 2 + 0.5

                    penalty = 0
                    self.last_action_success = True

                    chosen_idx = visible_candidates[min_index]
                    ring_status[chosen_idx] = 1
                    holding_rings += 1
                    ring_positions[chosen_idx] = robot_position

        # ----------------------------------------------------------------------------- 
        # CLIMB (Restrictions: more than 5 seconds remaining; less than 20 seconds remaining)
        # Sets the climbed flag to True and ends the episode.
        # -----------------------------------------------------------------------------
        elif action == Actions.CLIMB.value:
            if self.time_remaining > 5:
                penalty = 0
                if self.time_remaining > 20:
                    penalty = -1000
                
                self.climbed = True
                self.last_action_success = True
            else:
                penalty = DEFAULT_PENALTY
            time_cost = self.time_remaining

        # ----------------------------------------------------------------------------- 
        # DROP_GOAL (Restrictions: holding goal, area around bot is clear)
        # Drops the held goal at the robot's current position.
        # -----------------------------------------------------------------------------
        elif action == Actions.DROP_GOAL.value:
            penalty = DEFAULT_PENALTY
            time_cost = 0.5
            if holding_goal == 1:
                drop_position = robot_position
                other_goals = np.delete(mobile_goal_positions, self.holding_goal_index, axis=0)
                goal_distances = np.linalg.norm(other_goals - drop_position, axis=1)
                
                # Filter out rings that are on goals
                rings_not_on_goals = ring_positions[ring_status == 0]
                ring_distances = np.linalg.norm(rings_not_on_goals - drop_position, axis=1)
                
                if np.all(goal_distances >= 1.0) and np.all(ring_distances >= 1.0):
                    mobile_goal_positions[self.holding_goal_index] = drop_position
                    goal_available[self.holding_goal_index] = True
                    holding_goal = 0
                    self.holding_goal_index = -1
                    self.holding_goal_full = 0

                    penalty = 0
                    self.last_action_success = True

        # ----------------------------------------------------------------------------- 
        # DRIVE_TO_CORNER (Restrictions: robot is not at the corner; no goals are in the corner)
        # Drives the robot to the target corner and adjusts orientation to face the corner diagonally.
        # -----------------------------------------------------------------------------
        elif Actions.DRIVE_TO_CORNER_BL.value <= action <= Actions.DRIVE_TO_CORNER_TR.value:
            old_position = robot_position
            target_position = self.corner_positions[action - Actions.DRIVE_TO_CORNER_BL.value]
            direction = target_position - old_position
            distance = np.linalg.norm(target_position - old_position)

            if self.realistic_pathing:
                planned_x, planned_y, time = self.calculate_path(robot_position / ENV_FIELD_SIZE, target_position / ENV_FIELD_SIZE)
                time_cost = float(time)
            else:
                time_cost = distance / 2 + 0.1
            
            if distance == 0:
                penalty = DEFAULT_PENALTY
                self.last_action_success = False
            else:
                robot_position = target_position
                if action == Actions.DRIVE_TO_CORNER_TL.value:
                    robot_orientation = 3 * np.pi / 4  # Facing top-left
                elif action == Actions.DRIVE_TO_CORNER_TR.value:
                    robot_orientation = np.pi / 4  # Facing top-right
                elif action == Actions.DRIVE_TO_CORNER_BL.value:
                    robot_orientation = 5 * np.pi / 4  # Facing bottom-left
                elif action == Actions.DRIVE_TO_CORNER_BR.value:
                    robot_orientation = 7 * np.pi / 4  # Facing bottom-right
                self.last_action_success = True
            
            for goal_pos in mobile_goal_positions:
                if np.linalg.norm(goal_pos - target_position) < 0.5:
                    penalty = DEFAULT_PENALTY
                    self.last_action_success = False
                    break

        # ----------------------------------------------------------------------------- 
        # ADD_RING_TO_GOAL (Restrictions: robot has a goal; robot has rings)
        # Adds a ring to the held goal.
        # -----------------------------------------------------------------------------
        elif action == Actions.ADD_RING_TO_GOAL.value:
            penalty = DEFAULT_PENALTY
            time_cost = 0.5
            if holding_goal == 1 and holding_rings > 0:
                held_goal_idx = self.holding_goal_index
                if np.sum(ring_status == (held_goal_idx + 2)) < 6:
                    robot_ring_idx = np.where(ring_status == 1)[0]
                    if robot_ring_idx.size > 0:
                        robot_ring_idx = robot_ring_idx[0]
                        ring_status[robot_ring_idx] = held_goal_idx + 2
                        ring_positions[robot_ring_idx] = mobile_goal_positions[held_goal_idx]
                        holding_rings -= 1
                        self.holding_goal_full = 1 if np.sum(ring_status == (held_goal_idx + 2)) == 6 else 0
                        
                        penalty = 0
                        self.last_action_success = True

        # ----------------------------------------------------------------------------- 
        # DRIVE_TO_WALL_STAKE (Restrictions: robot is not at the target stake)
        # Drives the robot to the target wall stake.
        # -----------------------------------------------------------------------------
        elif Actions.DRIVE_TO_WALL_STAKE_L.value <= action <= Actions.DRIVE_TO_WALL_STAKE_T.value:
            stake_idx = action - Actions.DRIVE_TO_WALL_STAKE_L.value
            old_position = robot_position
            target_position = self.wall_stakes_positions[stake_idx]
            direction = target_position - old_position

            distance = np.linalg.norm(target_position - old_position)

            if self.realistic_pathing:
                planned_x, planned_y, time = self.calculate_path(robot_position / ENV_FIELD_SIZE, target_position / ENV_FIELD_SIZE)
                time_cost = float(time)
            else:
                time_cost = distance / 2 + 0.1

            if distance == 0:
                penalty = DEFAULT_PENALTY
                self.last_action_success = False
            else:
                robot_position = target_position
                if action == Actions.DRIVE_TO_WALL_STAKE_R.value:
                    robot_orientation = 0  # Facing right
                elif action == Actions.DRIVE_TO_WALL_STAKE_L.value:
                    robot_orientation = np.pi  # Facing left
                elif action == Actions.DRIVE_TO_WALL_STAKE_T.value:
                    robot_orientation = np.pi / 2  # Facing up
                elif action == Actions.DRIVE_TO_WALL_STAKE_B.value:
                    robot_orientation = -np.pi / 2  # Facing down
                self.last_action_success = True

        # ----------------------------------------------------------------------------- 
        # ADD_RING_TO_WALL_STAKE (Restrictions: robot is at a stake; robot has rings; stake is not full)
        # Adds a ring to the wall stake at the robot's position.
        # -----------------------------------------------------------------------------
        elif action == Actions.ADD_RING_TO_WALL_STAKE.value:
            penalty = DEFAULT_PENALTY
            time_cost = 0.5
            if holding_rings > 0:
                distances = np.linalg.norm(self.wall_stakes_positions - robot_position, axis=1)
                nearest_stake_idx = np.argmin(distances)
                max_rings_on_stake = 2 if nearest_stake_idx < 2 else 6
                if self.wall_stakes[nearest_stake_idx] < max_rings_on_stake and distances[nearest_stake_idx] < 1.0:
                    robot_ring_idx = np.where(ring_status == 1)[0]
                    if robot_ring_idx.size > 0:
                        robot_ring_idx = robot_ring_idx[0]
                        ring_status[robot_ring_idx] = nearest_stake_idx + 7
                        self.wall_stakes[nearest_stake_idx] += 1
                        holding_rings -= 1

                        penalty = 0
                        self.last_action_success = True

        # ----------------------------------------------------------------------------- 
        # TURN_TOWARDS_CENTER (Restrictions:robot is not facing the center)
        # Sets the robot's orientation to face the center of the field.
        # -----------------------------------------------------------------------------
        elif action == Actions.TURN_TOWARDS_CENTER.value:
            center_position = np.array([ENV_FIELD_SIZE/2, ENV_FIELD_SIZE/2])
            direction = center_position - robot_position
            new_orientation = np.arctan2(direction[1], direction[0])
            if np.isclose(robot_orientation, new_orientation, atol=1e-2):
                penalty = DEFAULT_PENALTY
            else:
                robot_orientation = new_orientation

                penalty = 0
                self.last_action_success = True
            time_cost = 0.5

        # ----------------------------------------------------------------------------- 
        # PICK_UP_NEXT_NEAREST_GOAL (Restrictions: not holding a goal; at least two goals are visible)
        # Drives the robot to the next nearest visible goal and picks it up.
        # -----------------------------------------------------------------------------
        elif action == Actions.PICK_UP_NEXT_NEAREST_GOAL.value:
            penalty = DEFAULT_PENALTY
            if holding_goal == 0:
                candidate = np.where(goal_available)[0]
                visible_candidates = [i for i in candidate if self.is_visible(mobile_goal_positions[i])]
                if len(visible_candidates) > 1:
                    goals = mobile_goal_positions[visible_candidates]
                    distances = np.linalg.norm(goals - robot_position, axis=1)
                    sorted_indices = np.argsort(distances)
                    next_nearest_index = sorted_indices[1]  # Get the second nearest goal
                    target_position = mobile_goal_positions[visible_candidates[next_nearest_index]]
                    old_position = robot_position
                    robot_position = target_position
                    robot_orientation = np.arctan2(target_position[1] - old_position[1],
                                                   target_position[0] - old_position[0])
                    time_cost = distances[next_nearest_index] / 2 + 0.5

                    penalty = 0
                    self.last_action_success = True

                    chosen_idx = visible_candidates[next_nearest_index]
                    holding_goal = 1
                    self.holding_goal_index = chosen_idx
                    goal_available[chosen_idx] = False
                    self.holding_goal_full = 1 if np.sum(ring_status == (chosen_idx + 2)) == 6 else 0

        self.time_remaining = np.clip(self.time_remaining - time_cost, 0, TIME_LIMIT)
        if self.time_remaining <= 0:
            done = True
        if holding_goal == 1:
            mobile_goal_positions[self.holding_goal_index] = robot_position
        
        # Update class attributes based on action results
        self.robot_position = robot_position
        self.robot_orientation = robot_orientation
        self.holding_goal = holding_goal
        self.holding_rings = holding_rings
        self.goal_available = goal_available
        self.ring_status = ring_status
        self.mobile_goal_positions = mobile_goal_positions
        self.ring_positions = ring_positions

        obs = self._get_observation()
        reward = self.reward_function(initial_score, initial_time_remaining, penalty)
        return obs, reward, done, truncated, {}

    # -----------------------------------------------------------------------------
    # Check if a given position is visible from the robot.
    # -----------------------------------------------------------------------------
    def is_visible(self, position):
        if not self.realistic_vision:
            return True
        direction = position - self.robot_position
        angle = np.arctan2(direction[1], direction[0])
        relative_angle = (angle - self.robot_orientation + np.pi) % (2 * np.pi) - np.pi
        return -np.pi / 2 <= relative_angle <= np.pi / 2

    # -----------------------------------------------------------------------------
    # Update internal representation of visible goals and rings.
    # -----------------------------------------------------------------------------
    def update_visible_objects(self):
        if not self.realistic_vision:
            self.padded_goals = self.mobile_goal_positions.flatten()
            self.padded_rings = self.ring_positions.flatten()
            self.visible_rings_count = self.num_rings
            self.visible_goals_count = self.num_goals
        else:
            visible_goals = np.full((NUM_GOALS * 2,), -1, dtype=np.float32)
            visible_rings = np.full((NUM_RINGS * 2,), -1, dtype=np.float32)
            for i, goal in enumerate(self.mobile_goal_positions):
                if self.is_visible(goal):
                    visible_goals[i*2:i*2+2] = goal
            for i, ring in enumerate(self.ring_positions):
                if self.is_visible(ring):
                    visible_rings[i*2:i*2+2] = ring
            self.padded_goals = visible_goals
            self.padded_rings = visible_rings
            self.visible_rings_count = np.sum(visible_rings != -1) // 2
            self.visible_goals_count = np.sum(visible_goals != -1) // 2

    # -----------------------------------------------------------------------------
    # Compute and return the field score based on VEXU skills rules.
    # -----------------------------------------------------------------------------
    def compute_field_score(self):
        score = 0
        for goal_idx, goal_pos in enumerate(self.mobile_goal_positions):
            if np.any(self.ring_status == (goal_idx + 2)):
                rings_on_goal = np.sum(self.ring_status == (goal_idx + 2))
                if self.holding_goal_index != goal_idx and any(np.linalg.norm(goal_pos - corner) < 0.5 for corner in self.corner_positions):
                    multiplier = 2
                else:
                    multiplier = 1
                if rings_on_goal > 0:
                    score += 3 * multiplier
                    score += (rings_on_goal - 1) * multiplier
        for stake_idx in range(4):
            if np.any(self.ring_status == (stake_idx + 7)):
                rings_on_stake = np.sum(self.ring_status == (stake_idx + 7))
                if rings_on_stake > 0:
                    score += 3
                    score += (rings_on_stake - 1)
        if self.climbed:
            score += 3
        self.total_score = score
        return score

    # -----------------------------------------------------------------------------
    # Calculate reward based on score change and penalty.
    # -----------------------------------------------------------------------------
    def reward_function(self, initial_score, initial_time_remaining, penalty):
        new_score = self.compute_field_score()
        delta_score = new_score - initial_score
        reward = delta_score + penalty / 100
        return reward
    
    def calculate_path(self, start_point, end_point):
        sp = np.array(start_point, dtype=np.float64)
        ep = np.array(end_point, dtype=np.float64)
        sol = self.path_planner.Solve(start_point=sp, end_point=ep, obstacles=self.permanent_obstacles)
        return self.path_planner.getPath(sol)

    def clearAuton(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        with open(f'{self.save_path}/auton.csv', 'w') as f:
            f.write("")
    
    def clearPNGs(self):
        if os.path.exists(self.steps_save_path):
            for file in os.listdir(self.steps_save_path):
                file_path = os.path.join(self.steps_save_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
        else:
            os.makedirs(self.steps_save_path, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Create a visual representation of the environment.
    # -----------------------------------------------------------------------------
    def render(self, step_num=0, action=None, reward=None):
        if isinstance(action, (np.ndarray,)):
            action = int(action)
        
        # Print step info to console
        action_str = Actions(action).name
        print(f"Step: {step_num:<3} | {action_str:<25} | {('Valid' if self.last_action_success else 'Invalid'):<8} | Reward: {reward:<5} | Score: {self.total_score:<3} | Time: {self.time_remaining:<7.2f}")
        
        # Generate path
        if not np.array_equal(self.robot_position, self.last_robot_position):
            planned_x, planned_y, time = self.calculate_path(self.last_robot_position / ENV_FIELD_SIZE, self.robot_position / ENV_FIELD_SIZE)
            planned_x *= ENV_FIELD_SIZE
            planned_y *= ENV_FIELD_SIZE
        else:
            planned_x = [self.robot_position[0]]
            planned_y = [self.robot_position[1]]

        # Add data to CSV
        FORWARD = "FORWARD"
        BACKWARD = "BACKWARD"

        if self.last_action_success:
            with open(f'{self.save_path}/auton.csv', 'a') as f:
                if Actions.PICK_UP_NEAREST_GOAL.value == action or Actions.PICK_UP_NEXT_NEAREST_GOAL.value == action:
                    f.write(f"{BACKWARD}, ")
                    for x, y in zip(planned_x, planned_y):
                        f.write(f"{x:.2f},{y:.2f}, ")
                    f.write(f"\n")
                    f.write(f"PICKUP_GOAL\n")
                if Actions.PICK_UP_NEAREST_RING.value == action:
                    f.write(f"{FORWARD}, ")
                    for x, y in zip(planned_x, planned_y):
                        f.write(f"{x:.2f},{y:.2f}, ")
                    f.write(f"\n")
                    f.write(f"PICKUP_RING\n")
                if Actions.CLIMB.value == action:
                    f.write("CLIMB\n")
                if Actions.DROP_GOAL.value == action:
                    f.write("DROP_GOAL\n")
                if Actions.DRIVE_TO_CORNER_BL.value <= action <= Actions.DRIVE_TO_CORNER_TR.value:
                    f.write(f"{BACKWARD}, ")
                    for x, y in zip(planned_x, planned_y):
                        f.write(f"{x:.2f},{y:.2f}, ")
                    f.write(f"\n")
                    f.write(f"TURN_TO, {self.robot_orientation:.2f}\n")
                if Actions.ADD_RING_TO_GOAL.value == action:
                    f.write("ADD_RING_TO_GOAL\n")
                if Actions.DRIVE_TO_WALL_STAKE_L.value <= action <= Actions.DRIVE_TO_WALL_STAKE_T.value:
                    f.write(f"{BACKWARD}, ")
                    for x, y in zip(planned_x, planned_y):
                        f.write(f"{x:.2f},{y:.2f}, ")
                    f.write(f"\n")
                    f.write(f"TURN_TO, {self.robot_orientation:.2f}\n")
                if Actions.ADD_RING_TO_WALL_STAKE.value == action:
                    f.write("ADD_RING_TO_WALL_STAKE\n")
                if Actions.TURN_TOWARDS_CENTER.value == action:
                    pass # Don't write anything to the auton, this action is irrevelant for pre-planned routes
                    # f.write(f"TURN_TO, {self.robot_orientation:.2f}\n")

        # Create visualization
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_xlim(0, ENV_FIELD_SIZE)
        ax.set_ylim(0, ENV_FIELD_SIZE)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
        robot = patches.Rectangle(self.robot_position - np.array([self.robot_length / 24, self.robot_width / 24]), 
              self.robot_length / ENV_FIELD_SIZE, self.robot_width / ENV_FIELD_SIZE, color='blue', alpha=0.25, ec='black', transform=ax.transData)
        t = patches.transforms.Affine2D().rotate_deg_around(self.robot_position[0], self.robot_position[1], np.degrees(self.robot_orientation)) + ax.transData
        robot.set_transform(t)
        ax.add_patch(robot)
        center = self.robot_position
        arrow_dx = np.cos(self.robot_orientation)
        arrow_dy = np.sin(self.robot_orientation)
        orientation_arrow = patches.FancyArrow(center[0], center[1], arrow_dx, arrow_dy,
                width=0.1, color='yellow', length_includes_head=True, alpha=0.25)
        ax.add_patch(orientation_arrow)

        ax.plot(planned_x, planned_y, 'k--', alpha=0.5)


        for obstacle in self.permanent_obstacles:
            circle = patches.Circle(
                (obstacle.x * ENV_FIELD_SIZE, obstacle.y * ENV_FIELD_SIZE), 
                obstacle.radius * ENV_FIELD_SIZE, 
                edgecolor='black', facecolor='none', 
                linestyle='dotted', alpha=0.5)
            ax.add_patch(circle)

        for goal_idx, (goal, available) in enumerate(zip(self.mobile_goal_positions, self.goal_available)):
            color = 'green' if self.is_visible(goal) else 'gray'
            hexagon = patches.RegularPolygon(goal, numVertices=6, radius=0.5, orientation=np.pi/6, color=color, alpha=0.25)
            ax.add_patch(hexagon)
            rings_on_goal = np.sum(self.ring_status == (goal_idx + 2))
            ax.text(goal[0], goal[1] + 0.6, str(rings_on_goal), color='black', ha='center')
        wall_stakes_positions = [
            np.array([0, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE, ENV_FIELD_SIZE/2]),
            np.array([ENV_FIELD_SIZE/2, 0]),
            np.array([ENV_FIELD_SIZE/2, ENV_FIELD_SIZE])
        ]
        for i, ring in enumerate(self.ring_positions):
            status = self.ring_status[i]
            if status == 0:
                pos = ring
            elif status == 1:
                pos = self.robot_position
            elif status >= 7:
                stake_idx = status - 7
                pos = self.wall_stakes_positions[stake_idx]
            else:
                goal_idx = status - 2
                pos = self.mobile_goal_positions[goal_idx]
            color = 'red' if self.is_visible(pos) else 'gray'
            circle = patches.Circle(pos, 0.3, color=color, alpha=0.7)
            ax.add_patch(circle)
        for idx, pos in enumerate(wall_stakes_positions):
            if idx == 0:
                text_pos = pos + np.array([-0.5, 0.0])
            elif idx == 1:
                text_pos = pos + np.array([0.5, 0.0])
            elif idx == 2:
                text_pos = pos + np.array([0.0, -0.5])
            elif idx == 3:
                text_pos = pos + np.array([0.0, 0.5])
            ax.text(text_pos[0], text_pos[1], f"{self.wall_stakes[idx]}", color='black', ha='center')
        ax.text(self.robot_position[0], self.robot_position[1], f"{self.holding_rings}", color='black', ha='center')
        ax.text(-2.5, 6, f"Total Score: {self.total_score}", color='black', ha='center')
        ax.text(6, 13.25, f'Step {step_num}', color='black', ha='center')
        ax.text(6, -1.25, f'{action_str}', color='black', ha='center')
        
        if self.realistic_vision:
            fov_length = 17
            left_fov_angle = self.robot_orientation + np.pi / 2
            right_fov_angle = self.robot_orientation - np.pi / 2
            left_fov_end = self.robot_position + fov_length * np.array([np.cos(left_fov_angle), np.sin(left_fov_angle)])
            right_fov_end = self.robot_position + fov_length * np.array([np.cos(right_fov_angle), np.sin(right_fov_angle)])
            ax.plot([self.robot_position[0], left_fov_end[0]], [self.robot_position[1], left_fov_end[1]], 'k--', alpha=0.5)
            ax.plot([self.robot_position[0], right_fov_end[0]], [self.robot_position[1], right_fov_end[1]], 'k--', alpha=0.5)
            overlay_polygon = np.array([
                self.robot_position,
                left_fov_end,
                left_fov_end - 100 * np.array([np.cos(self.robot_orientation), np.sin(self.robot_orientation)]),
                right_fov_end - 100 * np.array([np.cos(self.robot_orientation), np.sin(self.robot_orientation)]),
                right_fov_end,
                self.robot_position
            ])
            overlay = patches.Polygon(overlay_polygon, closed=True, color='gray', alpha=0.3)
            ax.add_patch(overlay)


        plt.savefig(f"{self.steps_save_path}/step_{step_num}.png")
        plt.close()